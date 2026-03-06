import os
import io
import re
import json
import asyncio
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import docx
from spellchecker import SpellChecker

from agno.agent import Agent
from agno.models.groq import Groq as AgnoGroq
from agno.workflow import Step, Workflow
from prompts import get_formatter_instruction, get_reviewer_instruction, get_poc2_reconstruction_prompt

app = FastAPI(
    title="Manuscript Extraction & Formatting Streaming API",
    description="Streaming API for Agno-powered document reconstruction.",
    version="2.0.0"
)

# Enable CORS for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HELPER FUNCTIONS ---
def extract_from_pdf(file_bytes: bytes) -> str:
    text = ""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted: text += extracted + "\n"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {e}")
    return text

def extract_from_docx(file_bytes: bytes) -> str:
    text = ""
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        for para in doc.paragraphs: text += para.text + "\n"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading DOCX: {e}")
    return text

def extract_from_tex(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading TEX: {e}")

def chunk_text(text, max_words=800):
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk.split()) + len(para.split()) > max_words and current_chunk.strip():
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n"
        else:
            current_chunk += para + "\n"
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

def _global_markdown_cleanup(md_code):
    lines = md_code.split('\n')
    cleaned_lines = []
    ref_header_seen = False
    for line in lines:
        lower_line = line.strip().lower()
        if lower_line in ["# references", "## references", "### references", "## works cited", "### works cited"]:
            if ref_header_seen:
                continue
            ref_header_seen = True
        cleaned_lines.append(line)
        
    md_code = '\n'.join(cleaned_lines)
    dummy_patterns = [
        r'Lorem ipsum.*?(?=\n\n|\Z)',
        r'\[Your Name\]',
        r'\[Insert .*?\]',
        r'Author Name Here',
        r'TODO:?.*?(?=\n)',
    ]
    for pattern in dummy_patterns:
        md_code = re.sub(pattern, '', md_code, flags=re.IGNORECASE)
    return md_code.strip()

# --- STREAMING ENDPOINT ---
@app.post("/api/v2/reconstruct/stream", tags=["Streaming Pipeline"], summary="Stream multi-agent manuscript formatting")
async def reconstruct_stream_endpoint(
    file: UploadFile = File(...),
    format_style: str = Form(...),
    custom_rules: str = Form(""),
    model: str = Form("meta-llama/llama-4-maverick-17b-128e-instruct")
):
    """
    Streams live logs of the Agno Multi-Agent reconstruction pipeline.
    Yields the final structured Markdown/LaTeX, errors, and chunking bifurcation.
    """
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not found in environment variables.")

    async def sse_generator():
        yield f"data: {{ \"log\": \"Starting file extraction...\" }}\n\n"
        await asyncio.sleep(0.1) # Small delay to flush to client
        
        filename = file.filename.lower()
        file_bytes = await file.read()
        if filename.endswith('.pdf'): text = extract_from_pdf(file_bytes)
        elif filename.endswith('.docx'): text = extract_from_docx(file_bytes)
        elif filename.endswith('.tex'): text = extract_from_tex(file_bytes)
        else:
            yield f"data: {{ \"error\": \"Unsupported file format. Must be pdf, docx, or tex.\" }}\n\n"
            return
            
        yield f"data: {{ \"log\": \"Running spellchecker...\" }}\n\n"
        
        spell = SpellChecker()
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        misspelled = list({word for word in spell.unknown(words) if len(word) > 2})
        yield f"data: {{ \"log\": \"Found {len(misspelled)} spelling errors.\", \"errors\": {json.dumps(misspelled)} }}\n\n"
        
        yield f"data: {{ \"log\": \"Bifurcating text into conceptual chunks...\" }}\n\n"
        chunks = chunk_text(text, max_words=800)
        yield f"data: {{ \"log\": \"Split document into {len(chunks)} chunks.\", \"total_chunks\": {len(chunks)} }}\n\n"
        await asyncio.sleep(0.1)

        # Build Agents
        groq_model = AgnoGroq(id=model, api_key=api_key)

        formatter_step = Step(
            name="Formatting",
            agent=Agent(
                name="FormatterAgent", model=groq_model,
                instructions=get_formatter_instruction(format_style, custom_rules), markdown=False
            ),
            description=f"Format raw text into a structured {format_style} manuscript"
        )
        reviewer_step = Step(
            name="Reviewing",
            agent=Agent(
                name="ReviewerAgent", model=groq_model,
                instructions=get_reviewer_instruction(format_style), markdown=False
            ),
            description="Review and refine the manuscript's citations and structure"
        )
        markdown_formatter_step = Step(
            name="MarkdownFormatting",
            agent=Agent(
                name="MarkdownFormatterAgent", model=groq_model,
                instructions=f"You are a world-class Markdown Formatter. Reformat and correct the Markdown code so it adheres to {format_style} publication standards. Ensure heading hierarchy, clear dummy text, keep ALL content. Return ONLY Markdown.", markdown=False
            ),
            description="Reformat and correct existing Markdown document into clean, publication-ready output"
        )
        latex_converter_step = Step(
            name="LaTeXConversion",
            agent=Agent(
                name="LaTeXConverterAgent", model=AgnoGroq(id="meta-llama/llama-4-maverick-17b-128e-instruct", api_key=api_key),
                instructions=f"You are an expert LaTeX document converter. You receive a Markdown manuscript and convert it into a complete, compilable LaTeX document formatted for {format_style} journal submission. RETURN ONLY RAW LATEX CODE. No conversational filler, no explanations, no 'Here is your output'. ALWAYS start with \\documentclass and end with \\end{{document}}.", markdown=False
            ),
            description=f"Convert the polished Markdown manuscript into a compilable {format_style} LaTeX document"
        )
        
        markdown_pipeline = Workflow(name="MarkdownPipeline", steps=[formatter_step, reviewer_step, markdown_formatter_step])
        
        final_doc = []
        for i, chunk in enumerate(chunks):
            yield f"data: {{ \"log\": \"Processing chunk {i+1}/{len(chunks)} through Agent sequence (Format -> Review -> Polish)...\" }}\n\n"
            
            prompt = get_poc2_reconstruction_prompt(chunk, i, len(chunks))
            
            # Since workflow.run is synchronous blocking, we run it in a threadpool to prevent blocking the event loop entirely
            result = await asyncio.to_thread(markdown_pipeline.run, prompt)
            
            chunk_content = result.content.strip()
            if chunk_content.startswith("```"):
                chunk_content = re.sub(r"^```[a-zA-Z]*\n?(.*?)```$", r"\1", chunk_content, flags=re.DOTALL).strip()
                
            final_doc.append(chunk_content)
            yield f"data: {{ \"log\": \"Chunk {i+1} successfully formatted.\" }}\n\n"
            
        yield f"data: {{ \"log\": \"Stitching and applying global Markdown cleanup...\" }}\n\n"
        merged_markdown = "\n\n".join(final_doc)
        merged_markdown = _global_markdown_cleanup(merged_markdown)
        await asyncio.sleep(0.1)

        yield f"data: {{ \"log\": \"Sending to LaTeX Converter Agent...\" }}\n\n"
        latex_agent = latex_converter_step.agent
        latex_prompt = f"Convert the following Markdown document to {format_style} LaTeX format strictly adhering to the rules:\n\n{merged_markdown}\n\nIMPORTANT: Return ONLY the raw LaTeX code. Do NOT wrap it in ```latex or ``` blocks. Start immediately with \\documentclass."
        
        latex_result = await asyncio.to_thread(latex_agent.run, latex_prompt)
        merged_latex = latex_result.content.strip()
        
        # Regex to forcefully isolate ONLY the LaTeX code block if the LLM hallucinated markdown/conversational wrappers
        latex_match = re.search(r'(\\documentclass.*?\\end\{document\})', merged_latex, re.DOTALL)
        if latex_match:
            merged_latex = latex_match.group(1)
        
        if merged_latex.startswith("```"):
            merged_latex = re.sub(r"^```[a-zA-Z]*\n?(.*?)```$", r"\1", merged_latex, flags=re.DOTALL).strip()
            
        yield f"data: {{ \"log\": \"LaTeX Compilation complete. Streaming final payload.\" }}\n\n"
        
        final_payload = {
            "is_final": True,
            "markdown": merged_markdown,
            "latex": merged_latex,
            "errors": misspelled,
            "raw_chunks": chunks 
        }
        yield f"data: {json.dumps(final_payload)}\n\n"
        
    return StreamingResponse(sse_generator(), media_type="text/event-stream")
@app.get("/")
def read_root():
    return {"message": "Welcome to the Manuscript Extraction & Formatting Streaming API"}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
