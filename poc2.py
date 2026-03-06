import streamlit as st
import PyPDF2
import docx
import re
from spellchecker import SpellChecker

# --- AGNO IMPORTS ---
from agno.agent import Agent
from agno.models.groq import Groq
from agno.workflow import Step, Workflow
from prompts import get_formatter_instruction, get_reviewer_instruction, get_poc2_reconstruction_prompt

# --- PAGE CONFIG ---
st.set_page_config(page_title="Multi-Agent Manuscript Formatter", layout="wide")

# --- EXTRACTION HELPERS ---
def extract_from_pdf(file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted: text += extracted + "\n"
    except Exception as e: st.error(f"Error reading PDF: {e}")
    return text

def extract_from_docx(file):
    text = ""
    try:
        doc = docx.Document(file)
        for para in doc.paragraphs: text += para.text + "\n"
    except Exception as e: st.error(f"Error reading DOCX: {e}")
    return text

def extract_from_tex(file):
    try: return file.read().decode('utf-8')
    except Exception as e:
        st.error(f"Error reading TEX: {e}")
        return ""

def extract_text(uploaded_file):
    filename = uploaded_file.name.lower()
    if filename.endswith('.pdf'): return extract_from_pdf(uploaded_file)
    elif filename.endswith('.docx'): return extract_from_docx(uploaded_file)
    elif filename.endswith('.tex'): return extract_from_tex(uploaded_file)
    return ""

def check_spelling(text):
    spell = SpellChecker()
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    misspelled = {word for word in spell.unknown(words) if len(word) > 2}
    return list(misspelled)

def chunk_text(text, max_words=800):
    """Splits text into chunks of roughly max_words, preserving paragraph boundaries to prevent LLM truncation."""
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

# --- AGNO MULTI-AGENT PIPELINE (VIA GROQ) ---
def reconstruct_with_agno(text, format_style, format_rules, api_key, selected_model):
    """Uses an Agno Workflow with sequential Steps to reconstruct the manuscript via Groq."""

    import os
    os.environ["GROQ_API_KEY"] = api_key

    groq_model = Groq(id=selected_model)

    # Step 1: Formatter Agent
    formatter_agent = Agent(
        name="FormatterAgent",
        model=groq_model,
        instructions=get_formatter_instruction(format_style, format_rules),
        markdown=False,
    )

    formatter_step = Step(
        name="Formatting",
        agent=formatter_agent,
        description=f"Format raw text into a structured {format_style} manuscript",
    )

    # Step 2: Reviewer Agent
    reviewer_agent = Agent(
        name="ReviewerAgent",
        model=groq_model,
        instructions=get_reviewer_instruction(format_style),
        markdown=False,
    )

    reviewer_step = Step(
        name="Reviewing",
        agent=reviewer_agent,
        description="Review and refine the manuscript's citations and structure",
    )

    # Step 3: Markdown Corrector Agent
    markdown_formatter_agent = Agent(
        name="MarkdownFormatterAgent",
        model=groq_model,
        instructions=f"""\
You are a world-class Markdown Formatter. You receive Markdown code that may contain structural errors, formatting inconsistencies, or styling issues. Your SOLE purpose is to reformat and correct the Markdown code so it is clean and adheres to {format_style} publication standards.

### FORMATTING & CORRECTION RULES
1. Ensure proper heading hierarchy (# for main title, ## for sections, ### for subsections).
2. Format citations dynamically per {format_style}.
3. Remove ALL dummy/placeholder text (e.g., "Lorem ipsum", "TODO").
4. You are FORBIDDEN from rephrasing, summarizing, reordering, or omitting ANY original manuscript content.
5. Return ONLY the corrected Markdown text. No markdown wrappers (```), no explanations.\
""",
        markdown=False,
    )

    markdown_formatter_step = Step(
        name="MarkdownFormatting",
        agent=markdown_formatter_agent,
        description="Reformat and correct existing Markdown document into clean, publication-ready output",
    )

    # Step 4: LaTeX Converter Agent
    latex_converter_agent = Agent(
        name="LaTeXConverterAgent",
        model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"),
        instructions=f"""\
You are an expert LaTeX document converter. You receive a Markdown manuscript and just only convert it into a complete, compilable LaTeX document formatted for {format_style} journal submission as it in markdown no any other changes just convert it to as it is in latex using {format_style} format and it should be complete transformation as it is in latex as in markdown.
""",
        markdown=False,
    )

    latex_converter_step = Step(
        name="LaTeXConversion",
        agent=latex_converter_agent,
        description=f"Convert the polished Markdown manuscript into a compilable {format_style} LaTeX document",
    )

    # Orchestrator: Sequential Workflow for Markdown Generation
    markdown_pipeline = Workflow(
        name="MarkdownPipeline",
        description="Three-step pipeline: Format → Review → Polish",
        steps=[formatter_step, reviewer_step, markdown_formatter_step],
    )

    try:
        chunks = chunk_text(text, max_words=800)
        final_doc = []
        progress_bar = st.progress(0)

        for i, chunk in enumerate(chunks):
            prompt = get_poc2_reconstruction_prompt(chunk, i, len(chunks))
            result = markdown_pipeline.run(prompt)
            chunk_content = result.content.strip()
            
            # Clean markdown wrappers if present
            if chunk_content.startswith("```") and chunk_content.endswith("```"):
                chunk_content = re.sub(r"^```[a-zA-Z]*\n?(.*?)```$", r"\1", chunk_content, flags=re.DOTALL)
                chunk_content = chunk_content.strip()
                
            final_doc.append(chunk_content)
            progress_bar.progress((i + 1) / len(chunks))

        progress_bar.empty()
        
        # --- Final global Markdown cleanup pass ---
        merged_markdown = "\n\n".join(final_doc)
        merged_markdown = _global_markdown_cleanup(merged_markdown)
        
        # --- Run LaTeX Conversion Step 4 on the globally cleaned Markdown ---
        latex_prompt = f"Convert the following Markdown document to {format_style} LaTeX format strictly adhering to the rules:\n\n{merged_markdown}"
        latex_result = latex_converter_agent.run(latex_prompt)
        merged_latex = latex_result.content.strip()

        # Clean markdown wrappers around LaTeX if present
        if merged_latex.startswith("```") and merged_latex.endswith("```"):
            merged_latex = re.sub(r"^```[a-zA-Z]*\n?(.*?)```$", r"\1", merged_latex, flags=re.DOTALL)
            merged_latex = merged_latex.strip()

        return merged_markdown, merged_latex
    except Exception as e:
        return None, f"Agno/Groq Error: {e}"

def _global_markdown_cleanup(md_code):
    """Rule-based post-processing to manage chunk artifacts."""
    
    # Remove duplicate Reference section headers
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
    
    # Remove common placeholder/dummy text patterns
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


# --- MAIN APP UI ---
st.title("📄 Multi-Agent Manuscript Formatter")
st.markdown("Upload a document to spell-check, evaluate, and reconstruct it into your target format.")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Groq API Key", type="password")

    groq_model = st.selectbox(
        "Select Groq Model",
        ["meta-llama/llama-4-maverick-17b-128e-instruct", "qwen/qwen3-32b", "openai/gpt-oss-120b"]
    )

    format_style = st.selectbox(
        "Target Formatting Style",
        ["IEEE", "Vancouver", "APA", "MLA", "Chicago"]
    )
    
    # Predefined rules mapping
    default_rules = {
        "IEEE": "Use IEEE standard conventions: Numbered citations in square brackets (e.g., [1]). Create a 'References' section at the end. Use a 2-column format layout mentally when structuring text blocks.",
        "Vancouver": "Use Vancouver style conventions: Numbered citations in superscript or brackets in order of appearance. Create a numbered 'References' list at the very end.",
        "APA": "Use APA 7th Edition conventions: Author-date citations (Author, Year). Create exactly one 'References' section at the end, formatted with a hanging indent.",
        "MLA": "Use MLA 9th Edition conventions: Parenthetical citations (Author Page). Create exactly one 'Works Cited' section at the end.",
        "Chicago": "Use Chicago Manual of Style conventions: Footnote or endnote citations. Create exactly one 'Bibliography' section at the end.",
    }
    
    st.subheader("📝 Formatting Rules")
    st.markdown("Edit these rules instructions to customize how the AI formats the document:")
    user_format_rules = st.text_area(
        label=f"Rules for {format_style}", 
        value=default_rules.get(format_style, "No predefined rules."),
        height=150
    )

uploaded_file = st.file_uploader("Upload Manuscript", type=["pdf", "docx", "tex"])

if uploaded_file is not None:
    with st.spinner("Extracting text..."):
        raw_text = extract_text(uploaded_file)

    if raw_text.strip():
        st.success("Text extracted successfully!")

        tab1, tab2, tab3 = st.tabs(["Original Text", "Spell Check", "Reconstruct & Download"])

        with tab1:
            st.subheader("Extracted Manuscript Text")
            st.text_area("Preview", raw_text[:2000] + "\n\n...[Text Truncated]...", height=400)

        with tab2:
            st.subheader("Spell Check Results")
            with st.spinner("Running spell checker..."):
                misspelled_words = check_spelling(raw_text)
                if misspelled_words:
                    st.warning(f"Found {len(misspelled_words)} potential spelling errors.")
                    st.write(", ".join(sorted(misspelled_words)))
                else:
                    st.success("No spelling errors detected!")

        with tab3:
            st.subheader(f"Reconstruct into {format_style} Format")
            if not api_key:
                st.error("Please enter a Groq API Key in the sidebar.")
            else:
                if st.button("Generate Structured Manuscript"):
                    with st.spinner(f"Agents are working (Format → Review → Polish → LaTeX) using {groq_model}..."):
                        structured_md, structured_tex = reconstruct_with_agno(raw_text, format_style, user_format_rules, api_key, groq_model)

                        if structured_md is None:
                            st.error(structured_tex)  # Contains error message
                        else:
                            st.success(f"Manuscript successfully reconstructed into Markdown & LaTeX ({format_style})!")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    label="📥 Download Markdown Version",
                                    data=structured_md,
                                    file_name=f"reconstructed_{format_style.lower()}.md",
                                    mime="text/markdown",
                                    use_container_width=True
                                )
                            with col2:
                                st.download_button(
                                    label="📥 Download LaTeX Version",
                                    data=structured_tex,
                                    file_name=f"reconstructed_{format_style.lower()}.tex",
                                    mime="application/x-tex",
                                    use_container_width=True
                                )
                            
                            st.markdown("### Document Preview (LaTeX Source)")
                            st.code(structured_tex, language="latex")