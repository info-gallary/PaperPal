import streamlit as st
import PyPDF2
import docx
import re
from spellchecker import SpellChecker
import os

# --- GOOGLE GENAI & ADK IMPORTS ---
from google.genai import types
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from prompts import get_formatter_instruction, get_reviewer_instruction, get_poc1_reconstruction_prompt

# --- PAGE CONFIG ---
st.set_page_config(page_title="Multi-Agent Manuscript Formatter", layout="wide")

# --- EXTRACTION ROUTERS ---
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

# --- ADK MULTI-AGENT PIPELINE ---
def reconstruct_with_adk(text, format_style, api_key):
    """Uses a Sequential Multi-Agent pipeline to reconstruct the manuscript."""
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Agent 1: The Formatter
    formatter_agent = LlmAgent(
        model='gemini-2.5-flash',
        name="FormatterAgent",
        description="Formats raw text into a structured manuscript.",
        instruction=f"""
        You are an expert academic typesetter and data recovery specialist. 
        You will be provided with unstructured, raw extracted text from a document.
        Your task is to comprehensively reconstruct this text into a properly formatted manuscript strictly according to {format_style} guidelines.
        CRITICAL RULES:
        1. FORMATTING CONSTRAINT: You must FORMAT the output STRICTLY and ONLY in {format_style}. Do not generate any other formats.
        2. ZERO DATA LOSS: You must retain absolutely 100% of the original text content. Every sentence and word must be preserved. Do not summarize, compress, or omit any information whatsoever.
        3. RECONSTRUCT STRUCTURE: Identify and restore headers, paragraphs, lists, and sections from the unstructured raw text to make it readable and logically structured.
        4. EXACT CITATIONS & DATA: Keep every citation, number, equation, and terminology exactly as they appear. Only reformat the citations to match {format_style}.
        5. NO HALLUCINATIONS: Do not add any conversational filler, introductory text, or concluding remarks. Output ONLY the meticulously reconstructed LaTeX code.
        6. COMPLETE LATEX DOCUMENT: Ensure the final output is a single, complete, compilable LaTeX file with \\documentclass, preamble, \\begin{document}, and \\end{document}. Do not wrap the code in markdown code blocks.
        """
    )
    
    # Agent 2: The Reviewer
    reviewer_agent = LlmAgent(
        model='gemini-2.5-flash',
        name="ReviewerAgent",
        description="Reviews and refines the manuscript's citations and structure.",
        instruction=f"""
        You are a strict academic editor, proofreader, and verifier.
        Review the newly reconstructed formatted text to ensure it perfectly aligns with {format_style} standards while guaranteeing absolutely zero data loss from the original unstructured text.
        CRITICAL RULES:
        1. STRICT VERIFICATION: You must verify that no data, paragraphs, sentences, or citations from the original raw input were omitted or abridged in the formatting process.
        2. NO HALLUCINATION: Ensure no new text, citations, or references were added by the formatter.
        3. FORMAT REFINEMENT: Fix any structural deviations, hallucinated citations, or layout errors to strictly match {format_style} guidelines.
        4. COMPLETE LATEX INTACT: Ensure the final output remains a single, complete, compilable LaTeX file with \\documentclass, preamble, \\begin{document}, and \\end{document}. Do not wrap the code in markdown code blocks.
        Output ONLY the final, pristine LaTeX code ready for compilation without any conversational filler.
        """
    )
    
    # Orchestrator
    pipeline = SequentialAgent(
        name="ManuscriptPipeline",
        sub_agents=[formatter_agent, reviewer_agent]
    )
    
    # Execution Environment
    runner = Runner(
        app_name="ManuscriptApp",
        agent=pipeline, 
        session_service=InMemorySessionService(),
        auto_create_session=True
    )
    
    try:
        # Format the string prompt into an ADK Content object
        prompt = get_poc1_reconstruction_prompt(text)
        content = types.Content(
            role='user', 
            parts=[types.Part.from_text(text=prompt)]
        )
        
        # Call runner.run() using strict keyword arguments
        final_result = ""
        for event in runner.run(
            user_id="user_1", 
            session_id="session_1", 
            new_message=content
        ):
            # Catch the final event in the stream and extract the text
            if event.is_final_response() and event.content and event.content.parts:
                final_result = event.content.parts[0].text
                
        # Clean markdown wrappers if present
        final_result = final_result.strip()
        if final_result.startswith("```") and final_result.endswith("```"):
            final_result = re.sub(r"^```[a-zA-Z]*\n?(.*?)```$", r"\1", final_result, flags=re.DOTALL)
            final_result = final_result.strip()
            
        return final_result
        
    except Exception as e:
        return f"ADK Error: {e}"

# --- MAIN APP UI ---
st.title("📄 Multi-Agent Manuscript Formatter")
st.markdown("Upload a document to spell-check, evaluate, and reconstruct it into your target format.")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Google Gemini API Key", type="password")
    format_style = st.selectbox(
        "Target Formatting Style", 
        ["IEEE", "Vancouver", "APA", "MLA", "Chicago"]
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
                st.error("Please enter a Gemini API Key in the sidebar to run the ADK agents.")
            else:
                if st.button("Generate Structured Manuscript"):
                    with st.spinner("Agents are working... Formatter is organizing, Reviewer is checking..."):
                        structured_doc = reconstruct_with_adk(raw_text, format_style, api_key)
                        
                        if "ADK Error" in structured_doc:
                            st.error(structured_doc)
                        else:
                            st.success("Manuscript successfully reconstructed!")
                            
                            # Provide the Download Button
                            st.download_button(
                                label="📥 Download Structured Manuscript (Markdown)",
                                data=structured_doc,
                                file_name=f"reconstructed_{format_style.lower()}.md",
                                mime="text/markdown"
                            )
                            
                            # Preview the final output
                            st.markdown("### Document Preview")
                            st.markdown(structured_doc)