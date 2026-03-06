import streamlit as st
import PyPDF2
import docx
import re
from spellchecker import SpellChecker
import google.generativeai as genai

# --- PAGE CONFIG ---
st.set_page_config(page_title="Manuscript Evaluator PoC", layout="wide")

# --- FILE EXTRACTION ROUTERS ---
def extract_from_pdf(file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

def extract_from_docx(file):
    text = ""
    try:
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
    return text

def extract_from_tex(file):
    try:
        # .tex files are plain text, so we read and decode the bytes
        text = file.read().decode('utf-8')
        return text
    except Exception as e:
        st.error(f"Error reading TEX file: {e}")
        return ""

def extract_text(uploaded_file):
    """Routes the file to the correct extractor based on extension."""
    filename = uploaded_file.name.lower()
    if filename.endswith('.pdf'):
        return extract_from_pdf(uploaded_file)
    elif filename.endswith('.docx'):
        return extract_from_docx(uploaded_file)
    elif filename.endswith('.tex'):
        return extract_from_tex(uploaded_file)
    else:
        st.error("Unsupported file format.")
        return ""

# --- EVALUATION FUNCTIONS ---
def check_spelling(text):
    """Finds misspelled words using pyspellchecker."""
    spell = SpellChecker()
    # Clean text to get words only
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    misspelled = spell.unknown(words)
    misspelled = {word for word in misspelled if len(word) > 2}
    return list(misspelled)

def evaluate_formatting_with_llm(text, format_style, api_key):
    """Uses Gemini LLM to evaluate citations and document structure."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')    
    prompt = f"""
    You are an expert academic editor. Review the following excerpt from a research manuscript.
    The target formatting and citation style is: {format_style}.
    
    Please evaluate the text based on:
    1. Proper Citation Style: Are the in-text citations and reference list adhering strictly to {format_style} guidelines?
    2. Structured Documentation: Does the text follow standard structural conventions for this format?
    
    Identify specific errors and provide actionable feedback on how to fix them.
    If the text contains LaTeX markup, evaluate the formatting intent based on the tags used.
    
    Manuscript Text Excerpt:
    {text[:5000]} # Limiting to 5000 chars for the PoC
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error communicating with LLM: {e}"

# --- MAIN APP UI ---
st.title("📄 Multi-Format Manuscript Evaluator")
st.markdown("Upload a PDF, Word document (.docx), or LaTeX file (.tex) for evaluation.")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Google Gemini API Key", type="password")
    format_style = st.selectbox(
        "Target Formatting Style", 
        ["IEEE", "Vancouver", "APA", "MLA", "Chicago"]
    )

# Updated file uploader to accept multiple formats
uploaded_file = st.file_uploader("Upload Manuscript", type=["pdf", "docx", "tex"])

if uploaded_file is not None:
    with st.spinner("Extracting text..."):
        raw_text = extract_text(uploaded_file)
        
    if not raw_text.strip():
        st.warning("Could not extract text from this file.")
    else:
        st.success("Text extracted successfully!")
        
        tab1, tab2, tab3 = st.tabs(["Original Text", "Spell Check", "Format & Citation Analysis"])
        
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
                st.info("Note: Scientific terms, author names, and LaTeX formatting tags may be flagged.")
            else:
                st.success("No spelling errors detected!")
                
        with tab3:
            st.subheader(f"Evaluation against {format_style} Guidelines")
            if not api_key:
                st.error("Please enter a Gemini API Key in the sidebar to run the format analysis.")
            else:
                if st.button("Run Format Analysis"):
                    with st.spinner("Analyzing citations and structure via LLM..."):
                        feedback = evaluate_formatting_with_llm(raw_text, format_style, api_key)
                        st.markdown(feedback)