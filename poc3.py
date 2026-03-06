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
    """Splits text into chunks of roughly max_words, preserving paragraph boundaries."""
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
def reconstruct_with_agno(text, format_style, api_key, selected_model):
    """Uses an Agno Workflow with 3 sequential Steps: Format → Review → LaTeX Correction."""

    import os
    os.environ["GROQ_API_KEY"] = api_key

    groq_model = Groq(id=selected_model)
    llama33_model = Groq(id="openai/gpt-oss-20b")

    # --- Step 1: Formatter Agent ---
    formatter_agent = Agent(
        name="FormatterAgent",
        model=groq_model,
        instructions=get_formatter_instruction(format_style),
        markdown=False,
    )

    formatter_step = Step(
        name="Formatting",
        agent=formatter_agent,
        description=f"Format raw text into a structured {format_style} manuscript",
    )

    # --- Step 2: Reviewer Agent ---
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

    # --- Step 3: Markdown Corrector Agent (LLaMA 3.3) ---
    markdown_formatter_agent = Agent(
        name="MarkdownFormatterAgent",
        model=llama33_model,
        instructions=f"""\
You are a world-class Markdown Formatter. You receive Markdown code that may contain structural errors, formatting inconsistencies, or styling issues. Your SOLE purpose is to reformat and correct the Markdown code so it is clean and adheres to {format_style} publication standards.

### FORMATTING & CORRECTION RULES

**A. DOCUMENT SKELETON**
1. Ensure proper heading hierarchy (# for main title, ## for sections, ### for subsections).
2. Format citations dynamically per {format_style}.

**B. CONTENT DEDUPLICATION**
3. If the same section or paragraph block appears more than once with identical or near-identical content, keep ONLY the first complete occurrence and remove all duplicates.
4. Remove duplicate entries in the bibliography.

**C. SPECIAL CHARACTER & MATH MODE**
5. Ensure all mathematical expressions are properly formatted.

**D. PLACEHOLDER REMOVAL**
6. Remove ALL dummy/placeholder text (e.g., "Lorem ipsum", "Insert Title Here", "Author Name", "TODO").

**E. ABSOLUTE CONTENT FIDELITY**
7. You are FORBIDDEN from rephrasing, summarizing, reordering, or omitting ANY original manuscript content. You may ONLY modify Markdown formatting.

**F. BIBLIOGRAPHY MERGING**
8. If multiple reference sections or bibliographies are provided, merge them into exactly ONE Reference section.

**OUTPUT REQUIREMENT:** Return ONLY the corrected, properly formatted Markdown text. No markdown wrappers (```), no explanations. The output must be directly saveable as a .md file.\
""",
        markdown=False,
    )

    markdown_formatter_step = Step(
        name="MarkdownFormatting",
        agent=markdown_formatter_agent,
        description="Reformat and correct existing Markdown document into clean, publication-ready output",
    )

    # --- Orchestrator: 3-Step Sequential Workflow ---
    pipeline = Workflow(
        name="ManuscriptPipeline",
        description="Three-step pipeline: Format → Review → Markdown Formatting",
        steps=[formatter_step, reviewer_step, markdown_formatter_step],
    )

    try:
        chunks = chunk_text(text, max_words=800)
        final_doc = []
        progress_bar = st.progress(0)

        for i, chunk in enumerate(chunks):
            prompt = get_poc2_reconstruction_prompt(chunk, i, len(chunks))
            result = pipeline.run(prompt)
            chunk_content = result.content.strip()

            # Clean markdown wrappers if present
            if chunk_content.startswith("```") and chunk_content.endswith("```"):
                chunk_content = re.sub(
                    r"^```[a-zA-Z]*\n?(.*?)```$", r"\1", chunk_content, flags=re.DOTALL
                )
                chunk_content = chunk_content.strip()

            final_doc.append(chunk_content)
            progress_bar.progress((i + 1) / len(chunks))

        progress_bar.empty()

        # --- Final global cleanup pass ---
        merged = "\n\n".join(final_doc)
        # merged = _global_latex_cleanup(merged)

        return merged
    except Exception as e:
        return f"Agno/Groq Error: {e}"




# --- MAIN APP UI ---
st.title("📄 Multi-Agent Manuscript Formatter")
st.markdown("Upload a document to spell-check, evaluate, and reconstruct it into your target format.")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Groq API Key", type="password")

    groq_model = st.selectbox(
        "Select Groq Model (Formatter & Reviewer)",
        [
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "qwen/qwen3-32b",
            "openai/gpt-oss-120b",
        ],
    )

    st.info("Markdown Corrector uses **LLaMA 3.3 70B** automatically.")

    format_style = st.selectbox(
        "Target Formatting Style",
        ["IEEE", "Vancouver", "APA", "MLA", "Chicago"],
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
                    with st.spinner(f"Pipeline: {groq_model} → LLaMA 3.3 Corrector..."):
                        structured_doc = reconstruct_with_agno(
                            raw_text, format_style, api_key, groq_model
                        )

                        if "Agno/Groq Error" in structured_doc:
                            st.error(structured_doc)
                        else:
                            st.success("Manuscript successfully reconstructed & corrected!")
                            st.download_button(
                                label="📥 Download Structured Manuscript (Markdown)",
                                data=structured_doc,
                                file_name=f"reconstructed_{format_style.lower()}.md",
                                mime="text/markdown",
                            )
                            st.markdown("### Document Preview")
                            st.markdown(structured_doc)