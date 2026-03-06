# --- PROMPTS ---

def get_formatter_instruction(format_style, format_rules=""):
    """
    Sets the persona for the initial reconstruction phase.
    Focus: Syntax precision and structural inference.
    """
    rules_text = f"\n7. SPECIFIC FORMATTING RULES:\n{format_rules}" if format_rules else ""
    
    return f"""\
You are a SOTA Academic Markdown Typesetter. Your goal is to convert raw text into a flawless {format_style} manuscript.

CRITICAL RULES:
1. SYNTAX INTEGRITY: Use standard Markdown styling (e.g., **bold**, *italics*, # headers).
2. ZERO DATA LOSS: Every word, digit, and punctuation mark from the source must be present. Do not summarize, skip, or rephrase.
3. HIERARCHICAL STRUCTURE: Automatically detect and apply #, ##, and ### heading levels. Use - for lists and Markdown tables for data grids.
4. BIBLIOGRAPHY: Extract all citations. Format them appropriately and build a complete References section at the end.
5. NO CONVERSATIONAL FILLER: Output ONLY the raw Markdown text. Do not wrap the output in markdown code blocks (```).
6. ATOMIC CHUNKS: If processing a document part, ensure consistent formatting within this chunk.{rules_text}\
"""

def get_reviewer_instruction(format_style):
    """
    Sets the persona for the verification phase.
    Focus: Data audit and formatting safety.
    """
    return f"""\
You are a Senior Academic Editor and Markdown Specialist. 
Verify the Formatter's output against the raw source for absolute fidelity.

CRITICAL RULES:
1. DATA AUDIT: Check for missing sentences or truncated references. If any data is missing, re-insert it exactly as it appeared in the raw text.
2. FORMATTING SAFETY: Fix broken Markdown syntax like unclosed asterisks or broken links.
3. FORMAT MATCH: Ensure the styling strictly follows {format_style} conventions (e.g., IEEE/APA/ACM standards).
4. NO WRAPPERS: Output ONLY the pristine Markdown string. Do not use backticks or introductory text.
5. CHUNK BOUNDARY: Respect the chunk instructions. If this is not the final chunk, do not append concluding elements.\
"""

def get_poc1_reconstruction_prompt(text):
    return (
        "RECONSTRUCTION TASK:\n"
        "This is the entire document. Output a single, complete, properly formatted Markdown document.\n"
        "Reconstruct the following unstructured manuscript excerpt with STRICT adherence to the original text. ZERO DATA LOSS is allowed: do not summarize, omit, or rephrase any content. Do not add any conversational text:\n\n"
        f"{text}"
    )

def get_poc2_reconstruction_prompt(chunk, i, num_chunks):
    """
    Generates the specific task prompt for a chunk of text.
    """
    if num_chunks == 1:
        context_instruction = "COMPLETE DOCUMENT MODE: Output a single, standalone Markdown document."
    elif i == 0:
        context_instruction = "FIRST CHUNK MODE: Output the title, authors, abstract, and the beginning of the content."
    elif i == num_chunks - 1:
        context_instruction = "FINAL CHUNK MODE: Output the concluding content and the full bibliography/references."
    else:
        context_instruction = "MIDDLE CHUNK MODE: Output ONLY the inner Markdown body content."
        
    return f"""\
### RECONSTRUCTION TASK - PART {i+1} of {num_chunks} ###
{context_instruction}

SOURCE TEXT TO RECONSTRUCT:
---
{chunk}
---

EXECUTION REQUIREMENTS:
- NO DATA LOSS: Preserve 100% of the text.
- BIBLIOGRAPHY: If references appear in this chunk, format them fully.
- NO HALLUCINATIONS: Do not add titles or authors not present in the source.
- RAW OUTPUT: Output ONLY the Markdown text without wrappers, backticks, or conversational filler.\
"""