import fitz  # PyMuPDF

def pdf_to_text(path: str) -> str:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    doc.close()
    return text