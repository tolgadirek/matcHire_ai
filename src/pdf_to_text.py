import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

def pdf_to_text(path: str) -> str:
    """PDF dosyasından metin çıkarır. Gerekirse OCR uygular."""
    text = ""
    doc = fitz.open(path)

    for page in doc:
        # Sayfadan doğrudan metin almayı dene
        page_text = page.get_text("text")
        if page_text.strip():
            text += page_text
        else:
            # Eğer metin yoksa OCR uygula (tarama PDF'ler için)
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text += pytesseract.image_to_string(img)

    doc.close()
    return text
