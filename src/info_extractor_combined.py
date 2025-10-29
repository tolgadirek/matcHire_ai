# -*- coding: utf-8 -*-
import re
import json
import os
import fitz  # PyMuPDF
from typing import Dict, Any
import google.generativeai as genai


# ==========================================================
# 🧩 1️⃣ PDF metni + font analizli okuma
# ==========================================================
def read_pdf_with_font_info(path: str):
    """PDF'ten metin ve font bilgisiyle satır listesi döndürür."""
    doc = fitz.open(path)
    lines_with_fonts = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            for l in b.get("lines", []):
                text = " ".join([s["text"] for s in l.get("spans", []) if s["text"].strip()])
                if not text.strip():
                    continue
                max_font = max([s["size"] for s in l["spans"]]) if l["spans"] else 10
                lines_with_fonts.append((text.strip(), max_font))
    doc.close()
    return lines_with_fonts


# ==========================================================
# 🧠 2️⃣ Başlık tabanlı gruplama
# ==========================================================
def group_sections(lines_with_fonts):
    """Font büyüklüğü veya büyük harf durumuna göre başlıkları tespit edip alt satırları gruplar."""
    sections = {}
    current_section = "general"
    sections[current_section] = []

    avg_font = sum(f for _, f in lines_with_fonts) / max(1, len(lines_with_fonts))
    threshold = avg_font * 1.15  # başlık tespiti için font farkı eşiği

    for text, font_size in lines_with_fonts:
        is_upper = text.isupper() and len(text) > 3
        is_bigger = font_size >= threshold

        if is_upper or is_bigger:
            current_section = text.strip().lower()
            sections[current_section] = []
        else:
            sections.setdefault(current_section, []).append(text.strip())

    # Birleştir
    structured = {k: "\n".join(v) for k, v in sections.items() if v}
    return structured


# ==========================================================
# 🤖 3️⃣ Gemini extraction (tam entegrasyon)
# ==========================================================
def extract_info_gemini_from_pdf(pdf_path: str) -> Dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("❌ GEMINI_API_KEY bulunamadı (.env dosyasını kontrol et).")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("models/gemini-2.5-flash")

    lines = read_pdf_with_font_info(pdf_path)
    sections = group_sections(lines)

    # Prompt oluştur
    prompt = f"""
    You are an expert CV parser. The CV content may be written in Turkish or other languages.
    Before processing, translate the entire CV into English. 
    Then, extract the information strictly based on the translated English version.

    Each section below corresponds to a block grouped by visual headers (uppercase or large font).
    Use this context to accurately map information into the structured JSON.

    Return ONLY valid JSON in this format:

    {{
      "personal_info": {{
        "name": "",
        "email": "",
        "phone": ""
      }},
      "education": [{{"institution": "", "degree": "", "start": "", "end": ""}}],
      "experience": [{{"company": "", "role": "", "start": "", "end": "", "description": ""}}],
      "certificates": [{{"course": "", "organization": "", "year": ""}}],
      "skills": [],
      "projects": [{{"title": "", "technologies": "", "description": ""}}],
      "languages": [{{"language": "", "level": ""}}],
      "summary": ""
    }}

    Rules:
    - Translate non-English content to English before extracting.
    - Use detected sections for accurate grouping.
    - Treat all-uppercase or large-font lines as headers.
    - Select the longest paragraph as summary.
    - If duplicate info appears (e.g., both education and experience mention dates), choose best fit.
    - Return JSON only.

    CV Sections (from PDF):
    {json.dumps(sections, indent=2, ensure_ascii=False)}
    """

    print("⚙️ Gemini'ye gönderilen başlıklar:", list(sections.keys()))

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0
            )
        )

        text = response.text.strip()
        if "```" in text:
            text = text.split("```")[-1]
        text = text.replace("json", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"⚠️ Gemini parse hatası: {e}")
        return {"error": str(e)}


# ==========================================================
# 🔁 4️⃣ Harici kullanım için basit fonksiyon
# ==========================================================
def extract_info(cv_text_or_path: str) -> Dict[str, Any]:
    """
    PDF path verilirse doğrudan PDF'ten okur.
    Yoksa metin olarak işlenir.
    """
    if os.path.exists(cv_text_or_path) and cv_text_or_path.lower().endswith(".pdf"):
        return extract_info_gemini_from_pdf(cv_text_or_path)
    else:
        # Fallback: metin geldiğinde sadece Gemini'ye gönder
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("models/gemini-2.5-flash")

        prompt = f"""
        Extract structured CV information from the text below and return valid JSON
        (same schema as before).
        Translate non-English parts to English first.

        CV Text:
        {cv_text_or_path}
        """
        resp = model.generate_content(prompt)
        text = resp.text.strip().replace("json", "").replace("```", "")
        return json.loads(text)