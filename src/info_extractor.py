# -*- coding: utf-8 -*-
import re
import spacy
from typing import Dict, Any, List, Optional, Tuple

# ----------- NLP MODEL YÜKLEME -----------
def _load_nlp():
    try:
        return spacy.load("tr_core_news_sm")
    except Exception:
        return spacy.load("en_core_web_sm")

nlp = _load_nlp()

# ----------- REGEXLER -----------
EMAIL_REGEX = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+", re.IGNORECASE)
PHONE_REGEX = re.compile(
    r"(\+?\s*90[\s\-\.]?)?(?:\(?0?\)?\s*)?(?:5\d{2}|[2348]\d{2})[\s\-\.]?\d{3}[\s\-\.]?\d{2,4}"
)
YEAR_REGEX = re.compile(r"(?:19|20)\d{2}")

# ----------- KARA LİSTE VE BAŞLIKLAR -----------
NAME_BLACKLIST = {
    "tarihi", "doğum", "adres", "ehliyet", "üniversitesi",
    "mah", "mahallesi", "sokak", "cadde", "no", "bursa",
    "manisa", "istanbul", "ankara", "izmir"
}

SECTION_ALIASES = {
    "education": ["education", "eğitim", "okul", "üniversite", "university", "college", "lise", "mezuniyet", "degree"],
    "experience": ["experience", "deneyim", "iş deneyimi", "work", "employment", "career", "staj", "projeler"],
    "skills": ["skills", "yetenekler", "technologies", "stack", "teknolojiler"],
    "languages": ["languages", "diller", "yabancı dil", "language skills"],
    "certificates": ["certificate", "certificates", "sertifika", "kurs", "course", "training", "bootcamp", "workshop"],
    "summary": ["summary", "profil", "profile", "about", "hakkımda"]
}

# ----------- YARDIMCI FONKSİYONLAR -----------
def _normalize(s: str) -> str:
    return " ".join(s.strip().split())

def _filter_short_lines(lines: List[str]) -> List[str]:
    """1–2 kelimelik gürültü satırlarını atar."""
    return [l for l in lines if len(l.split()) > 2 and len(l) > 4]

def _detect_sections(lines: List[str]) -> Dict[str, List[str]]:
    sections = {k: [] for k in SECTION_ALIASES}
    current = None

    for raw in lines:
        line = _normalize(raw)
        if not line:
            continue
        lower = line.lower()

        # Başlık tespiti
        header_found = False
        for sec, keys in SECTION_ALIASES.items():
            if any(k in lower for k in keys):
                if len(line) < 80 or line.endswith(":"):
                    current = sec
                    header_found = True
                    break
        if header_found:
            continue

        # Bölüm birikimi
        if current:
            sections[current].append(line)
        else:
            # Başlık öncesi kısa özet cümleleri
            if len(line.split()) > 5:
                sections["summary"].append(line)

    # Gürültü temizliği
    for k in sections:
        sections[k] = _filter_short_lines(sections[k])
    return sections

def _extract_name(doc, first_lines: List[str]) -> Optional[str]:
    head_doc = nlp("\n".join(first_lines[:25]))
    candidates = []

    for ent in head_doc.ents:
        if ent.label_.upper() == "PERSON":
            name = _normalize(ent.text)
            if 2 <= len(name.split()) <= 4 and not any(w.lower() in NAME_BLACKLIST for w in name.split()):
                candidates.append(name)

    if candidates:
        return sorted(candidates, key=lambda x: len(x.split()), reverse=True)[0]
    return None

def _extract_email(text: str) -> Optional[str]:
    m = EMAIL_REGEX.search(text)
    return m.group(0) if m else None

def _extract_phone(text: str) -> Optional[str]:
    m = PHONE_REGEX.search(text)
    return _normalize(m.group(0)) if m else None

def _extract_social(lines, key: str) -> Optional[str]:
    for ln in lines:
        low = ln.lower()
        if key in low:
            if any(w in low for w in ["eğitim", "egitim", "kurs", "bootcamp", "workshop", "akademi"]):
                continue  # Bu bir kurs satırı, sosyal bağlantı değil
            if len(ln.split()) > 2:
                return _normalize(ln)
    return None

def _parse_education(block: List[str]) -> List[Dict[str, Any]]:
    edu = []
    for line in block:
        if len(line) < 3:
            continue
        is_uni = any(k in line.lower() for k in ["üniversite", "university", "college", "bachelor", "master", "lise"])
        year = None
        years = YEAR_REGEX.findall(line)
        if years:
            year = re.findall(r"(19|20)\d{2}", line)[-1]
        edu.append({
            "text": line,
            "institution": line if is_uni else None,
            "degree": None,
            "year": year
        })
    return edu

def _parse_experience(block: List[str]) -> List[Dict[str, Any]]:
    exp = []
    for line in block:
        if any(x in line.lower() for x in ["udemy", "btk", "coursera", "bootcamp", "academy", "workshop"]):
            continue  # Sertifika kısmına ait olma ihtimali yüksek
        years = YEAR_REGEX.findall(line)
        start = years[0] if len(years) >= 1 else None
        end = years[1] if len(years) >= 2 else None
        exp.append({
            "text": line,
            "start": start,
            "end": end
        })
    return exp

def _parse_certificates(block: List[str]) -> List[Dict[str, Any]]:
    certs = []
    for line in block:
        if len(line.split()) < 2:
            continue
        org = None
        match = re.search(
            r"(Udemy|Coursera|Google|Microsoft|BTK|Kaggle|LinkedIn Learning|Ecodation|IBM|Udacity|Patika|Akademi)",
            line, re.IGNORECASE
        )
        if match:
            org = match.group(0)
        years = YEAR_REGEX.findall(line)
        year = years[-1] if years else None
        certs.append({
            "text": line,
            "organization": org,
            "year": year
        })
    return certs

def _parse_skills(block: List[str]) -> List[str]:
    tokens = []
    for line in block:
        parts = re.split(r"[,;•·\|\-/]+", line)
        for p in parts:
            t = _normalize(p)
            if 1 < len(t) <= 64:
                tokens.append(t)
    # Yinelenenleri kaldır
    seen, out = set(), []
    for t in tokens:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out

def _parse_languages(block: List[str], text: str) -> List[Dict[str, str]]:
    langs = []
    vocab = [
        "turkish","türkçe","english","ingilizce","german","almanca","french","fransızca","spanish","ispanyolca"
    ]
    for line in block + text.splitlines():
        for l in vocab:
            if re.search(rf"\b{l}\b", line, re.IGNORECASE):
                langs.append({"language": l.title(), "level": "Unknown"})
    # Yinelenenleri kaldır
    unique = {l["language"].lower(): l for l in langs}
    return list(unique.values())

def _make_summary(lines: List[str]) -> str:
    summary = []
    for line in lines[:20]:
        if any(k in line.lower() for k in ["kişisel", "personal", "contact"]):
            break
        if len(line.split()) > 5:
            summary.append(line)
    return " ".join(summary[:3])

# ----------- ANA FONKSİYON -----------
def extract_info(cv_text: str) -> Dict[str, Any]:
    lines = [l.strip() for l in cv_text.splitlines()]
    doc = nlp(cv_text)
    sections = _detect_sections(lines)

    # Temel bilgiler
    name = _extract_name(doc, lines)
    email = _extract_email(cv_text)
    phone = _extract_phone(cv_text)
    linkedin = _extract_social(lines, "linkedin")
    github = _extract_social(lines, "github")

    # Boş sosyal linkleri None yap
    if linkedin and len(linkedin.split()) <= 2:
        linkedin = None
    if github and len(github.split()) <= 2:
        github = None

    # Alanlar
    education = _parse_education(sections.get("education", []))
    experience = _parse_experience(sections.get("experience", []))
    certificates = _parse_certificates(sections.get("certificates", []))
    skills = _parse_skills(sections.get("skills", []))
    languages = _parse_languages(sections.get("languages", []), cv_text)
    summary = _make_summary(sections.get("summary", []))

    return {
        "personal_info": {
            "name": name,
            "email": email,
            "phone": phone,
            "linkedin": linkedin,
            "github": github
        },
        "education": education,
        "experience": experience,
        "certificates": certificates,
        "skills": skills,
        "languages": languages,
        "summary": summary
    }
