from sentence_transformers import SentenceTransformer, util
import json, os, re

MODEL_PATH = "./models/smart_job_model"
SETTINGS_PATH = os.path.join(MODEL_PATH, "scoring_settings.json")
model = SentenceTransformer(MODEL_PATH)

# 🔹 alpha'yı kaydedilen config'ten oku (yoksa varsayılan 0.7)
alpha = 1.0
if os.path.exists(SETTINGS_PATH):
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            alpha = json.load(f).get("alpha", 1.0)
    except Exception:
        pass

def keyword_overlap(cv_text: str, job_text: str) -> float:
    """Basit kelime örtüşme oranı"""
    cv_tokens = set(re.findall(r'\b\w+\b', cv_text.lower()))
    job_tokens = set(re.findall(r'\b\w+\b', job_text.lower()))
    if not job_tokens:
        return 0.0
    return len(cv_tokens & job_tokens) / len(job_tokens)

def calculate_similarity(cv_text: str, job_text: str) -> float:
    """Hybrid similarity: semantic + keyword overlap"""
    if not cv_text.strip() or not job_text.strip():
        return 0.0

    cv_emb = model.encode(cv_text, convert_to_tensor=True)
    job_emb = model.encode(job_text, convert_to_tensor=True)
    semantic = util.cos_sim(cv_emb, job_emb).item()
    keyword = keyword_overlap(cv_text, job_text)

    final_score = alpha * semantic + (1 - alpha) * keyword
    return round(float(final_score), 3)
