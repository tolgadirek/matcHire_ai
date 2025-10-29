# src/hybrid_scorer.py
import re
import torch
from sentence_transformers import SentenceTransformer, util

def extract_keywords(text):
    """Basit keyword çıkarımı (sen burayı TF-IDF veya keybert ile geliştirebilirsin)"""
    text = text.lower()
    keywords = re.findall(r"\b[a-zA-Z0-9_+#]+\b", text)
    return set(keywords)

def keyword_overlap(cv_text, job_text):
    cv_kw = extract_keywords(cv_text)
    job_kw = extract_keywords(job_text)
    if not cv_kw or not job_kw:
        return 0.0
    overlap = len(cv_kw & job_kw) / len(job_kw)
    return overlap

def hybrid_score(model, cv_text, job_text, alpha=0.7):
    """alpha = anlam skorunun ağırlığı, 1-alpha = keyword skorunun ağırlığı"""
    emb1 = model.encode(cv_text, convert_to_tensor=True)
    emb2 = model.encode(job_text, convert_to_tensor=True)
    semantic_score = util.cos_sim(emb1, emb2).item()
    keyword_score = keyword_overlap(cv_text, job_text)
    final_score = alpha * semantic_score + (1 - alpha) * keyword_score
    return round(final_score, 3)


if __name__ == "__main__":
    model = SentenceTransformer("../models/job_match_model_v2")

    cv = """Computer engineer with experience in Python, data analysis, and machine learning."""
    job = """We are looking for a waiter who can handle customer service and table organization."""

    score = hybrid_score(model, cv, job)
    print(f"💡 Hybrid Score: {score}")
