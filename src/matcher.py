# src/matcher.py
from sentence_transformers import SentenceTransformer, util

# Model sadece bir kez yüklensin
MODEL_PATH = "./models/smart_job_model"
model = SentenceTransformer(MODEL_PATH)

def calculate_similarity(cv_text: str, job_text: str) -> float:
    """
    CV metni ile iş ilanı metni arasında benzerlik skoru hesaplar.
    Çıktı: 0.0 - 1.0 arası float değer.
    """
    if not cv_text.strip() or not job_text.strip():
        return 0.0

    cv_emb = model.encode(cv_text, convert_to_tensor=True)
    job_emb = model.encode(job_text, convert_to_tensor=True)
    score = util.cos_sim(cv_emb, job_emb).item()
    return round(float(score), 3)
