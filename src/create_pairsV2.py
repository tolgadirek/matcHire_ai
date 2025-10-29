import pandas as pd
import random
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch

def create_pairs():
    base_dir = "./data"
    resume_path = os.path.join(base_dir, "resume_dataset.csv")
    job_path = os.path.join(base_dir, "job_description_dataset.csv")

    resumes = pd.read_csv(resume_path)
    jobs = pd.read_csv(job_path)

    resumes.columns = [c.lower() for c in resumes.columns]
    jobs.columns = [c.lower() for c in jobs.columns]

    # ---- Sütun isimleri ----
    if "resume_str" in resumes.columns:
        resumes.rename(columns={"resume_str": "text"}, inplace=True)
    elif "resume" in resumes.columns:
        resumes.rename(columns={"resume": "text"}, inplace=True)
    if "category" not in resumes.columns:
        resumes["category"] = "General"

    jobs.rename(columns={"jobdescription": "text"}, inplace=True)

    resumes = resumes.dropna(subset=["text"])
    jobs = jobs.dropna(subset=["text"])

    print("🔍 SentenceTransformer modeli yükleniyor...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    print("📐 Embedding hesaplanıyor...")
    resume_embeddings = model.encode(resumes["text"].tolist(), convert_to_tensor=True, show_progress_bar=True)
    job_embeddings = model.encode(jobs["text"].tolist(), convert_to_tensor=True, show_progress_bar=True)

    pairs = []

    print("✨ Pozitif örnekler oluşturuluyor...")
    for i in tqdm(range(len(resumes))):
        resume_emb = resume_embeddings[i]
        cos_scores = util.cos_sim(resume_emb, job_embeddings)[0]
        best_idx = torch.argmax(cos_scores).item()
        best_score = cos_scores[best_idx].item()

        if best_score > 0.55:  # benzerlik eşiği (ayarlanabilir)
            pairs.append([resumes.iloc[i]["text"], jobs.iloc[best_idx]["text"], 1])

    print("❌ Negatif örnekler oluşturuluyor...")
    for i in tqdm(range(len(resumes))):
        job_text = jobs.sample(1).iloc[0]["text"]
        pairs.append([resumes.iloc[i]["text"], job_text, 0])

    df_pairs = pd.DataFrame(pairs, columns=["cv_text", "job_text", "label"])
    out_path = os.path.join(base_dir, "train_pairs2.csv")
    df_pairs.to_csv(out_path, index=False)
    print(f"\n✅ train_pairs.csv oluşturuldu: {out_path} ({len(df_pairs)} satır)")

if __name__ == "__main__":
    create_pairs()
