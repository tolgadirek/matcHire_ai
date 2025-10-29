# src/create_test_pairs.py
import pandas as pd
import random
import os
from tqdm import tqdm

def create_test_pairs():
    base_dir = "./data"
    resume_path = os.path.join(base_dir, "resume_dataset.csv")
    job_path = os.path.join(base_dir, "job_description_dataset.csv")

    # Veri setlerini oku
    resumes = pd.read_csv(resume_path)
    jobs = pd.read_csv(job_path)

    # Kolon isimlerini normalize et
    resumes.columns = [c.lower() for c in resumes.columns]
    jobs.columns = [c.lower() for c in jobs.columns]

    # Resume ve Job text kolonlarını düzelt
    if "resume_str" in resumes.columns:
        resumes.rename(columns={"resume_str": "text"}, inplace=True)
    elif "resume" in resumes.columns:
        resumes.rename(columns={"resume": "text"}, inplace=True)

    if "category" not in resumes.columns:
        resumes["category"] = "General"

    if "jobdescription" in jobs.columns:
        jobs.rename(columns={"jobdescription": "text"}, inplace=True)

    # Eksik verileri temizle
    resumes = resumes.dropna(subset=["text"])
    jobs = jobs.dropna(subset=["text"])

    pairs = []

    print("✅ Pozitif test örnekleri oluşturuluyor...")
    for i, r in tqdm(resumes.iterrows(), total=len(resumes)):
        cat = str(r.get("category", "")).lower().strip()
        if not cat or cat == "nan":
            continue
        # Aynı kategoriye ait job açıklamaları seç
        matches = [j for j in jobs["text"].tolist() if cat in j.lower()]
        if matches:
            job_text = random.choice(matches)
            pairs.append([r["text"], job_text, 1])

    print("❌ Negatif test örnekleri oluşturuluyor...")
    for i, r in tqdm(resumes.iterrows(), total=len(resumes)):
        # Rastgele farklı kategoriye ait iş açıklaması al
        diff_jobs = jobs.sample(1).iloc[0]["text"]
        pairs.append([r["text"], diff_jobs, 0])

    # Test DataFrame'i oluştur
    df_pairs = pd.DataFrame(pairs, columns=["cv_text", "job_text", "label"])
    test_path = os.path.join(base_dir, "test_pairs.csv")

    # Eğer çok büyükse sadece bir kısmını al (örneğin 500 örnek)
    if len(df_pairs) > 1000:
        df_pairs = df_pairs.sample(500, random_state=42)

    df_pairs.to_csv(test_path, index=False, encoding="utf-8")
    print(f"✅ test_pairs.csv oluşturuldu: {test_path} ({len(df_pairs)} satır)")

if __name__ == "__main__":
    create_test_pairs()
