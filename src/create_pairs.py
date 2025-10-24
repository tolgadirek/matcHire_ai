import pandas as pd
import random
import os
from tqdm import tqdm

def create_pairs():
    base_dir = "../data"
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

    pairs = []

    print("Pozitif örnekler oluşturuluyor...")
    for i, r in tqdm(resumes.iterrows(), total=len(resumes)):
        cat = str(r.get("category", "")).lower().strip()
        if not cat or cat == "nan":
            continue
        matches = [j for j in jobs["text"].tolist() if cat in j.lower()]
        if matches:
            job_text = random.choice(matches)
            pairs.append([r["text"], job_text, 1])

    print("Negatif örnekler oluşturuluyor...")
    for i, r in tqdm(resumes.iterrows(), total=len(resumes)):
        job_text = jobs.sample(1).iloc[0]["text"]
        pairs.append([r["text"], job_text, 0])

    df_pairs = pd.DataFrame(pairs, columns=["cv_text", "job_text", "label"])
    out_path = os.path.join(base_dir, "train_pairs.csv")
    df_pairs.to_csv(out_path, index=False)
    print(f"✅ train_pairs.csv oluşturuldu: {out_path} ({len(df_pairs)} satır)")

if __name__ == "__main__":
    create_pairs()
