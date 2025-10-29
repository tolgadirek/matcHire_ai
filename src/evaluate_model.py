# src/evaluate_model.py
import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sentence_transformers import SentenceTransformer, util

def evaluate_model(model_dir, data_path, threshold=0.6):
    print(f"Model yükleniyor: {model_dir}")
    model = SentenceTransformer(model_dir)
    df = pd.read_csv(data_path)

    # Boş satırları temizle
    df = df.dropna(subset=["cv_text", "job_text", "label"])

    print(f"Test örnekleri: {len(df)}")

    cv_emb = model.encode(df["cv_text"].tolist(), convert_to_tensor=True, show_progress_bar=True)
    job_emb = model.encode(df["job_text"].tolist(), convert_to_tensor=True, show_progress_bar=True)

    similarities = util.cos_sim(cv_emb, job_emb).diagonal()
    scores = similarities.cpu().numpy()

    # Tahmin
    preds = (scores >= threshold).astype(int)
    labels = df["label"].astype(int).values

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, scores)

    print(f"📊 Accuracy: {acc:.4f}")
    print(f"📈 F1 Score: {f1:.4f}")
    print(f"💡 ROC AUC: {auc:.4f}")

    df["score"] = scores
    result_path = os.path.join(model_dir, "eval_results.csv")
    df.to_csv(result_path, index=False)
    print(f"Sonuçlar kaydedildi: {result_path}")

if __name__ == "__main__":
    evaluate_model(
        model_dir="./models/smart_job_model",
        data_path="./data/test_pairs.csv",
        threshold=0.45
    )
