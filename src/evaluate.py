# src/evaluate_model_smart.py
import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MinMaxScaler

def evaluate_model(model_dir, data_path, threshold=None):
    print(f"Model yükleniyor: {model_dir}")
    model = SentenceTransformer(model_dir)

    df = pd.read_csv(data_path)
    df = df.dropna(subset=["cv_text", "job_text", "label"])
    print(f"Test örnekleri: {len(df)}")

    cv_emb = model.encode(df["cv_text"].tolist(), convert_to_tensor=True, show_progress_bar=True, batch_size=16)
    job_emb = model.encode(df["job_text"].tolist(), convert_to_tensor=True, show_progress_bar=True, batch_size=16)

    similarities = util.cos_sim(cv_emb, job_emb).diagonal().cpu().numpy()

    # Min-Max normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    scores = scaler.fit_transform(similarities.reshape(-1,1)).flatten()

    # Threshold belirleme
    if threshold is None:
        # ROC-AUC’ya göre optimal threshold
        from sklearn.metrics import roc_curve
        fpr, tpr, ths = roc_curve(df["label"].values, scores)
        youden = tpr - fpr
        threshold = ths[youden.argmax()]
        print(f"📌 Optimal threshold: {threshold:.3f}")

    preds = (scores >= threshold).astype(int)
    labels = df["label"].astype(int).values

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, scores)

    print(f"📊 Accuracy: {acc:.4f}")
    print(f"📈 F1 Score: {f1:.4f}")
    print(f"💡 ROC AUC: {auc:.4f}")

    df["score"] = scores
    df["pred"] = preds
    result_path = os.path.join(model_dir, "eval_results.csv")
    df.to_csv(result_path, index=False)
    print(f"Sonuçlar kaydedildi: {result_path}")

if __name__ == "__main__":
    evaluate_model(
        model_dir="./models/smart_job_model",
        data_path="./data/test_pairs.csv"
    )
