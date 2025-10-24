import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os

def train_model():
    base_dir = "../data"
    model_out = "../models/job_match_model"
    os.makedirs(model_out, exist_ok=True)

    data_path = os.path.join(base_dir, "train_pairs.csv")
    df = pd.read_csv(data_path)

    print(f"Yüklendi: {len(df)} örnek")
    print(f"Pozitif: {sum(df.label==1)}, Negatif: {sum(df.label==0)}")

    # NaN kontrolü
    df = df.dropna(subset=["cv_text", "job_text"])

    # Eğitim örnekleri oluştur
    train_examples = [
        InputExample(texts=[row.cv_text, row.job_text], label=float(row.label))
        for _, row in df.iterrows()
    ]

    # Hazır modeli yükle
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # DataLoader + Loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    print("Fine-tuning başlıyor...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=2,
        warmup_steps=100,
        show_progress_bar=True
    )

    # Modeli kaydet
    model.save(model_out)
    print(f"✅ Model kaydedildi: {model_out}")

if __name__ == "__main__":
    train_model()
