"""
GeliÅ?miÅ? training script for CV - Job matching using SentenceTransformers.
Features:
- Data cleaning and balancing
- Train/dev/test split
- Support for multiple models (configurable)
- MultipleNegativesRankingLoss fine-tuning
- Learning-rate, epochs, batch_size, warmup_steps configurable
- Evaluation on dev set (ROC-AUC and accuracy@threshold)
- Save model & encoder embeddings cache
- Simple data augmentation (synonym swap placeholder)

KullanÄ±m:
python train_model_advanced.py --data_dir ../data --model_out ../models/job_match_model --model_name BAAI/bge-base-en-v1.5 --epochs 4 --batch_size 32

Not: CUDA varsa otomatik kullanÄ±lÄ±r.
"""

import os
import argparse
import random
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader
import torch
from torch.cuda.amp import autocast, GradScaler


scaler = GradScaler()


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🟢 Aktif cihaz: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # Basit temizleme; ihtiyaca göre geliÅ?tirilebilir
    s = s.replace('\n', ' ').replace('\r', ' ')
    s = s.strip()
    # Remove common signature phrases (örnek)
    for token in ['saygÄ±larÄ±mla', 'saygÄ±lar', 'iletisim', 'telefon', 'adres', 'e-posta', 'email']:
        s = s.replace(token, '')
    return ' '.join(s.split())


def balance_dataframe(df: pd.DataFrame, label_col='label', seed=42):
    pos = df[df[label_col] == 1]
    neg = df[df[label_col] == 0]
    if len(pos) == 0:
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)
    if len(neg) == 0:
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = min(len(pos), len(neg))
    pos_s = pos.sample(n, random_state=seed)
    neg_s = neg.sample(n, random_state=seed)
    balanced = pd.concat([pos_s, neg_s]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return balanced


def build_input_examples(df: pd.DataFrame):
    # For MultipleNegativesRankingLoss we need (anchor, positive) pairs
    examples = []
    for _, row in df.iterrows():
        if row.label == 1:
            # anchor = cv, positive = job
            examples.append(InputExample(texts=[row.cv_text, row.job_text]))
    return examples


def evaluate(model: SentenceTransformer, df_eval: pd.DataFrame) -> dict:
    # Encode in batches
    cvs = df_eval.cv_text.tolist()
    jobs = df_eval.job_text.tolist()
    labels = df_eval.label.values

    emb_cvs = model.encode(cvs, convert_to_numpy=True, batch_size=64, show_progress_bar=False)
    emb_jobs = model.encode(jobs, convert_to_numpy=True, batch_size=64, show_progress_bar=False)

    # Cosine similarity
    def cos_sim(a, b):
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        return np.sum(a * b, axis=1)

    sims = cos_sim(emb_cvs, emb_jobs)
    roc = roc_auc_score(labels, sims)

    # Find best threshold for accuracy on dev set
    thresholds = np.linspace(0, 1, 101)
    best_acc = 0
    best_th = 0.5
    for th in thresholds:
        preds = (sims >= th).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return {"roc_auc": float(roc), "best_acc": float(best_acc), "best_threshold": float(best_th)}


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_dir = Path(args.data_dir)
    data_path = base_dir / args.train_csv
    if not data_path.exists():
        raise FileNotFoundError(f"Veri bulunamadÄ±: {data_path}")

    df = pd.read_csv(data_path)
    print(f"YÃ¼klendi toplam: {len(df)}")

    # Drop NaN and clean
    df = df.dropna(subset=['cv_text', 'job_text'])
    df['cv_text'] = df['cv_text'].apply(clean_text)
    df['job_text'] = df['job_text'].apply(clean_text)

    # Optional quick length filter: remove extremely short rows
    df['cv_len'] = df['cv_text'].str.len()
    df['job_len'] = df['job_text'].str.len()
    df = df[(df.cv_len > 20) & (df.job_len > 20)].copy()
    df = df.drop(columns=['cv_len', 'job_len'])

    # Train/dev/test split (stratify by label)
    train_df, temp_df = train_test_split(df, test_size=args.dev_test_ratio, stratify=df.label, random_state=args.seed)
    dev_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df.label, random_state=args.seed)

    print(f"Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")

    # Balance training data (important!)
    if args.balance:
        train_df = balance_dataframe(train_df, label_col='label', seed=args.seed)
        print(f"Balanced train size: {len(train_df)}")

    # Build InputExample list (only positives used for MultipleNegativesRankingLoss)
    train_examples = build_input_examples(train_df)
    print(f"Pozitif anchor-positive ciftleri: {len(train_examples)}")

    # Model initialization (with pooling head)
    print(f"Model yÃ¼kleniyor: {args.model_name}")
    model = SentenceTransformer(args.model_name)

    # Use MultipleNegativesRankingLoss by default
    if args.loss == 'multiple_negatives':
        train_loss = losses.MultipleNegativesRankingLoss(model)
    elif args.loss == 'triplet':
        train_loss = losses.TripletLoss(model)
    else:
        train_loss = losses.CosineSimilarityLoss(model)

    # DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)

    # Warmup steps heuristic
    total_steps = int(len(train_dataloader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    print("EÄ?itim başlatılıyor...")
    torch.cuda.empty_cache()

    with autocast():
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=args.epochs,
            warmup_steps=100,
            optimizer_params={'lr': args.learning_rate},
            show_progress_bar=True,
            use_amp=True
        )

    # Kaydet
    out_dir = Path(args.model_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir)
    print(f"Model kaydedildi: {out_dir}")

    # Değerlendirme
    print("Dev set üzerinde değerlendirme...")
    dev_metrics = evaluate(model, dev_df)
    print(dev_metrics)

    print("Test set üzerinde değerlendirme...")
    test_metrics = evaluate(model, test_df)
    print(test_metrics)

    # Opsiyonel: embeddings cache
    if args.save_embeddings:
        emb_dir = out_dir / 'embeddings'
        emb_dir.mkdir(exist_ok=True)
        cvs = df.cv_text.tolist()
        jobs = df.job_text.tolist()
        emb_cvs = model.encode(cvs, convert_to_numpy=True, batch_size=64, show_progress_bar=True)
        emb_jobs = model.encode(jobs, convert_to_numpy=True, batch_size=64, show_progress_bar=True)
        np.save(emb_dir / 'cv_embeddings.npy', emb_cvs)
        np.save(emb_dir / 'job_embeddings.npy', emb_jobs)
        df.to_parquet(emb_dir / 'meta.parquet')
        print(f"Embeddings kaydedildi: {emb_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data', help='Data directory')
    parser.add_argument('--train_csv', type=str, default='train_pairs.csv', help='Train CSV filename')
    parser.add_argument('--model_name', type=str, default='paraphrase-multilingual-MiniLM-L12-v2', help='SentenceTransformer model name')
    parser.add_argument('--model_out', type=str, default='../models/job_match_model', help='Output directory to save model')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--dev_test_ratio', type=float, default=0.2)
    parser.add_argument('--balance', action='store_true', help='Balance training data (undersample majority)')
    parser.add_argument('--loss', type=str, default='multiple_negatives', choices=['multiple_negatives','triplet','cosine'])
    parser.add_argument('--save_embeddings', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    main(args)
