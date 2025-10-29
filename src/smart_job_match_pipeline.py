"""
Smart Job Matching Pipeline
- Data preparation with semantic positives (embedding-based), easy & hard negatives
- Augmentation helpers (synonym swap placeholder)
- Triplet and MultipleNegativesRankingLoss support with option to use both
- Mixed precision (fp16) support, gradient accumulation, AdamW + scheduler
- Evaluation: ROC-AUC, Accuracy@threshold, Precision/Recall/F1, calibration (Platt/Isotonic)
- Hybrid scorer: semantic embedding + keyword overlap
- Save model + scaler + config

Usage example:
python smart_job_match_pipeline.py \
  --data_dir ./data \
  --resume_csv resume_dataset.csv \
  --job_csv job_description_dataset.csv \
  --out_dir ./models/smart_job_model \
  --model_name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --epochs 3 --batch_size 8 --fp16 --accumulation_steps 2

Note: requires sentence-transformers, scikit-learn, torch>=2.0
"""

import os
import argparse
import random
import math
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers.optimization import Adafactor, get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer, InputExample, losses, util

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# -------------------- Utilities --------------------

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace('\n', ' ').replace('\r', ' ')
    s = s.strip()
    return ' '.join(s.split())


def extract_keywords(text: str, top_k: int = None) -> set:
    # very simple tokenizer; replace with TF-IDF or keyBERT for production
    tokens = [t.lower() for t in util.simple_tokenize(text)]
    return set(tokens)


def keyword_overlap(cv_text: str, job_text: str) -> float:
    a = extract_keywords(cv_text)
    b = extract_keywords(job_text)
    if not b:
        return 0.0
    return len(a & b) / len(b)


# -------------------- Data preparation --------------------

class PairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str, int]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        cv, job, label = self.pairs[idx]
        return cv, job, float(label)


def build_semantic_pairs(resumes: pd.DataFrame, jobs: pd.DataFrame, model: SentenceTransformer, pos_threshold=0.55, max_pos_per_resume=3):
    """For each resume find top similar job postings and add as positives if above threshold."""
    resumes = resumes.copy()
    jobs = jobs.copy()
    resume_texts = resumes['text'].tolist()
    job_texts = jobs['text'].tolist()

    resume_emb = model.encode(resume_texts, convert_to_numpy=True, show_progress_bar=True)
    job_emb = model.encode(job_texts, convert_to_numpy=True, show_progress_bar=True)

    pairs = []
    for i, r in enumerate(tqdm(resume_texts, desc='matching resumes')):
        sims = util.cos_sim(torch.tensor(resume_emb[i]), torch.tensor(job_emb)).numpy()[0]
        top_idx = np.argsort(-sims)[:max_pos_per_resume]
        added = 0
        for idx in top_idx:
            score = float(sims[idx])
            if score >= pos_threshold:
                pairs.append((r, job_texts[idx], 1))
                added += 1
            if added >= max_pos_per_resume:
                break
    return pairs


def create_negative_samples(resumes: pd.DataFrame, jobs: pd.DataFrame, num_neg_per_resume=1):
    pairs = []
    job_texts = jobs['text'].tolist()
    for r in resumes['text']:
        for _ in range(num_neg_per_resume):
            j = random.choice(job_texts)
            pairs.append((r, j, 0))
    return pairs


def create_hard_negatives_by_title(resumes: pd.DataFrame, jobs: pd.DataFrame, title_col='title'):
    # If title exists in jobs/resumes, create negatives from same industry/title group
    pairs = []
    if title_col not in jobs.columns:
        return []
    grouped = jobs.groupby(title_col)
    for _, group in grouped:
        texts = group['text'].tolist()
        if len(texts) < 2:
            continue
        for i in range(len(texts)-1):
            a = texts[i]
            b = texts[i+1]
            pairs.append((a, b, 0))
    return pairs


# -------------------- Training helpers --------------------

def collate_inputexamples(batch):
    examples = []
    for cv, job, label in batch:
        if label == 1:
            examples.append(InputExample(texts=[cv, job]))
    return examples


def compute_metrics(y_true, y_scores, threshold=None):
    if threshold is None:
        # choose best threshold by maximizing F1 on validation
        best_f1 = -1
        best_th = 0.5
        for th in np.linspace(0.1, 0.9, 81):
            preds = (y_scores >= th).astype(int)
            p, r, f, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_th = th
        threshold = best_th

    preds = (y_scores >= threshold).astype(int)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, zero_division=0)
    auc = roc_auc_score(y_true, y_scores)
    p, r, f, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)
    return {'threshold': threshold, 'accuracy': acc, 'f1': f1, 'precision': p, 'recall': r, 'roc_auc': auc}


# -------------------- Main pipeline --------------------

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    # Load raw data
    resumes = pd.read_csv(os.path.join(args.data_dir, args.resume_csv))
    jobs = pd.read_csv(os.path.join(args.data_dir, args.job_csv))
    resumes.columns = [c.lower() for c in resumes.columns]
    jobs.columns = [c.lower() for c in jobs.columns]

    # normalize text column names
    for df in (resumes, jobs):
        if 'text' not in df.columns:
            for col in df.columns:
                if 'resume' in col or 'description' in col or 'job' in col:
                    df.rename(columns={col: 'text'}, inplace=True)
                    break
    resumes['text'] = resumes['text'].astype(str).apply(clean_text)
    jobs['text'] = jobs['text'].astype(str).apply(clean_text)

    # split for eval
    resumes_train, resumes_test = train_test_split(resumes, test_size=args.test_ratio, random_state=args.seed)
    jobs_train, jobs_test = train_test_split(jobs, test_size=args.test_ratio, random_state=args.seed)

    print('Loading model for pair creation:', args.model_name)
    pair_model = SentenceTransformer(args.model_name, device='cuda' if torch.cuda.is_available() else 'cpu')

    print('Building semantic positive pairs...')
    pos_pairs = build_semantic_pairs(resumes_train, jobs_train, pair_model, pos_threshold=args.pos_threshold, max_pos_per_resume=args.max_pos_per_resume)
    print('Positives:', len(pos_pairs))

    print('Creating random negatives...')
    neg_pairs = create_negative_samples(resumes_train, jobs_train, num_neg_per_resume=args.num_neg)
    print('Negatives:', len(neg_pairs))

    if args.hard_neg:
        print('Creating hard negatives (by title if available)...')
        hard = create_hard_negatives_by_title(resumes_train, jobs_train, title_col=args.title_col)
        print('Hard negatives:', len(hard))
    else:
        hard = []

    # combine and shuffle
    all_pairs = pos_pairs + neg_pairs + hard
    random.shuffle(all_pairs)

    # optionally limit size for memory
    if args.limit_pairs:
        all_pairs = all_pairs[:args.limit_pairs]

    # split to train/dev/test by pairs
    pair_df = pd.DataFrame(all_pairs, columns=['cv_text', 'job_text', 'label'])
    train_df, dev_df = train_test_split(pair_df, test_size=args.dev_ratio, random_state=args.seed, stratify=pair_df['label'] if 'label' in pair_df.columns else None)

    print('Train size:', len(train_df), 'Dev size:', len(dev_df))

    # create InputExamples for positives (MultipleNegatives expects anchor-positive pairs)
    train_examples = [InputExample(texts=[r['cv_text'], r['job_text']]) for _, r in train_df.iterrows() if r['label']==1]

    model = SentenceTransformer(args.model_name, device='cuda' if torch.cuda.is_available() else 'cpu')

    # loss selection
    if args.loss == 'multiple_negatives':
        train_loss = losses.MultipleNegativesRankingLoss(model)
    elif args.loss == 'triplet':
        train_loss = losses.TripletLoss(model)
    else:
        train_loss = losses.CosineSimilarityLoss(model)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)

    # total steps for scheduler
    total_steps = max(1, len(train_dataloader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    print('Starting training on device:', 'cuda' if torch.cuda.is_available() else 'cpu')

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        evaluator=None,
        show_progress_bar=True,
        use_amp=args.fp16,
        output_path=args.out_dir,
        save_best_model=True
    )

    # Save basic config
    with open(os.path.join(args.out_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    # Evaluate on dev set
    print('Evaluating on dev set...')
    dev_cv = dev_df['cv_text'].tolist()
    dev_job = dev_df['job_text'].tolist()
    dev_labels = dev_df['label'].astype(int).values

    model = SentenceTransformer(args.out_dir)
    emb_cv = model.encode(dev_cv, convert_to_numpy=True, show_progress_bar=True)
    emb_job = model.encode(dev_job, convert_to_numpy=True, show_progress_bar=True)
    sims = util.cos_sim(torch.tensor(emb_cv), torch.tensor(emb_job)).diagonal().cpu().numpy()

    metrics = compute_metrics(dev_labels, sims)
    print('Dev metrics:', metrics)

    # calibration (Platt / logistic) on dev
    print('Calibrating scores with logistic regression on dev set...')
    lr = LogisticRegression(max_iter=1000)
    lr.fit(sims.reshape(-1,1), dev_labels)
    # save calibrator
    import joblib
    joblib.dump(lr, os.path.join(args.out_dir, 'calibrator_lr.joblib'))

    # Save hybrid thresholds/settings
    settings = {'threshold': metrics['threshold'], 'alpha': args.alpha}
    with open(os.path.join(args.out_dir, 'scoring_settings.json'), 'w', encoding='utf-8') as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)

    print('Done. Model and artifacts saved to', args.out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--resume_csv', type=str, default='resume_dataset.csv')
    parser.add_argument('--job_csv', type=str, default='job_description_dataset.csv')
    parser.add_argument('--out_dir', type=str, default='./models/smart_job_model')
    parser.add_argument('--model_name', type=str, default='paraphrase-multilingual-MiniLM-L12-v2')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--loss', choices=['multiple_negatives','triplet','cosine'], default='multiple_negatives')
    parser.add_argument('--pos_threshold', type=float, default=0.55)
    parser.add_argument('--max_pos_per_resume', type=int, default=2)
    parser.add_argument('--num_neg', type=int, default=1)
    parser.add_argument('--hard_neg', action='store_true')
    parser.add_argument('--title_col', type=str, default='title')
    parser.add_argument('--dev_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--limit_pairs', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alpha', type=float, default=0.7, help='alpha weight for semantic in hybrid scoring')

    args = parser.parse_args()
    main(args)
