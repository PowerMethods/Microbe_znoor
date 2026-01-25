#!/usr/bin/env python
"""
================================================================================
CNN256 PHASE-7: CONTROLLED FINE-TUNING + PROBABILITY CALIBRATION (3-CLASS)
================================================================================

WHAT THIS SCRIPT DOES
---------------------
This script performs **Phase-7 fine-tuning and calibration** of the CNN256 model
trained in Phase-6.

Phase-7 objectives:
• Start from the best Phase-6 checkpoint (best_model.pt)
• Freeze all convolutional layers (feature extractor)
• Fine-tune ONLY the classification head
• Use a low learning rate and short training schedule
• Maintain the same pair-aware evaluation logic as Phase-6
• Select the best model based on Macro-F1 + MCC (pair-level)
• Prepare outputs for post-hoc probability calibration

Key properties:
----------------
• Segment-level training, pair-level evaluation
• Pair-aware 60/40 split (same logic as Phase-6)
• Majority voting aggregation at pair level
• Extensive uncertainty metrics (agreement, entropy, confidence)
• Calibration diagnostics (ECE, Brier, NLL)
• Final output is a **fine-tuned classifier head** ready for temperature scaling

Outputs:
--------
• pair_summary_[train|val]_epoch_XXX.csv
• cnn256_phase7_finetuned.pt (best fine-tuned model)
• Calibration metrics for downstream temperature scaling

================================================================================
"""

# =========================
# IMPORTS
# =========================
import os, glob, csv, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score,
    roc_auc_score, log_loss, confusion_matrix
)

# =========================
# LIVE LOGGING
# =========================
def log(msg):
    """Timestamped logging for SLURM/stdout traceability."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# =========================
# CONFIGURATION
# =========================
SEG_DIR = "/vol/projects/znoor/cnn_training/3class_exp/segments_256"

# Best Phase-6 checkpoint
CKPT = "/vol/projects/znoor/cnn_training/3class_exp_CNNBESTMODEL_Phase6/best_model.pt"

# Phase-7 output directory
OUT_DIR = "/vol/projects/znoor/cnn_training/3class_exp/Phase7_Finetune_Calibration"
os.makedirs(OUT_DIR, exist_ok=True)

# Fine-tuning hyperparameters
EPOCHS = 10               # Short, controlled fine-tuning
BATCH_SIZE = 16
LR = 1e-5                 # Low LR to avoid destroying Phase-6 features
NUM_CLASSES = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

log("Using device: " + DEVICE)
log("Loading Phase-6 best model: " + CKPT)

# =========================
# DATASET
# =========================
class SegmentDataset(Dataset):
    """
    Dataset for segment-level CNN inputs.

    Each NPZ contains:
    • windows : (256, 19, 500) → reshaped to (256, features)
    • label   : integer class label
    """

    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx], allow_pickle=True)

        # Flatten windows for Conv1D input
        x = d["windows"].astype(np.float32).reshape(256, -1)

        # Segment label
        y = int(d["label"].item())

        # Pair ID (used for aggregation)
        pid = os.path.basename(self.files[idx]).split("_seg")[0]

        return torch.tensor(x), torch.tensor(y), pid

# =========================
# MODEL ARCHITECTURE (CNN256)
# =========================
class CNN256(nn.Module):
    """
    Same CNN256 architecture as Phase-6.
    """

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 2375, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.flatten(1))

# =========================
# LOAD FILES (PAIR-AWARE SPLIT)
# =========================
all_files = sorted(glob.glob(f"{SEG_DIR}/*.npz"))

# Group segment files by biological pair
pair_map = defaultdict(list)
for f in all_files:
    pid = os.path.basename(f).split("_seg")[0]
    pair_map[pid].append(f)

pairs = sorted(pair_map.keys())

np.random.seed(42)
np.random.shuffle(pairs)

# 60 / 40 split at PAIR level
n_train = int(0.6 * len(pairs))
train_pairs = set(pairs[:n_train])
val_pairs   = set(pairs[n_train:])

train_files = [f for p in train_pairs for f in pair_map[p]]
val_files   = [f for p in val_pairs   for f in pair_map[p]]

log(f"Phase-7 Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")

train_loader = DataLoader(
    SegmentDataset(train_files),
    BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    SegmentDataset(val_files),
    BATCH_SIZE,
    shuffle=False
)

# =========================
# METRIC HELPERS (PAIR-LEVEL)
# =========================
def pair_stats(votes, probs, true):
    """
    Computes pair-level prediction and uncertainty metrics.
    """
    c = Counter(votes)
    total = len(votes)

    pred = c.most_common(1)[0][0]
    agree = c[pred] / total

    p = np.array([c.get(i, 0) / total for i in range(NUM_CLASSES)])
    entropy = -np.sum(p * np.log(p + 1e-9))

    mean_conf = np.mean([pr[pred] for pr in probs])

    return pred, agree, entropy, mean_conf


def ece_score(probs, labels, bins=10):
    """
    Expected Calibration Error (ECE).
    """
    conf = np.max(probs, 1)
    pred = np.argmax(probs, 1)
    acc = pred == labels

    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0

    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if np.any(m):
            ece += abs(conf[m].mean() - acc[m].mean()) * m.mean()

    return ece

# =========================
# LOAD PHASE-6 MODEL + FREEZE BACKBONE
# =========================
model = CNN256().to(DEVICE)

# Load Phase-6 weights
model.load_state_dict(
    torch.load(CKPT, map_location=DEVICE)["model"]
)

# Freeze convolutional layers
for p in model.conv.parameters():
    p.requires_grad = False

# Optimizer only updates classifier head
opt = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

criterion = nn.CrossEntropyLoss()

# =========================
# PHASE-7 TRAINING LOOP
# =========================
best_score = -1
best_state = None

for epoch in range(1, EPOCHS + 1):

    log("=" * 60)
    log(f"PHASE-7 EPOCH {epoch}/{EPOCHS}")
    log("=" * 60)

    for split, loader in [("train", train_loader), ("val", val_loader)]:

        model.train() if split == "train" else model.eval()

        pair_votes = defaultdict(list)
        pair_probs = defaultdict(list)
        pair_truth = {}

        with torch.set_grad_enabled(split == "train"):
            for x, y, pid in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)

                out = model(x)
                probs = torch.softmax(out, 1)
                preds = probs.argmax(1)

                if split == "train":
                    loss = criterion(out, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                for i, p in enumerate(pid):
                    pair_votes[p].append(preds[i].item())
                    pair_probs[p].append(probs[i].detach().cpu().numpy())
                    pair_truth[p] = y[i].item()

        # =========================
        # PAIR-LEVEL EVALUATION
        # =========================
        rows = []
        y_true, y_pred, y_prob = [], [], []

        log(f"[{split.upper()}][PAIR] summaries")

        for i, p in enumerate(sorted(pair_votes)):
            pred, agree, H, conf = pair_stats(
                pair_votes[p], pair_probs[p], pair_truth[p]
            )

            y_true.append(pair_truth[p])
            y_pred.append(pred)
            y_prob.append(np.mean(pair_probs[p], 0))

            rows.append({
                "pair_id": p,
                "true": pair_truth[p],
                "pred": pred,
                "agreement": agree,
                "entropy": H,
                "mean_conf": conf
            })

            log(
                f"[{split.upper()}][PAIR {i+1:03d}] "
                f"agree={agree:.2f} H={H:.2f} conf={conf:.2f}"
            )

        pd.DataFrame(rows).to_csv(
            f"{OUT_DIR}/pair_summary_{split}_epoch_{epoch:03d}.csv",
            index=False
        )

        # =========================
        # METRICS
        # =========================
        y_prob = np.array(y_prob)
        y_true = np.array(y_true)

        acc   = accuracy_score(y_true, y_pred)
        bal   = balanced_accuracy_score(y_true, y_pred)
        f1m   = f1_score(y_true, y_pred, average="macro")
        f1w   = f1_score(y_true, y_pred, average="weighted")
        mcc   = matthews_corrcoef(y_true, y_pred)
        roc   = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        nll   = log_loss(y_true, y_prob)
        brier = np.mean(np.sum((y_prob - np.eye(NUM_CLASSES)[y_true]) ** 2, 1))
        ece   = ece_score(y_prob, y_true)
        cm    = confusion_matrix(y_true, y_pred)

        log(f"[{split.upper()}] Accuracy          : {acc:.4f}")
        log(f"[{split.upper()}] Balanced Accuracy : {bal:.4f}")
        log(f"[{split.upper()}] Macro F1          : {f1m:.4f}")
        log(f"[{split.upper()}] Weighted F1       : {f1w:.4f}")
        log(f"[{split.upper()}] MCC               : {mcc:.4f}")
        log(f"[{split.upper()}] ROC-AUC (macro)   : {roc:.6f}")
        log(f"[{split.upper()}] NLL (log-loss)    : {nll:.6f}")
        log(f"[{split.upper()}] Brier score       : {brier:.4f}")
        log(f"[{split.upper()}] ECE (10 bins)     : {ece:.4f}")
        log(f"[{split.upper()}] Confusion Matrix:")
        for r in cm:
            log(f"[{split.upper()}] {r.tolist()}")

        # Best model selection based on robustness
        if split == "val":
            score = f1m + mcc
            if score > best_score:
                best_score = score
                best_state = model.state_dict()

# =========================
# SAVE BEST PHASE-7 MODEL
# =========================
torch.save(best_state, f"{OUT_DIR}/cnn256_phase7_finetuned.pt")
log("Saved best Phase-7 fine-tuned model")

# =========================
# TEMPERATURE CALIBRATION (NEXT STEP)
# =========================
log("Running temperature calibration...")
