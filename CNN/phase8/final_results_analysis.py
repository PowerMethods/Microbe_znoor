#!/usr/bin/env python
"""
FINAL PAIR-LEVEL INFERENCE SCRIPT (PHASE-7 CNN256)

Purpose
-------
This script performs FINAL inference for the trained Phase-7 CNN256 model.
It:
1. Loads segment-level NPZ files (256 windows per segment)
2. Runs calibrated CNN inference (temperature scaling)
3. Aggregates segment predictions into pair-level predictions
4. Computes uncertainty metrics (agreement, entropy, confidence)
5. Flags pairs as CONFIDENT vs AMBIGUOUS
6. Saves final pair-level results for downstream analysis & thesis reporting

This script represents the FINAL DEPLOYMENT STAGE of the CNN pipeline.
"""

# =========================
# IMPORTS
# =========================
import os
import glob
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict, Counter

# =========================
# CONFIGURATION (LOCKED PATHS)
# =========================

# Directory containing segment-level NPZ files
# Each file = one segment of 256 windows for a pair
SEG_DIR = "/vol/projects/znoor/cnn_training/3class_exp/segments_256"

# Best Phase-7 fine-tuned CNN checkpoint
MODEL_PATH = (
    "/vol/projects/znoor/cnn_training/3class_exp/"
    "Phase7_Finetune_Calibration/cnn256_phase7_finetuned.pt"
)

# Temperature scaling parameter learned post-hoc
TEMP_PATH = (
    "/vol/projects/znoor/cnn_training/3class_exp/"
    "Phase7_Finetune_Calibration/temperature.pkl"
)

# Output directory for final pair-level predictions
OUT_DIR = (
    "/vol/projects/znoor/cnn_training/3class_exp/"
    "FINAL_RESULTS"
)
os.makedirs(OUT_DIR, exist_ok=True)

# Hardware selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directionality classification (3-class)
NUM_CLASSES = 3

print("Using device:", DEVICE)

# =========================
# MODEL DEFINITION
# =========================
# IMPORTANT:
# This architecture MUST EXACTLY MATCH the training architecture
# used in Phase-6 / Phase-7, otherwise weights will not align.
class CNN256(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional feature extractor
        # Input shape: (batch, 256 segments, flattened window features)
        self.conv = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Fully connected classifier head
        self.fc = nn.Sequential(
            nn.Linear(64 * 2375, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)   # flatten all but batch
        return self.fc(x)

# =========================
# LOAD TRAINED MODEL + TEMPERATURE
# =========================
model = CNN256().to(DEVICE)

# Load trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Load temperature scalar (for calibration)
with open(TEMP_PATH, "rb") as f:
    T = pickle.load(f)

print("Loaded model and temperature scaling factor:", T)

# =========================
# LOAD SEGMENT FILES
# =========================
# Group segments by PAIR ID
# Assumes filenames like:
#   <PAIR_ID>_segXXX.npz
all_files = sorted(glob.glob(f"{SEG_DIR}/*.npz"))

pair_map = defaultdict(list)

for f in all_files:
    pid = os.path.basename(f).split("_seg")[0]
    pair_map[pid].append(f)

print("Total unique pairs found:", len(pair_map))

# =========================
# INFERENCE (SEGMENT LEVEL)
# =========================
pair_votes = defaultdict(list)   # predicted class per segment
pair_probs = defaultdict(list)   # probability vectors per segment
pair_true  = {}                  # ground truth label per pair

with torch.no_grad():
    for pid, files in pair_map.items():
        for f in files:
            # Load NPZ
            d = np.load(f, allow_pickle=True)

            # windows shape: (256, 19, 500) → flattened to (256, features)
            x = d["windows"].astype(np.float32).reshape(256, -1)

            # Ground-truth direction label
            y = int(d["label"].item())

            # Convert to tensor and add batch dimension
            x = torch.tensor(x).unsqueeze(0).to(DEVICE)

            # Calibrated inference
            logits = model(x) / T
            probs = torch.softmax(logits, dim=1)

            pred = probs.argmax(dim=1).item()

            # Store segment-level outputs
            pair_votes[pid].append(pred)
            pair_probs[pid].append(probs.cpu().numpy()[0])
            pair_true[pid] = y

# =========================
# PAIR-LEVEL AGGREGATION
# =========================
rows = []

for pid in sorted(pair_votes):
    votes = pair_votes[pid]
    probs = np.array(pair_probs[pid])
    true  = pair_true[pid]

    # Majority voting across segments
    vote_counter = Counter(votes)
    pred = vote_counter.most_common(1)[0][0]

    # Agreement ratio
    agreement = vote_counter[pred] / len(votes)

    # Vote distribution entropy (uncertainty)
    p = np.array([vote_counter.get(i, 0) / len(votes)
                  for i in range(NUM_CLASSES)])
    entropy = -np.sum(p * np.log(p + 1e-9))

    # Mean confidence of winning class
    mean_conf = probs[:, pred].mean()

    # Confidence decision rule
    # (Thresholds justified in thesis)
    confident = (
        agreement >= 0.80 and
        entropy <= 0.50 and
        mean_conf >= 0.70
    )

    rows.append({
        "pair_id": pid,
        "true": true,
        "pred": pred,
        "agreement": round(agreement, 3),
        "entropy": round(entropy, 3),
        "mean_confidence": round(mean_conf, 3),
        "num_segments": len(votes),
        "confident_prediction": confident
    })

df = pd.DataFrame(rows)

# =========================
# SAVE FINAL RESULTS
# =========================
OUT_CSV = f"{OUT_DIR}/final_pair_predictions.csv"
df.to_csv(OUT_CSV, index=False)
print("Saved final pair-level predictions to:", OUT_CSV)

# =========================
# GLOBAL DISTRIBUTION SUMMARY
# =========================
print("\n===== FINAL SUMMARY =====")

print("Total pairs:", len(df))
print("Confident pairs:", df["confident_prediction"].sum())
print("Ambiguous pairs:", (~df["confident_prediction"]).sum())

print("\nPredicted class distribution (ALL pairs):")
print(df["pred"].value_counts().sort_index())

print("\nPredicted class distribution (CONFIDENT only):")
print(df[df["confident_prediction"]]["pred"].value_counts().sort_index())

print("\nAgreement distribution:")
print(pd.cut(df["agreement"], bins=[0, 0.6, 0.8, 0.9, 1.0]).value_counts())

print("\nEntropy distribution:")
print(pd.cut(df["entropy"], bins=[0, 0.3, 0.5, 1.5]).value_counts())

print("\nOverall accuracy (all pairs):",
      round((df["pred"] == df["true"]).mean(), 3))

print("\nAccuracy on confident pairs:",
      round(
          (df[df["confident_prediction"]]["pred"]
           == df[df["confident_prediction"]]["true"]).mean(),
          3
      ))

print("==========================")
print("FINAL INFERENCE COMPLETE")
