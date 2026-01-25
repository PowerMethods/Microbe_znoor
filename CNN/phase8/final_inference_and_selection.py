#!/usr/bin/env python
"""
================================================================================
CNN256 PHASE-8: FINAL INFERENCE + TEMPERATURE-CALIBRATED CONFIDENCE FILTERING
================================================================================

WHAT THIS SCRIPT DOES
---------------------
This script performs **final inference** using the fully trained and calibrated
CNN256 pipeline (Phase-6 + Phase-7).

Pipeline stage:
---------------
• Loads the Phase-7 fine-tuned CNN256 model
• Applies post-hoc temperature scaling for calibrated probabilities
• Runs inference on *all available segment files*
• Aggregates segment predictions at the *pair level*
• Computes uncertainty and confidence statistics per pair
• Applies a rule-based confidence filter
• Produces the final results used for reporting and downstream analysis

Key characteristics:
--------------------
• Segment-level inference → Pair-level aggregation
• Majority voting for pair prediction
• Entropy, agreement, and confidence used as uncertainty measures
• Conservative confidence rule to identify high-reliability predictions
• Final outputs saved as a single CSV for analysis and visualization

Outputs:
--------
• final_pair_predictions.csv
• Console summary statistics for reporting

This script represents the **final operational model output**.

================================================================================
"""

# =========================
# IMPORTS
# =========================
import os, glob, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict, Counter

# =========================
# CONFIGURATION (FIXED PATHS)
# =========================
SEG_DIR = "/vol/projects/znoor/cnn_training/3class_exp/segments_256"

# Phase-7 fine-tuned model
MODEL_PATH = (
    "/vol/projects/znoor/cnn_training/3class_exp/"
    "Phase7_Finetune_Calibration/cnn256_phase7_finetuned.pt"
)

# Temperature scaling parameter (learned post-hoc)
TEMP_PATH = (
    "/vol/projects/znoor/cnn_training/3class_exp/"
    "Phase7_Finetune_Calibration/temperature.pkl"
)

# Output directory for final results
OUT_DIR = (
    "/vol/projects/znoor/cnn_training/3class_exp/"
    "FINAL_RESULTS"
)
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 3

print("Using device:", DEVICE)

# =========================
# MODEL DEFINITION
# (MUST MATCH TRAINING EXACTLY)
# =========================
class CNN256(nn.Module):
    """
    CNN256 architecture used in Phase-6 and Phase-7.
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
# LOAD MODEL + TEMPERATURE
# =========================
model = CNN256().to(DEVICE)

# Load Phase-7 fine-tuned weights
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)
model.eval()

# Load learned temperature scalar
with open(TEMP_PATH, "rb") as f:
    T = pickle.load(f)

print("Loaded model + temperature:", T)

# =========================
# LOAD ALL SEGMENT FILES
# =========================
all_files = sorted(glob.glob(f"{SEG_DIR}/*.npz"))

# Group segments by biological pair
pair_map = defaultdict(list)
for f in all_files:
    pid = os.path.basename(f).split("_seg")[0]
    pair_map[pid].append(f)

print("Total pairs:", len(pair_map))

# =========================
# SEGMENT-LEVEL INFERENCE
# =========================
pair_votes = defaultdict(list)   # predicted class per segment
pair_probs = defaultdict(list)   # calibrated probabilities per segment
pair_true  = {}                  # true label per pair

with torch.no_grad():
    for pid, files in pair_map.items():
        for f in files:
            d = np.load(f, allow_pickle=True)

            # Prepare input
            x = d["windows"].astype(np.float32).reshape(256, -1)
            y = int(d["label"].item())

            x = torch.tensor(x).unsqueeze(0).to(DEVICE)

            # Apply temperature scaling at inference time
            logits = model(x) / T
            probs = torch.softmax(logits, 1)

            pred = probs.argmax(1).item()

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

    # Majority vote
    c = Counter(votes)
    pred = c.most_common(1)[0][0]

    # Agreement = fraction of segments voting for final class
    agreement = c[pred] / len(votes)

    # Predictive entropy (uncertainty)
    p = np.array([c.get(i, 0) / len(votes) for i in range(NUM_CLASSES)])
    entropy = -np.sum(p * np.log(p + 1e-9))

    # Mean calibrated confidence for predicted class
    mean_conf = probs[:, pred].mean()

    # =========================
    # CONFIDENCE FILTER (RULE-BASED)
    # =========================
    # A prediction is considered "confident" if:
    # • Strong majority agreement
    # • Low entropy (low uncertainty)
    # • High calibrated confidence
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

print("Saved final results to:", OUT_CSV)

# =========================
# GLOBAL SUMMARY (REPORTING)
# =========================
print("\n===== FINAL SUMMARY =====")
print("Total pairs:", len(df))
print("Confident pairs:", df["confident_prediction"].sum())
print("Ambiguous pairs:", (~df["confident_prediction"]).sum())

print("\nConfident class distribution:")
print(df[df["confident_prediction"]]["pred"].value_counts())

print("\nOverall accuracy (all pairs):",
      (df["pred"] == df["true"]).mean().round(3))

print("\nAccuracy on confident pairs:",
      (df[df["confident_prediction"]]["pred"]
       == df[df["confident_prediction"]]["true"]).mean().round(3))

print("==========================")
print("FINAL INFERENCE COMPLETE")
