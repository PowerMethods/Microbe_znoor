#!/usr/bin/env python
import os, glob, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict, Counter

# =========================
# CONFIG (YOUR CURRENT PATHS)
# =========================
SEG_DIR = "/vol/projects/znoor/cnn_training/3class_exp/segments_256"

MODEL_PATH = (
    "/vol/projects/znoor/cnn_training/3class_exp/"
    "Phase7_Finetune_Calibration/cnn256_phase7_finetuned.pt"
)

TEMP_PATH = (
    "/vol/projects/znoor/cnn_training/3class_exp/"
    "Phase7_Finetune_Calibration/temperature.pkl"
)

OUT_DIR = (
    "/vol/projects/znoor/cnn_training/3class_exp/"
    "FINAL_RESULTS"
)
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 3

print("Using device:", DEVICE)

# =========================
# MODEL DEFINITION (MUST MATCH TRAINING)
# =========================
class CNN256(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(256, 128, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 64, 5, padding=2),
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
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

with open(TEMP_PATH, "rb") as f:
    T = pickle.load(f)

print("Loaded model + temperature:", T)

# =========================
# LOAD ALL SEGMENTS
# =========================
all_files = sorted(glob.glob(f"{SEG_DIR}/*.npz"))
pair_map = defaultdict(list)

for f in all_files:
    pid = os.path.basename(f).split("_seg")[0]
    pair_map[pid].append(f)

print("Total pairs:", len(pair_map))

# =========================
# INFERENCE
# =========================
pair_votes = defaultdict(list)
pair_probs = defaultdict(list)
pair_true  = {}

with torch.no_grad():
    for pid, files in pair_map.items():
        for f in files:
            d = np.load(f, allow_pickle=True)
            x = d["windows"].astype(np.float32).reshape(256, -1)
            y = int(d["label"].item())

            x = torch.tensor(x).unsqueeze(0).to(DEVICE)

            logits = model(x) / T
            probs = torch.softmax(logits, 1)

            pred = probs.argmax(1).item()

            pair_votes[pid].append(pred)
            pair_probs[pid].append(probs.cpu().numpy()[0])
            pair_true[pid] = y

# =========================
# PER-PAIR SUMMARY
# =========================
rows = []

for pid in sorted(pair_votes):
    votes = pair_votes[pid]
    probs = np.array(pair_probs[pid])
    true  = pair_true[pid]

    c = Counter(votes)
    pred = c.most_common(1)[0][0]

    agreement = c[pred] / len(votes)

    p = np.array([c.get(i, 0) / len(votes) for i in range(NUM_CLASSES)])
    entropy = -np.sum(p * np.log(p + 1e-9))

    mean_conf = probs[:, pred].mean()

    # Confidence rule (you can justify this in thesis)
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
# GLOBAL SUMMARY (FOR REPORTING)
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

