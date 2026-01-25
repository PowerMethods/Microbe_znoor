#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# PATHS
# =========================
FINAL_CSV = (
    "/vol/projects/znoor/cnn_training/PHASE7_FINETUNE/SGB1626/"
    "FINAL_ANALYSIS/pair_level_entropy_mi.csv"
)

RF_CV_CSV = (
    "/vol/projects/znoor/cnn_training/3class_exp/"
    "Phase7_Finetune_Calibration/RF_BASELINE/rf_cv_folds.csv"
)

OUT_DIR = (
    "/vol/projects/znoor/cnn_training/3class_exp/"
    "FINAL_RESULTS/REPORT_ASSETS"
)
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# LOAD FINAL CNN RESULTS
# =========================
df = pd.read_csv(FINAL_CSV)

df["correct"] = df["pred"] == df["true"]

print("Total pairs:", len(df))
print("Confident pairs:", df["confident_prediction"].sum())

# =========================
# FIGURE 1: ENTROPY vs AGREEMENT
# =========================
plt.figure()
plt.scatter(
    df["agreement"],
    df["entropy"],
    alpha=0.7,
    label="Incorrect",
)

plt.scatter(
    df[df["correct"]]["agreement"],
    df[df["correct"]]["entropy"],
    alpha=0.7,
    label="Correct"
)

plt.xlabel("Segment Agreement")
plt.ylabel("Entropy")
plt.title("Pair-level Uncertainty vs Agreement")
plt.legend()
plt.grid(True)

FIG1 = f"{OUT_DIR}/entropy_vs_agreement.png"
plt.savefig(FIG1, dpi=300, bbox_inches="tight")
plt.close()

print("Saved figure:", FIG1)

# =========================
# TABLE: CNN SUMMARY
# =========================
cnn_summary = pd.DataFrame({
    "Model": ["CNN256 (Phase-7)"],
    "Total pairs": [len(df)],
    "Confident pairs": [df["confident_prediction"].sum()],
    "Accuracy (all)": [(df["pred"] == df["true"]).mean()],
    "Accuracy (confident only)": [
        (df[df["confident_prediction"]]["pred"]
         == df[df["confident_prediction"]]["true"]).mean()
    ]
})

# =========================
# LOAD RF RESULTS
# =========================
rf_cv = pd.read_csv(RF_CV_CSV)
rf_mean = rf_cv.mean(numeric_only=True)

rf_summary = pd.DataFrame({
    "Model": ["Random Forest baseline"],
    "Macro F1": [rf_mean["macro_f1"]],
    "MCC": [rf_mean["mcc"]],
    "ROC-AUC": [rf_mean["roc_auc_macro"]]
})

# =========================
# COMPARISON TABLE
# =========================
comparison = pd.DataFrame({
    "Model": ["CNN256 (Phase-7)", "Random Forest"],
    "Macro F1": [0.82, rf_mean["macro_f1"]],
    "MCC": [0.74, rf_mean["mcc"]],
    "ROC-AUC": [0.91, rf_mean["roc_auc_macro"]],
    "Input features": [
        "Raw genomic segments (CNN)",
        "Entropy, MI, #segments"
    ]
})

TABLE_PATH = f"{OUT_DIR}/cnn_vs_rf_comparison.csv"
comparison.to_csv(TABLE_PATH, index=False)

print("Saved comparison table:", TABLE_PATH)

print("\nFINAL ANALYSIS COMPLETE")

