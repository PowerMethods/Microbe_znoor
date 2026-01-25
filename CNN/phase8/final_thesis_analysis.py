#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================
FINAL_CSV = (
    "/vol/projects/znoor/cnn_training/3class_exp/"
    "FINAL_RESULTS/final_pair_predictions.csv"
)

RF_SUMMARY = {
    "Macro F1": 0.7978918688803746,
    "MCC": 0.7203471383525607,
    "ROC-AUC": 0.9312339989759344
}

OUT_DIR = (
    "/vol/projects/znoor/cnn_training/3class_exp/"
    "FINAL_RESULTS/THESIS_ASSETS"
)
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD FINAL RESULTS
# ============================================================
df = pd.read_csv(FINAL_CSV)

df["correct"] = df["pred"] == df["true"]
df["confidence_group"] = np.where(
    df["confident_prediction"], "Confident", "Ambiguous"
)

print("="*60)
print("FINAL PAIR-LEVEL ANALYSIS")
print("="*60)
print("Total pairs:", len(df))
print("Confident pairs:", (df["confidence_group"]=="Confident").sum())
print("Ambiguous pairs:", (df["confidence_group"]=="Ambiguous").sum())

# ============================================================
# 1. ENTROPY vs AGREEMENT PLOT
# ============================================================
plt.figure(figsize=(6,5))

plt.scatter(
    df.loc[~df["correct"], "agreement"],
    df.loc[~df["correct"], "entropy"],
    alpha=0.6,
    label="Incorrect",
)

plt.scatter(
    df.loc[df["correct"], "agreement"],
    df.loc[df["correct"], "entropy"],
    alpha=0.6,
    label="Correct",
)

plt.xlabel("Segment Agreement")
plt.ylabel("Entropy")
plt.title("Uncertainty vs Agreement (Pair Level)")
plt.legend()
plt.grid(alpha=0.3)

FIG1 = f"{OUT_DIR}/entropy_vs_agreement.png"
plt.savefig(FIG1, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", FIG1)

# ============================================================
# 2. CONFIDENT vs AMBIGUOUS SUMMARY
# ============================================================
summary_conf = df.groupby("confidence_group").agg(
    pairs=("pair_id","count"),
    accuracy=("correct","mean"),
    mean_entropy=("entropy","mean"),
    mean_agreement=("agreement","mean")
).reset_index()

CONF_TABLE = f"{OUT_DIR}/confidence_summary.csv"
summary_conf.to_csv(CONF_TABLE, index=False)

print("Saved:", CONF_TABLE)

# ============================================================
# 3. ERROR / AMBIGUITY ANALYSIS
# ============================================================
error_table = df.groupby(
    ["confidence_group","correct"]
).size().reset_index(name="count")

ERR_TABLE = f"{OUT_DIR}/error_breakdown.csv"
error_table.to_csv(ERR_TABLE, index=False)

print("Saved:", ERR_TABLE)

# ============================================================
# 4. CNN vs RF COMPARISON TABLE
# ============================================================
cnn_metrics = {
    "Macro F1": df["correct"].mean(),  # proxy
    "MCC": 0.74,
    "ROC-AUC": 0.91
}

comparison = pd.DataFrame([
    {
        "Model": "CNN256 (Phase-7)",
        "Macro F1": 0.82,
        "MCC": 0.74,
        "ROC-AUC": 0.91,
        "Input features": "Raw genomic segments"
    },
    {
        "Model": "Random Forest",
        "Macro F1": RF_SUMMARY["Macro F1"],
        "MCC": RF_SUMMARY["MCC"],
        "ROC-AUC": RF_SUMMARY["ROC-AUC"],
        "Input features": "Entropy, MI, #segments"
    }
])

COMP_TABLE = f"{OUT_DIR}/cnn_vs_rf_comparison.csv"
comparison.to_csv(COMP_TABLE, index=False)

print("Saved:", COMP_TABLE)

# ============================================================
# 5. FINAL PRINTABLE SUMMARY (FOR RESULTS SECTION)
# ============================================================
acc_all = df["correct"].mean()
acc_conf = df.loc[df["confidence_group"]=="Confident","correct"].mean()

print("\n===== FINAL SUMMARY =====")
print(f"Overall accuracy           : {acc_all:.3f}")
print(f"Accuracy (confident pairs) : {acc_conf:.3f}")
print("==========================")

print("\nFINAL THESIS ANALYSIS COMPLETE")

