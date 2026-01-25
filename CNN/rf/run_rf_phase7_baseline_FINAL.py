#!/usr/bin/env python
"""
RANDOM FOREST BASELINE — PHASE 7 (PAIR LEVEL)

Purpose
-------
This script implements a strong, interpretable Random Forest (RF) baseline
using pair-level summary features derived from the CNN pipeline.

The RF baseline serves as:
1) A non–deep-learning comparison model
2) A feature-engineered alternative to raw CNN representations
3) A sanity check for whether CNN performance arises purely from summary metrics

Input features (pair-level):
- entropy        : prediction uncertainty from segment voting
- MI             : mutual information between samples
- num_segments   : number of CNN segments available for the pair

Evaluation:
- Stratified 5-fold cross-validation
- Macro-F1, MCC, ROC-AUC (OvR)
- Final model trained on full dataset
"""

# =========================
# IMPORTS
# =========================
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    classification_report
)

# =========================
# PATHS (LOCKED FOR PHASE 7)
# =========================

# Pair-level feature table generated after Phase-7 inference
CSV_PATH = (
    "/vol/projects/znoor/cnn_training/3class_exp/"
    "Phase7_Finetune_Calibration/FINAL_ANALYSIS/"
    "pair_level_entropy_mi.csv"
)

# Output directory for all RF artifacts
OUT_DIR = (
    "/vol/projects/znoor/cnn_training/3class_exp/"
    "Phase7_Finetune_Calibration/RF_BASELINE_final"
)
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH)

# Explicitly define feature set
# These are *engineered* features, not learned representations
FEATURES = ["entropy", "MI", "num_segments"]

X = df[FEATURES].values
y = df["true"].values   # true pair-level direction label

print("=" * 60)
print("RF BASELINE — PHASE 7")
print("=" * 60)
print("Total pairs:", len(df))
print("Class counts:\n", df["true"].value_counts())
print("Features used:", FEATURES)
print("=" * 60)

# =========================
# RANDOM FOREST DEFINITION
# =========================
"""
Design choices (thesis justification):

- n_estimators=500:
  Ensures stable ensemble predictions

- max_depth=6:
  Prevents overfitting on low-dimensional features

- min_samples_leaf=5:
  Enforces robustness at leaf nodes

- class_weight='balanced':
  Corrects for class imbalance

- oob_score=True:
  Provides out-of-bag generalization estimate
"""
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=6,
    min_samples_leaf=5,
    class_weight="balanced",
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

# =========================
# STRATIFIED 5-FOLD CV
# =========================
"""
Stratified CV ensures:
- Class proportions preserved in each fold
- Fair evaluation across directionality classes
"""
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fold_metrics = []

for fold, (tr, te) in enumerate(skf.split(X, y), 1):
    rf.fit(X[tr], y[tr])

    # Predictions
    y_pred = rf.predict(X[te])
    y_prob = rf.predict_proba(X[te])

    # Metrics
    acc  = accuracy_score(y[te], y_pred)
    bal  = balanced_accuracy_score(y[te], y_pred)
    f1m  = f1_score(y[te], y_pred, average="macro")
    f1w  = f1_score(y[te], y_pred, average="weighted")
    mcc  = matthews_corrcoef(y[te], y_pred)

    # Multi-class ROC-AUC (One-vs-Rest)
    roc  = roc_auc_score(
        y[te],
        y_prob,
        multi_class="ovr",
        average="macro"
    )

    fold_metrics.append({
        "fold": fold,
        "accuracy": acc,
        "balanced_accuracy": bal,
        "macro_f1": f1m,
        "weighted_f1": f1w,
        "mcc": mcc,
        "roc_auc_macro": roc
    })

    print(
        f"[RF][Fold {fold}] "
        f"Acc={acc:.3f} "
        f"BalAcc={bal:.3f} "
        f"MacroF1={f1m:.3f} "
        f"MCC={mcc:.3f} "
        f"ROC-AUC={roc:.3f}"
    )

# =========================
# CV SUMMARY STATISTICS
# =========================
cv_df = pd.DataFrame(fold_metrics)

cv_mean = cv_df.mean(numeric_only=True)
cv_std  = cv_df.std(numeric_only=True)

print("\n[RF] ===== 5-FOLD CV MEAN =====")
print(cv_mean)

print("\n[RF] ===== 5-FOLD CV STD =====")
print(cv_std)

cv_df.to_csv(f"{OUT_DIR}/rf_cv_folds.csv", index=False)

# =========================
# TRAIN FINAL RF ON FULL DATA
# =========================
"""
Final RF model trained on ALL available pairs.
Used for:
- Feature importance analysis
- Final confusion matrix
- Direct comparison with CNN outputs
"""
rf.fit(X, y)

joblib.dump(
    rf,
    f"{OUT_DIR}/rf_phase7_model.joblib"
)

print("\n[RF] OOB score:", rf.oob_score_)

# =========================
# FEATURE IMPORTANCE
# =========================
"""
Feature importance interpretation:
- entropy        → model uncertainty
- MI             → biological signal similarity
- num_segments   → evidence quantity
"""
imp = pd.Series(
    rf.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

imp.to_csv(
    f"{OUT_DIR}/rf_feature_importance.csv"
)

print("\n[RF] Feature importance:")
print(imp)

# =========================
# FULL-DATA EVALUATION
# =========================
y_pred_full = rf.predict(X)
y_prob_full = rf.predict_proba(X)

cm = confusion_matrix(y, y_pred_full)
np.savetxt(
    f"{OUT_DIR}/rf_confusion_matrix.txt",
    cm,
    fmt="%d"
)

report = classification_report(
    y,
    y_pred_full,
    digits=4
)
with open(
    f"{OUT_DIR}/rf_classification_report.txt",
    "w"
) as f:
    f.write(report)

roc_full = roc_auc_score(
    y,
    y_prob_full,
    multi_class="ovr",
    average="macro"
)

print("\n[RF] ===== FULL DATA =====")
print("Macro ROC-AUC:", roc_full)
print("Confusion Matrix:\n", cm)

print("\nSaved RF baseline outputs to:", OUT_DIR)
