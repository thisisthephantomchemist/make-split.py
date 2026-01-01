# -*- coding: utf-8 -*-
"""
Run baselines + RandomForest for MOF water stability (2-class)
using a fixed 80/20 split file (best split: random_state=105).

Metrics prioritized: ROC-AUC + F1 (authors' focus).
Outputs:
- water_stability_metrics_baselines.csv
- test_roc_curves.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from sklearn.base import clone


# -----------------------------
# 1) Paths (EDIT THESE)
# -----------------------------
FEATURES_CSV = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\features_and_labels.csv"
SPLIT_CSV    = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\output\split2080_best.csv"  # <-- the best split you created
OUT_DIR      = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\output"
os.makedirs(OUT_DIR, exist_ok=True)

METRICS_OUT  = os.path.join(OUT_DIR, "water_stability_metrics_baselines.csv")
ROC_OUT      = os.path.join(OUT_DIR, "test_roc_curves.png")


# -----------------------------
# 2) Helpers
# -----------------------------
def detect_split_column(split_df: pd.DataFrame) -> str:
    for c in split_df.columns:
        if c.lower() in ("split", "set", "subset"):
            return c
    candidates = [c for c in split_df.columns if ("split" in c.lower()) or ("set" in c.lower())]
    if not candidates:
        raise ValueError("Could not detect split column in split CSV.")
    return candidates[0]


def oof_proba(estimator, Xtr, ytr, cv):
    """Fast out-of-fold predicted probabilities (no joblib overhead)."""
    proba = np.zeros(len(ytr), dtype=float)
    for tr_idx, va_idx in cv.split(Xtr, ytr):
        est = clone(estimator)
        est.fit(Xtr.iloc[tr_idx], ytr[tr_idx])
        proba[va_idx] = est.predict_proba(Xtr.iloc[va_idx])[:, 1]
    return proba


def best_f1_threshold(y_true, probs, grid=301):
    """Pick threshold maximizing F1 on provided probs."""
    thr = np.linspace(0, 1, grid)
    f1s = np.array([f1_score(y_true, (probs >= t).astype(int)) for t in thr])
    j = int(f1s.argmax())
    return float(thr[j]), float(f1s[j])


# -----------------------------
# 3) Load data + split
# -----------------------------
df = pd.read_csv(FEATURES_CSV)
spl = pd.read_csv(SPLIT_CSV)

# Binary target: labels 1&2 unstable->0 ; labels 3&4 stable->1
df["y"] = df["water_label"].map(lambda v: 1 if v in (3, 4) else 0)

split_col = detect_split_column(spl)

m = df.merge(spl[["MOF_name", split_col]], on="MOF_name", how="inner")
m = m.rename(columns={split_col: "split"})

exclude = {"MOF_name", "data_set", "water_label", "acid_label", "base_label", "boiling_label", "y", "split"}
feature_cols = [c for c in m.columns if c not in exclude]

X = m[feature_cols].apply(pd.to_numeric, errors="coerce")
y = m["y"].astype(int).values

train_mask = m["split"].astype(str).str.lower().str.startswith("train").values
test_mask  = m["split"].astype(str).str.lower().str.startswith("test").values

X_train, y_train = X.loc[train_mask], y[train_mask]
X_test,  y_test  = X.loc[test_mask], y[test_mask]

print(f"Total: {len(m)} | Train: {train_mask.sum()} | Test: {test_mask.sum()} | n_features: {X.shape[1]}")
print(f"Train stable%: {y_train.mean()*100:.2f} | Test stable%: {y_test.mean()*100:.2f}")


# -----------------------------
# 4) CV setup
# -----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# -----------------------------
# 5) Models (baseline -> stronger)
# -----------------------------
models = {
    "Dummy (most_frequent)": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DummyClassifier(strategy="most_frequent"))
    ]),
    "LogReg (C=1)": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=4000, solver="liblinear"))
    ]),
    "RandomForest (200 trees)": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            max_features="sqrt",
            class_weight="balanced"
        ))
    ])
}


# -----------------------------
# 6) Evaluate (OOF on train + one-shot test)
# -----------------------------
rows = []
roc_data = {}

for name, est in models.items():
    # OOF on training (for honest CV metrics + threshold selection)
    probs_oof = oof_proba(est, X_train, y_train, cv)
    auc_cv = roc_auc_score(y_train, probs_oof)
    f1_cv_05 = f1_score(y_train, (probs_oof >= 0.5).astype(int))
    thr, f1_cv_best = best_f1_threshold(y_train, probs_oof, grid=301)

    # Fit on full train, evaluate once on test
    est.fit(X_train, y_train)
    probs_te = est.predict_proba(X_test)[:, 1]

    auc_te = roc_auc_score(y_test, probs_te)
    f1_te_05 = f1_score(y_test, (probs_te >= 0.5).astype(int))
    f1_te_thr = f1_score(y_test, (probs_te >= thr).astype(int))

    rows.append([
        name,
        auc_cv, f1_cv_05, f1_cv_best, thr,
        auc_te, f1_te_05, f1_te_thr
    ])

    roc_data[name] = roc_curve(y_test, probs_te)

metrics_df = pd.DataFrame(rows, columns=[
    "Model",
    "CV ROC-AUC (OOF)",
    "CV F1@0.5 (OOF)",
    "CV Best-F1 (OOF)",
    "Chosen threshold (from CV)",
    "TEST ROC-AUC",
    "TEST F1@0.5",
    "TEST F1@CV-threshold"
]).sort_values("TEST ROC-AUC", ascending=False)

print("\n=== Metrics ===")
print(metrics_df.to_string(index=False))

metrics_df.to_csv(METRICS_OUT, index=False)
print("\nSaved:", METRICS_OUT)


# -----------------------------
# 7) Plot ROC curves on test
# -----------------------------
plt.figure(figsize=(7, 5))
for name in ["LogReg (C=1)", "RandomForest (200 trees)"]:
    fpr, tpr, _ = roc_data[name]
    plt.plot(fpr, tpr, label=name)

plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves on TEST set (fixed 80/20 split)")
plt.legend()
plt.tight_layout()
plt.savefig(ROC_OUT, dpi=200)
print("Saved:", ROC_OUT)
