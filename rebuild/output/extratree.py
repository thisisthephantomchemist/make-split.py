# -*- coding: utf-8 -*-
"""
FINAL PIPELINE (your chosen policy):
Model = ExtraTrees
Threshold policy = minimize (FP + FN) on TRAIN-OOF

Outputs:
- final_metrics_extratrees_min_fp_fn.csv
- confusion_matrix_test_raw.csv
- confusion_matrix_test_normalized.csv
- confusion_matrix_test.png
- roc_curve_test.png
- pr_curve_test.png
- threshold_sweep_train_oof.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, roc_curve, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.base import clone


# -----------------------------
# 1) Paths (EDIT THESE)
# -----------------------------
FEATURES_CSV  = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\features_and_labels.csv"
SPLIT_CSV     = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\output\split2080_best.csv"
RFA_FEATS_TXT = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\output\rfa_selected_features.txt"
OUT_DIR       = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\output"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_METRICS   = os.path.join(OUT_DIR, "final_metrics_extratrees_min_fp_fn.csv")
OUT_CM_RAW    = os.path.join(OUT_DIR, "confusion_matrix_test_raw.csv")
OUT_CM_NORM   = os.path.join(OUT_DIR, "confusion_matrix_test_normalized.csv")
OUT_CM_PNG    = os.path.join(OUT_DIR, "confusion_matrix_test.png")
OUT_ROC_PNG   = os.path.join(OUT_DIR, "roc_curve_test.png")
OUT_PR_PNG    = os.path.join(OUT_DIR, "pr_curve_test.png")
OUT_THR_SWEEP = os.path.join(OUT_DIR, "threshold_sweep_train_oof.csv")


# -----------------------------
# 2) Helpers
# -----------------------------
def detect_split_column(split_df: pd.DataFrame) -> str:
    for c in split_df.columns:
        if c.lower() in ("split","set","subset"):
            return c
    candidates = [c for c in split_df.columns if ("split" in c.lower()) or ("set" in c.lower())]
    if not candidates:
        raise ValueError("Could not detect split column.")
    return candidates[0]


def oof_probs(estimator, Xtr, ytr, cv):
    probs = np.zeros(len(ytr), dtype=float)
    for tr_idx, va_idx in cv.split(Xtr, ytr):
        est = clone(estimator)
        est.fit(Xtr.iloc[tr_idx], ytr[tr_idx])
        probs[va_idx] = est.predict_proba(Xtr.iloc[va_idx])[:, 1]
    return probs


def best_threshold_min_fp_fn(y_true, probs, grid=501):
    thr_grid = np.linspace(0, 1, grid)
    best = None
    rows = []
    for t in thr_grid:
        yhat = (probs >= t).astype(int)
        TN, FP, FN, TP = confusion_matrix(y_true, yhat, labels=[0,1]).ravel()
        err = FP + FN
        rows.append([t, TN, FP, FN, TP, err])

        if (best is None) or (err < best["err"]):
            best = {"thr": float(t), "TN": TN, "FP": FP, "FN": FN, "TP": TP, "err": int(err)}

    sweep = pd.DataFrame(rows, columns=["thr","TN","FP","FN","TP","FP_plus_FN"])
    return best, sweep


# -----------------------------
# 3) Load data
# -----------------------------
df = pd.read_csv(FEATURES_CSV)
spl = pd.read_csv(SPLIT_CSV)

df["y"] = df["water_label"].map(lambda v: 1 if v in (3,4) else 0)

split_col = detect_split_column(spl)
m = df.merge(spl[["MOF_name", split_col]], on="MOF_name", how="inner").rename(columns={split_col:"split"})

train_mask = m["split"].astype(str).str.lower().str.startswith("train").values
test_mask  = m["split"].astype(str).str.lower().str.startswith("test").values

y = m["y"].astype(int).values

with open(RFA_FEATS_TXT, "r", encoding="utf-8") as f:
    feats = [line.strip() for line in f if line.strip()]

X = m[feats].apply(pd.to_numeric, errors="coerce")
X_train, y_train = X.loc[train_mask], y[train_mask]
X_test,  y_test  = X.loc[test_mask],  y[test_mask]

print("Train:", X_train.shape, "| Test:", X_test.shape, "| Selected features:", len(feats))


# -----------------------------
# 4) Model: ExtraTrees
# -----------------------------
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", ExtraTreesClassifier(
        n_estimators=800,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
        class_weight=None
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# -----------------------------
# 5) Threshold selection on TRAIN-OOF (min FP+FN)
# -----------------------------
p_oof = oof_probs(model, X_train, y_train, cv)
auc_oof = roc_auc_score(y_train, p_oof)

best_thr, sweep_df = best_threshold_min_fp_fn(y_train, p_oof, grid=501)
thr = best_thr["thr"]

sweep_df.to_csv(OUT_THR_SWEEP, index=False)
print("Saved:", OUT_THR_SWEEP)

print("\nChosen threshold (TRAIN-OOF min FP+FN):", thr)
print("TRAIN-OOF counts @thr:", {k: best_thr[k] for k in ["TN","FP","FN","TP","err"]})
print("TRAIN-OOF ROC-AUC:", round(auc_oof, 4))


# -----------------------------
# 6) Fit on full train, evaluate once on TEST
# -----------------------------
model.fit(X_train, y_train)
p_test = model.predict_proba(X_test)[:,1]
y_pred = (p_test >= thr).astype(int)

auc_test = roc_auc_score(y_test, p_test)
f1_test  = f1_score(y_test, y_pred)

TN, FP, FN, TP = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()

print("\n=== TEST RESULTS (FINAL) ===")
print("TEST ROC-AUC:", round(auc_test, 4))
print("TEST F1:", round(f1_test, 4))
print("Confusion (TN,FP,FN,TP):", TN, FP, FN, TP, "| FP+FN:", FP+FN)

# Save metrics
metrics = pd.DataFrame([{
    "model": "ExtraTrees",
    "policy": "min(FP+FN) on TRAIN-OOF",
    "chosen_threshold": thr,
    "train_oof_roc_auc": auc_oof,
    "test_roc_auc": auc_test,
    "test_f1": f1_test,
    "TN": TN, "FP": FP, "FN": FN, "TP": TP,
    "FP_plus_FN": FP + FN
}])
metrics.to_csv(OUT_METRICS, index=False)
print("Saved:", OUT_METRICS)

# Save confusion matrices
cm = np.array([[TN, FP],[FN, TP]])
cm_raw = pd.DataFrame(cm, index=["True Unstable(0)", "True Stable(1)"],
                         columns=["Pred Unstable(0)", "Pred Stable(1)"])
cm_raw.to_csv(OUT_CM_RAW)

cm_norm = cm / cm.sum(axis=1, keepdims=True)
cm_norm_df = pd.DataFrame(cm_norm, index=cm_raw.index, columns=cm_raw.columns)
cm_norm_df.to_csv(OUT_CM_NORM)

print("Saved:", OUT_CM_RAW)
print("Saved:", OUT_CM_NORM)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Unstable(0)", "Stable(1)"])
fig, ax = plt.subplots(figsize=(5,4))
disp.plot(ax=ax, values_format="d", colorbar=False)
ax.set_title(f"Confusion Matrix TEST (thr={thr:.3f})")
plt.tight_layout()
plt.savefig(OUT_CM_PNG, dpi=200)
plt.close(fig)
print("Saved:", OUT_CM_PNG)

# ROC curve plot
fpr, tpr, _ = roc_curve(y_test, p_test)
plt.figure(figsize=(6.5,5))
plt.plot(fpr, tpr, label="ExtraTrees")
plt.plot([0,1],[0,1], linestyle="--", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve on TEST (ExtraTrees)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_ROC_PNG, dpi=200)
plt.close()
print("Saved:", OUT_ROC_PNG)

# PR curve plot
prec, rec, _ = precision_recall_curve(y_test, p_test)
plt.figure(figsize=(6.5,5))
plt.plot(rec, prec, label="ExtraTrees")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve on TEST (ExtraTrees)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PR_PNG, dpi=200)
plt.close()
print("Saved:", OUT_PR_PNG)
