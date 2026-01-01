# -*- coding: utf-8 -*-
"""
Find thresholds that try to reduce BOTH FP and FN (as much as possible)
using TRAIN OOF predictions only, then evaluate once on TEST.

Criteria:
1) Minimize FP + FN   (total mistakes)
2) Minimize max(FPR, FNR)  (balance the two error types)
3) Optional: Reject option (two-threshold) to lower BOTH among decided samples
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.base import clone

# -----------------------------
# Paths (EDIT)
# -----------------------------
FEATURES_CSV = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\features_and_labels.csv"
SPLIT_CSV    = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\output\split2080_best.csv"
RFA_FEATS_TXT= r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\output\rfa_selected_features.txt"
OUT_DIR      = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\output"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_THR_CURVE = os.path.join(OUT_DIR, "threshold_fp_fn_curve.csv")
OUT_PLOT      = os.path.join(OUT_DIR, "threshold_fp_fn_plot.png")

# -----------------------------
# Load
# -----------------------------
df = pd.read_csv(FEATURES_CSV)
spl = pd.read_csv(SPLIT_CSV)

df["y"] = df["water_label"].map(lambda v: 1 if v in (3,4) else 0)

# detect split column
split_col = None
for c in spl.columns:
    if c.lower() in ("split","set","subset"):
        split_col = c
        break
if split_col is None:
    split_col = [c for c in spl.columns if ("split" in c.lower()) or ("set" in c.lower())][0]

m = df.merge(spl[["MOF_name", split_col]], on="MOF_name", how="inner").rename(columns={split_col:"split"})
train_mask = m["split"].astype(str).str.lower().str.startswith("train").values
test_mask  = m["split"].astype(str).str.lower().str.startswith("test").values

y = m["y"].astype(int).values

with open(RFA_FEATS_TXT, "r", encoding="utf-8") as f:
    feats = [line.strip() for line in f if line.strip()]

X = m[feats].apply(pd.to_numeric, errors="coerce")

X_train, y_train = X.loc[train_mask], y[train_mask]
X_test,  y_test  = X.loc[test_mask],  y[test_mask]

# -----------------------------
# Your tuned RF params (from your run)
# -----------------------------
best_params = dict(
    n_estimators=400,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    max_depth=32,
    class_weight=None,
    random_state=42,
    n_jobs=-1
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(**best_params))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def oof_probs(estimator, Xtr, ytr, cv):
    probs = np.zeros(len(ytr), dtype=float)
    for tr_idx, va_idx in cv.split(Xtr, ytr):
        est = clone(estimator)
        est.fit(Xtr.iloc[tr_idx], ytr[tr_idx])
        probs[va_idx] = est.predict_proba(Xtr.iloc[va_idx])[:,1]
    return probs

# -----------------------------
# 1) OOF probs on TRAIN
# -----------------------------
p_oof = oof_probs(model, X_train, y_train, cv)
auc_oof = roc_auc_score(y_train, p_oof)
print("Train OOF ROC-AUC:", round(auc_oof, 4))

# -----------------------------
# 2) Sweep thresholds and compute FP/FN (and rates)
# -----------------------------
thr_grid = np.linspace(0, 1, 501)
rows = []

P = int((y_train==1).sum())
N = int((y_train==0).sum())

for t in thr_grid:
    yhat = (p_oof >= t).astype(int)
    # cm order: [[TN, FP],[FN, TP]] for labels [0,1]
    TN, FP, FN, TP = confusion_matrix(y_train, yhat, labels=[0,1]).ravel()

    FPR = FP / (FP + TN) if (FP + TN) else 0.0
    FNR = FN / (FN + TP) if (FN + TP) else 0.0

    rows.append([t, FP, FN, FPR, FNR, FP+FN, max(FPR, FNR)])

curve = pd.DataFrame(rows, columns=["thr","FP","FN","FPR","FNR","FP_plus_FN","max_FPR_FNR"])
curve.to_csv(OUT_THR_CURVE, index=False)
print("Saved:", OUT_THR_CURVE)

# Criteria A: minimize FP+FN
best_A = curve.loc[curve["FP_plus_FN"].idxmin()]

# Criteria B: minimize max(FPR,FNR)  (balance)
best_B = curve.loc[curve["max_FPR_FNR"].idxmin()]

print("\nBest thr (min FP+FN):", float(best_A["thr"]), "| FP:", int(best_A["FP"]), "FN:", int(best_A["FN"]))
print("Best thr (min max(FPR,FNR)):", float(best_B["thr"]), "| FP:", int(best_B["FP"]), "FN:", int(best_B["FN"]))

# -----------------------------
# 3) Evaluate chosen thresholds ON TEST once
# -----------------------------
model.fit(X_train, y_train)
p_test = model.predict_proba(X_test)[:,1]

def eval_on_test(thr):
    yhat = (p_test >= thr).astype(int)
    TN, FP, FN, TP = confusion_matrix(y_test, yhat, labels=[0,1]).ravel()
    return TN, FP, FN, TP

TN_A, FP_A, FN_A, TP_A = eval_on_test(float(best_A["thr"]))
TN_B, FP_B, FN_B, TP_B = eval_on_test(float(best_B["thr"]))

print("\nTEST @ thr(min FP+FN):", float(best_A["thr"]))
print("TN FP FN TP =", TN_A, FP_A, FN_A, TP_A)

print("\nTEST @ thr(min max(FPR,FNR)):", float(best_B["thr"]))
print("TN FP FN TP =", TN_B, FP_B, FN_B, TP_B)

# -----------------------------
# 4) Plot FP and FN rates vs threshold (TRAIN-OOF)
# -----------------------------
plt.figure(figsize=(7,5))
plt.plot(curve["thr"], curve["FPR"], label="FPR (FP rate)")
plt.plot(curve["thr"], curve["FNR"], label="FNR (FN rate)")
plt.axvline(float(best_A["thr"]), linestyle="--", label="thr: min(FP+FN)")
plt.axvline(float(best_B["thr"]), linestyle="--", label="thr: min max(FPR,FNR)")
plt.xlabel("Threshold")
plt.ylabel("Rate")
plt.title("TRAIN-OOF error rates vs threshold")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=200)
print("Saved:", OUT_PLOT)
