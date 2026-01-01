# -*- coding: utf-8 -*-
"""
Compare models to reduce BOTH FP and FN with a CONSTRAINED threshold:
- Tuned RandomForest (your current best)
- ExtraTreesClassifier
- HistGradientBoostingClassifier

Threshold policies (chosen on TRAIN-OOF only):
1) thr=0.5
2) min(FP+FN)
3) minimax (min max(FPR,FNR))
4) constrained: minimize FP subject to FN <= FN_CAP  (counts on TRAIN-OOF)

Outputs:
- model_comparison_metrics.csv
- confusion_matrices_test.csv
- roc_curves_test.png
- threshold_choices_train_oof.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, roc_curve, confusion_matrix
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

OUT_METRICS   = os.path.join(OUT_DIR, "model_comparison_metrics.csv")
OUT_CM        = os.path.join(OUT_DIR, "confusion_matrices_test.csv")
OUT_ROC_PNG   = os.path.join(OUT_DIR, "roc_curves_test.png")
OUT_THR_CH    = os.path.join(OUT_DIR, "threshold_choices_train_oof.csv")


# -----------------------------
# 2) Config
# -----------------------------
FN_CAP = 10  # سقف FN روی TRAIN-OOF برای threshold قیددار (قابل تغییر)
RANDOM_STATE = 42
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# -----------------------------
# 3) Helpers
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


def cm_counts(y_true, probs, thr):
    yhat = (probs >= thr).astype(int)
    TN, FP, FN, TP = confusion_matrix(y_true, yhat, labels=[0,1]).ravel()
    return TN, FP, FN, TP


def best_threshold_min_fp_fn(y_true, probs, grid=501):
    thr_grid = np.linspace(0, 1, grid)
    best = None
    for t in thr_grid:
        TN, FP, FN, TP = cm_counts(y_true, probs, t)
        err = FP + FN
        if (best is None) or (err < best["err"]):
            best = {"thr": float(t), "TN": TN, "FP": FP, "FN": FN, "TP": TP, "err": int(err)}
    return best


def best_threshold_minimax(y_true, probs, grid=501):
    thr_grid = np.linspace(0, 1, grid)
    best = None
    for t in thr_grid:
        TN, FP, FN, TP = cm_counts(y_true, probs, t)
        FPR = FP / (FP + TN) if (FP + TN) else 0.0
        FNR = FN / (FN + TP) if (FN + TP) else 0.0
        score = max(FPR, FNR)
        if (best is None) or (score < best["score"]):
            best = {"thr": float(t), "TN": TN, "FP": FP, "FN": FN, "TP": TP, "score": float(score)}
    return best


def best_threshold_fp_min_given_fn_cap(y_true, probs, fn_cap, grid=501):
    """
    Constrained threshold:
    minimize FP subject to FN <= fn_cap
    If no threshold satisfies FN cap, returns the threshold with minimal FN (fallback).
    """
    thr_grid = np.linspace(0, 1, grid)
    best = None

    # First pass: feasible thresholds
    for t in thr_grid:
        TN, FP, FN, TP = cm_counts(y_true, probs, t)
        if FN <= fn_cap:
            if (best is None) or (FP < best["FP"]):
                best = {"thr": float(t), "TN": TN, "FP": FP, "FN": FN, "TP": TP, "feasible": True}

    if best is not None:
        return best

    # Fallback: no feasible threshold -> choose minimal FN
    best2 = None
    for t in thr_grid:
        TN, FP, FN, TP = cm_counts(y_true, probs, t)
        if (best2 is None) or (FN < best2["FN"]):
            best2 = {"thr": float(t), "TN": TN, "FP": FP, "FN": FN, "TP": TP, "feasible": False}
    return best2


def eval_on_test(model, X_train, y_train, X_test, y_test, thr):
    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    TN, FP, FN, TP = cm_counts(y_test, p, thr)
    auc = roc_auc_score(y_test, p)
    f1 = f1_score(y_test, (p >= thr).astype(int))
    return auc, f1, TN, FP, FN, TP, p


# -----------------------------
# 4) Load data
# -----------------------------
df = pd.read_csv(FEATURES_CSV)
spl = pd.read_csv(SPLIT_CSV)

# Binary target: 1&2 -> 0, 3&4 -> 1
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

print("Train:", X_train.shape, "| Test:", X_test.shape, "| FN_CAP:", FN_CAP)


# -----------------------------
# 5) Define models
# -----------------------------
rf_params = dict(
    n_estimators=400,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    max_depth=32,
    class_weight=None,
    random_state=42,
    n_jobs=-1
)

models = {
    "Tuned RF": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(**rf_params))
    ]),
    "ExtraTrees": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", ExtraTreesClassifier(
            n_estimators=800,
            random_state=42,
            n_jobs=-1,
            max_features="sqrt",
            class_weight=None
        ))
    ]),
    "HistGB": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            random_state=42,
            max_depth=6,
            learning_rate=0.05,
            max_iter=400
        ))
    ])
}


# -----------------------------
# 6) Evaluate OOF + choose thresholds on TRAIN only
# -----------------------------
metrics_rows = []
cm_rows = []
roc_data = {}
thr_choice_rows = []

for name, model in models.items():
    # TRAIN OOF probs
    p_oof = oof_probs(model, X_train, y_train, cv)
    auc_oof = roc_auc_score(y_train, p_oof)

    # Thresholds (TRAIN only)
    thr_05 = 0.5
    bestA = best_threshold_min_fp_fn(y_train, p_oof, grid=501)
    bestB = best_threshold_minimax(y_train, p_oof, grid=501)
    bestC = best_threshold_fp_min_given_fn_cap(y_train, p_oof, fn_cap=FN_CAP, grid=501)

    thr_choice_rows.append({
        "model": name,
        "train_oof_auc": auc_oof,
        "thr_0.5": thr_05,
        "thr_min_fp_fn": bestA["thr"],
        "train_oof_FP_min_fp_fn": bestA["FP"],
        "train_oof_FN_min_fp_fn": bestA["FN"],
        "thr_minimax": bestB["thr"],
        "train_oof_score_minimax(maxFPRFNR)": bestB["score"],
        "thr_constrained_FPmin_given_FNcap": bestC["thr"],
        "train_oof_FP_constrained": bestC["FP"],
        "train_oof_FN_constrained": bestC["FN"],
        "constrained_feasible": bestC.get("feasible", None)
    })

    # Evaluate on TEST for each policy
    policies = [
        ("thr_0.5", thr_05),
        ("thr_min_fp_fn", bestA["thr"]),
        ("thr_minimax", bestB["thr"]),
        ("thr_constrained", bestC["thr"])
    ]

    # fit once for ROC curve (we'll call eval which fits; OK for reproducibility)
    # For ROC curve, we can fit once more:
    model.fit(X_train, y_train)
    p_test = model.predict_proba(X_test)[:,1]
    roc_data[name] = roc_curve(y_test, p_test)

    # Base test AUC (threshold independent)
    test_auc = roc_auc_score(y_test, p_test)

    # Save policy-based metrics
    row = {"model": name, "train_oof_roc_auc": auc_oof, "test_roc_auc": test_auc}

    for tag, thr in policies:
        auc_t, f1_t, TN, FP, FN, TP, _ = eval_on_test(model, X_train, y_train, X_test, y_test, thr)
        row[f"test_f1_{tag}"] = f1_t
        row[f"test_FP_{tag}"] = FP
        row[f"test_FN_{tag}"] = FN

        cm_rows.append({
            "model": name,
            "policy": tag,
            "threshold": thr,
            "TN": TN, "FP": FP, "FN": FN, "TP": TP
        })

    metrics_rows.append(row)

# Save outputs
pd.DataFrame(metrics_rows).to_csv(OUT_METRICS, index=False)
pd.DataFrame(cm_rows).to_csv(OUT_CM, index=False)
pd.DataFrame(thr_choice_rows).to_csv(OUT_THR_CH, index=False)

print("Saved:", OUT_METRICS)
print("Saved:", OUT_CM)
print("Saved:", OUT_THR_CH)

# Plot ROC curves
plt.figure(figsize=(7,5))
for name, (fpr, tpr, _) in roc_data.items():
    plt.plot(fpr, tpr, label=name)
plt.plot([0,1],[0,1], linestyle="--", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves on TEST (same split & features)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_ROC_PNG, dpi=200)
print("Saved:", OUT_ROC_PNG)
