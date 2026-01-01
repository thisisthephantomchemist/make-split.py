# -*- coding: utf-8 -*-
"""
Paper-like model: RFA-style feature selection + tuned Random Forest
Target: water stability 2-class (1&2=0, 3&4=1)

Workflow:
1) Load features_and_labels.csv + fixed split file (train/test)
2) Rank features by RF importance (inside CV-safe training on train only)
3) RFA-style add features in batches, evaluate via CV ROC-AUC on train
4) Pick best #features
5) Tune RF hyperparams on selected features (CV on train only)
6) Fit final model on full train, evaluate once on test
7) Save artifacts (feature list, curves, metrics, ROC plot)

Outputs:
- rfa_selected_features.txt
- rfa_curve.csv
- rf_rfa_metrics.csv
- test_roc_curve_rfa_rf.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from sklearn.base import clone


# -----------------------------
# 1) Paths (EDIT THESE)
# -----------------------------
FEATURES_CSV = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\features_and_labels.csv"
SPLIT_CSV    = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\output\split2080_best.csv"
OUT_DIR      = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\output"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_FEATS_TXT = os.path.join(OUT_DIR, "rfa_selected_features.txt")
OUT_RFA_CURVE = os.path.join(OUT_DIR, "rfa_curve.csv")
OUT_METRICS   = os.path.join(OUT_DIR, "rf_rfa_metrics.csv")
OUT_ROC_PNG   = os.path.join(OUT_DIR, "test_roc_curve_rfa_rf.png")


# -----------------------------
# 2) Config
# -----------------------------
RANDOM_STATE = 42
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# RFA settings
BATCH_SIZE = 5          # هر بار چند ویژگی اضافه شود (برای سرعت)
MAX_FEATURES = 120      # سقف تعداد ویژگی‌هایی که بررسی می‌کنیم

# Tuning settings
N_ITER_TUNING = 25      # تعداد نمونه‌ها در RandomizedSearch (برای سرعت/دقت)
F1_GRID = 301           # grid برای انتخاب threshold F1


# -----------------------------
# 3) Helper functions
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
    """OOF probabilities for class=1 without joblib overhead."""
    proba = np.zeros(len(ytr), dtype=float)
    for tr_idx, va_idx in cv.split(Xtr, ytr):
        est = clone(estimator)
        est.fit(Xtr.iloc[tr_idx], ytr[tr_idx])
        proba[va_idx] = est.predict_proba(Xtr.iloc[va_idx])[:, 1]
    return proba


def best_f1_threshold(y_true, probs, grid=301):
    thr = np.linspace(0, 1, grid)
    f1s = np.array([f1_score(y_true, (probs >= t).astype(int)) for t in thr])
    j = int(f1s.argmax())
    return float(thr[j]), float(f1s[j])


def cv_auc_for_features(Xtr, ytr, feat_list):
    """Evaluate a simple RF (fixed params) on selected features using CV ROC-AUC."""
    base_rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_features="sqrt",
            class_weight="balanced"
        ))
    ])
    probs = oof_proba(base_rf, Xtr[feat_list], ytr, cv)
    return roc_auc_score(ytr, probs)


# -----------------------------
# 4) Load data + split
# -----------------------------
df = pd.read_csv(FEATURES_CSV)
spl = pd.read_csv(SPLIT_CSV)

df["y"] = df["water_label"].map(lambda v: 1 if v in (3, 4) else 0)

split_col = detect_split_column(spl)
m = df.merge(spl[["MOF_name", split_col]], on="MOF_name", how="inner").rename(columns={split_col: "split"})

exclude = {"MOF_name", "data_set", "water_label", "acid_label", "base_label", "boiling_label", "y", "split"}
feature_cols = [c for c in m.columns if c not in exclude]

X = m[feature_cols].apply(pd.to_numeric, errors="coerce")
y = m["y"].astype(int).values

train_mask = m["split"].astype(str).str.lower().str.startswith("train").values
test_mask  = m["split"].astype(str).str.lower().str.startswith("test").values

X_train, y_train = X.loc[train_mask], y[train_mask]
X_test,  y_test  = X.loc[test_mask],  y[test_mask]

print(f"Train: {train_mask.sum()} | Test: {test_mask.sum()} | n_features: {X.shape[1]}")


# -----------------------------
# 5) Step 1: Feature ranking by RF importance (train only)
# -----------------------------
rank_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(
        n_estimators=800,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",
        class_weight="balanced"
    ))
])

rank_model.fit(X_train, y_train)
importances = rank_model.named_steps["rf"].feature_importances_
ranked_features = [f for f, _ in sorted(zip(feature_cols, importances), key=lambda t: t[1], reverse=True)]

# Limit features for RFA search (speed)
ranked_features = ranked_features[:min(MAX_FEATURES, len(ranked_features))]


# -----------------------------
# 6) Step 2: RFA-style selection (add features in batches)
# -----------------------------
selected = []
curve = []  # (n_features, cv_auc)

best_auc = -1
best_k = 0

for i in range(0, len(ranked_features), BATCH_SIZE):
    batch = ranked_features[i:i+BATCH_SIZE]
    selected.extend(batch)

    auc_cv = cv_auc_for_features(X_train, y_train, selected)
    curve.append((len(selected), auc_cv))
    print(f"RFA step: {len(selected):3d} features | CV ROC-AUC = {auc_cv:.4f}")

    if auc_cv > best_auc:
        best_auc = auc_cv
        best_k = len(selected)

# Best feature set
best_features = selected[:best_k]
print("\n=== RFA selected ===")
print(f"Best #features: {best_k} | Best CV ROC-AUC: {best_auc:.4f}")

# Save curve + features
pd.DataFrame(curve, columns=["n_features", "cv_roc_auc"]).to_csv(OUT_RFA_CURVE, index=False)
with open(OUT_FEATS_TXT, "w", encoding="utf-8") as f:
    for feat in best_features:
        f.write(feat + "\n")

print("Saved:", OUT_RFA_CURVE)
print("Saved:", OUT_FEATS_TXT)


# -----------------------------
# 7) Step 3: Tune RF on selected features (CV on train only)
# -----------------------------
rf_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
])

param_dist = {
    "rf__n_estimators": [400, 600, 900, 1200],
    "rf__max_depth": [None, 8, 12, 16, 24, 32],
    "rf__max_features": ["sqrt", 0.3, 0.5, 0.7],
    "rf__min_samples_split": [2, 5, 10],
    "rf__min_samples_leaf": [1, 2, 4],
    "rf__class_weight": [None, "balanced"]
}

rs = RandomizedSearchCV(
    rf_pipe,
    param_distributions=param_dist,
    n_iter=N_ITER_TUNING,
    scoring="roc_auc",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)
rs.fit(X_train[best_features], y_train)

best_model = rs.best_estimator_
print("\nBest params:", rs.best_params_)
print("Best CV ROC-AUC (tuning):", rs.best_score_)


# -----------------------------
# 8) Step 4: Final evaluation (OOF on train + one-shot test)
# -----------------------------
# OOF on train (for threshold selection & honest train metrics)
probs_oof = oof_proba(best_model, X_train[best_features], y_train, cv)
auc_cv = roc_auc_score(y_train, probs_oof)
thr, f1_cv_best = best_f1_threshold(y_train, probs_oof, grid=F1_GRID)
f1_cv_05 = f1_score(y_train, (probs_oof >= 0.5).astype(int))

# Fit on full train
best_model.fit(X_train[best_features], y_train)
probs_te = best_model.predict_proba(X_test[best_features])[:, 1]

auc_te = roc_auc_score(y_test, probs_te)
f1_te_05 = f1_score(y_test, (probs_te >= 0.5).astype(int))
f1_te_thr = f1_score(y_test, (probs_te >= thr).astype(int))

metrics = pd.DataFrame([{
    "model": "RFA + Tuned RF",
    "best_k_features": best_k,
    "CV ROC-AUC (OOF)": auc_cv,
    "CV F1@0.5 (OOF)": f1_cv_05,
    "CV Best-F1 (OOF)": f1_cv_best,
    "Chosen threshold (CV)": thr,
    "TEST ROC-AUC": auc_te,
    "TEST F1@0.5": f1_te_05,
    "TEST F1@CV-threshold": f1_te_thr,
    "best_params": str(rs.best_params_)
}])

print("\n=== FINAL METRICS ===")
print(metrics.to_string(index=False))

metrics.to_csv(OUT_METRICS, index=False)
print("\nSaved:", OUT_METRICS)


# -----------------------------
# 9) Save ROC curve plot (test)
# -----------------------------
fpr, tpr, _ = roc_curve(y_test, probs_te)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label="RFA + Tuned RF")
plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve on TEST (RFA + Tuned RF)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_ROC_PNG, dpi=200)
print("Saved:", OUT_ROC_PNG)
