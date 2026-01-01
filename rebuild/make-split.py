# -*- coding: utf-8 -*-
"""
Better 80/20 split selection:
- Generate many candidate stratified splits
- Choose the one with lowest adversarial AUC (closest to 0.5),
  while keeping class/dataset/difficulty balanced.

Output:
- split2080_best.csv
"""

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.base import clone


# -----------------------------
# 1) Paths (EDIT THESE)
# -----------------------------
FEATURES_CSV = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\features_and_labels.csv"
OUT_DIR      = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\output"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_SPLIT_CSV = os.path.join(OUT_DIR, "split2080_best.csv")


# -----------------------------
# 2) Config
# -----------------------------
BASE_RANDOM_STATE = 42
TEST_SIZE = 0.20

K_NEIGHBORS = 15
N_BINS = 5

N_CANDIDATES = 80          # تعداد splitهای کاندید
ADV_TARGET = 0.50          # هدف ما نزدیک 0.5
MAX_CLASS_RATE_DIFF = 0.01 # حداکثر اختلاف نسبت stable بین train/test
MAX_DIFF_MEAN_DIFF = 0.01  # حداکثر اختلاف میانگین difficulty (تقریباً)


# -----------------------------
# 3) Helper: adversarial AUC
# -----------------------------
def adversarial_auc(X, is_test, random_state=42):
    y_adv = is_test.astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    adv = LogisticRegression(max_iter=3000, solver="liblinear")

    oof = np.zeros(len(y_adv), dtype=float)
    for tr_idx, va_idx in cv.split(X, y_adv):
        est = clone(adv)
        est.fit(X[tr_idx], y_adv[tr_idx])
        oof[va_idx] = est.predict_proba(X[va_idx])[:, 1]

    return roc_auc_score(y_adv, oof)


# -----------------------------
# 4) Load data
# -----------------------------
df = pd.read_csv(FEATURES_CSV)
df["y"] = df["water_label"].map(lambda v: 1 if v in (3, 4) else 0)

exclude = {"MOF_name", "data_set", "water_label", "acid_label", "base_label", "boiling_label", "y"}
feature_cols = [c for c in df.columns if c not in exclude]

X = df[feature_cols].apply(pd.to_numeric, errors="coerce").values
y = df["y"].astype(int).values
data_set = df["data_set"].astype(str).values

if np.isnan(X).any():
    raise ValueError("Found NaNs in features. Please impute before splitting.")

print(f"Loaded: n={len(df)}, n_features={X.shape[1]}")
print(f"Class balance (stable=1): {y.mean():.4f}")

# Standardize for kNN distances + adversarial model
scaler = StandardScaler()
Xz = scaler.fit_transform(X)


# -----------------------------
# 5) Difficulty (kNN purity)
# -----------------------------
nn = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, metric="euclidean")
nn.fit(Xz)
_, idx = nn.kneighbors(Xz, return_distance=True)

neighbor_idx = idx[:, 1:]
neighbor_labels = y[neighbor_idx]
same_frac = (neighbor_labels == y.reshape(-1, 1)).mean(axis=1)
difficulty = 1.0 - same_frac

difficulty_bin = pd.qcut(difficulty, q=N_BINS, labels=False, duplicates="drop").astype(int)

# Strata key: (class, dataset, difficulty_bin)
strata = (
    y.astype(str) + "_" +
    pd.Series(data_set).astype(str) + "_" +
    difficulty_bin.astype(str)
).values


# -----------------------------
# 6) Try many candidate splits, choose best
# -----------------------------
best = None
records = []

for i in range(N_CANDIDATES):
    rs = BASE_RANDOM_STATE + i

    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=rs)
    train_idx, test_idx = next(sss.split(np.zeros(len(df)), strata))

    split = np.array(["train"] * len(df), dtype=object)
    split[test_idx] = "test"

    train_mask = split == "train"
    test_mask = split == "test"

    # Balance checks
    train_rate = y[train_mask].mean()
    test_rate = y[test_mask].mean()
    class_rate_diff = abs(train_rate - test_rate)

    diff_mean_train = difficulty[train_mask].mean()
    diff_mean_test = difficulty[test_mask].mean()
    diff_mean_diff = abs(diff_mean_train - diff_mean_test)

    # Adversarial AUC
    auc_adv = adversarial_auc(Xz, is_test=test_mask.astype(int), random_state=rs)

    records.append((rs, auc_adv, class_rate_diff, diff_mean_diff))

    # Accept only splits within constraints
    if (class_rate_diff <= MAX_CLASS_RATE_DIFF) and (diff_mean_diff <= MAX_DIFF_MEAN_DIFF):
        # Score = closeness to 0.5
        score = abs(auc_adv - ADV_TARGET)
        if (best is None) or (score < best["score"]):
            best = {
                "score": score,
                "rs": rs,
                "split": split,
                "auc_adv": auc_adv,
                "class_rate_diff": class_rate_diff,
                "diff_mean_diff": diff_mean_diff
            }

# Summary of candidates
cand_df = pd.DataFrame(records, columns=["random_state", "adv_auc", "class_rate_diff", "diff_mean_diff"])
cand_df = cand_df.sort_values("adv_auc")
print("\nTop 10 candidates by adv_auc (closest to 0.5 is good):")
print(cand_df.head(10).to_string(index=False))

if best is None:
    print("\nNo split satisfied constraints. Relax MAX_* constraints or increase N_CANDIDATES.")
    # fallback: just take the closest to 0.5
    best_row = cand_df.iloc[(cand_df["adv_auc"] - 0.5).abs().argsort()].iloc[0]
    best_rs = int(best_row["random_state"])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=best_rs)
    train_idx, test_idx = next(sss.split(np.zeros(len(df)), strata))
    split = np.array(["train"] * len(df), dtype=object)
    split[test_idx] = "test"
    best = {"rs": best_rs, "split": split, "auc_adv": float(best_row["adv_auc"]),
            "class_rate_diff": float(best_row["class_rate_diff"]),
            "diff_mean_diff": float(best_row["diff_mean_diff"]), "score": abs(float(best_row["adv_auc"]) - 0.5)}

print("\n=== BEST SPLIT SELECTED ===")
print(f"random_state: {best['rs']}")
print(f"adversarial AUC: {best['auc_adv']:.4f} (target 0.50)")
print(f"class rate diff: {best['class_rate_diff']:.5f}")
print(f"difficulty mean diff: {best['diff_mean_diff']:.5f}")

# Save best split file
out = pd.DataFrame({
    "MOF_name": df["MOF_name"].astype(str),
    "split": best["split"]
})
out.to_csv(OUT_SPLIT_CSV, index=False)
print("\nSaved BEST split file:", OUT_SPLIT_CSV)
