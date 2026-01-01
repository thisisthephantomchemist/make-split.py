# -*- coding: utf-8 -*-
"""
Multi-seed evaluation of YOUR protocol + clustering on selected RFA features.

Your protocol:
- 2-class target: labels {1,2}->0 (unstable), {3,4}->1 (stable)
- Split: stratified 80/20 on (class, data_set, difficulty_bin)
  difficulty = 1 - kNN label purity (model-free)
- Model: ExtraTrees (tree model)
- Threshold selection: choose thr on TRAIN-OOF by minimizing (FP + FN)
- Metrics reported on TEST:
  ROC-AUC, F1(pos=stable), Macro-F1, Balanced Accuracy, TN/FP/FN/TP
- Multi-seed: generate multiple splits via different random_state seeds

Clustering (option 3):
- Pick ONE representative seed (median ROC-AUC) and:
  (a) Spearman correlation clustermap on TRAIN for 70 RFA-selected features
  (b) Class mean profiles (z-scored on TRAIN) clustermap

Outputs:
- output/multiseed_results.csv
- output/multiseed_summary.csv
- output/splits/split_seed_<seed>.csv   (optional but enabled)
- output/clustering/corr_clustermap_labeled.png
- output/clustering/corr_clustermap_unlabeled.png
- output/clustering/class_means_clustermap.png
- output/clustering/feature_clusters.csv
"""

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.base import clone

# For clustering plots
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


# =============================
# 1) PATHS (EDIT THESE)
# =============================
FEATURES_CSV = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\features_and_labels.csv"
RFA_FEATS_TXT = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\output\rfa_selected_features.txt"

OUT_DIR = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\output\multiseed"
SPLITS_DIR = os.path.join(OUT_DIR, "splits")
CLUST_DIR = os.path.join(OUT_DIR, "clustering")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(CLUST_DIR, exist_ok=True)

OUT_RESULTS = os.path.join(OUT_DIR, "multiseed_results.csv")
OUT_SUMMARY = os.path.join(OUT_DIR, "multiseed_summary.csv")


# =============================
# 2) CONFIG
# =============================
TEST_SIZE = 0.20

# Difficulty config
K_NEIGHBORS = 15
N_BINS = 5

# Model config (your ExtraTrees)
ET_PARAMS = dict(
    n_estimators=800,
    max_features="sqrt",
    random_state=42,   # will be overwritten per seed for full reproducibility
    n_jobs=-1,
    class_weight=None
)

# Seeds to try (feel free to expand)
SEEDS = [11, 21, 42, 63, 77, 91, 105, 123, 202, 333]

# Save each split file?
SAVE_SPLITS = True

# CV folds for threshold selection (OOF)
N_FOLDS = 5

# Threshold grid resolution
THR_GRID = 501

# Clustering cutoff (smaller => more clusters)
CLUSTER_CUTOFF = 0.35

# Label density for clustermap
LABEL_STEP = 2  # show every 2nd label


# =============================
# 3) HELPERS
# =============================
def adversarial_auc(X, is_test, random_state=42):
    """AUC for separating train vs test; ~0.5 is good (similar distributions)."""
    y_adv = is_test.astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    adv = LogisticRegression(max_iter=3000, solver="liblinear")
    oof = np.zeros(len(y_adv), dtype=float)
    for tr, va in cv.split(X, y_adv):
        est = clone(adv)
        est.fit(X[tr], y_adv[tr])
        oof[va] = est.predict_proba(X[va])[:, 1]
    return roc_auc_score(y_adv, oof)


def oof_probs(estimator, Xtr, ytr, random_state):
    """OOF predicted probabilities for class 1."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)
    probs = np.zeros(len(ytr), dtype=float)
    for tr_idx, va_idx in cv.split(Xtr, ytr):
        est = clone(estimator)
        est.fit(Xtr.iloc[tr_idx], ytr[tr_idx])
        probs[va_idx] = est.predict_proba(Xtr.iloc[va_idx])[:, 1]
    return probs


def best_threshold_min_fp_fn(y_true, probs, grid=THR_GRID):
    """Pick threshold minimizing FP+FN on given probs."""
    thr_grid = np.linspace(0, 1, grid)
    best = None
    for t in thr_grid:
        yhat = (probs >= t).astype(int)
        TN, FP, FN, TP = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
        err = FP + FN
        if (best is None) or (err < best["FP_plus_FN"]):
            best = dict(threshold=float(t), TN=int(TN), FP=int(FP), FN=int(FN), TP=int(TP), FP_plus_FN=int(err))
    return best


def macro_f1_from_cm(TN, FP, FN, TP):
    """Compute macro-F1 from confusion counts."""
    # F1 for class 1 (stable)
    f1_pos = 0.0
    denom_pos = 2*TP + FP + FN
    if denom_pos > 0:
        f1_pos = 2*TP / denom_pos

    # F1 for class 0 (unstable) treating class 0 as "positive"
    # For class 0: TP0 = TN, FP0 = FN, FN0 = FP
    denom_neg = 2*TN + FN + FP
    f1_neg = 0.0
    if denom_neg > 0:
        f1_neg = 2*TN / denom_neg

    return 0.5 * (f1_pos + f1_neg), f1_pos, f1_neg


def balanced_accuracy_from_cm(TN, FP, FN, TP):
    """Balanced Accuracy = (TPR + TNR)/2."""
    tpr = TP / (TP + FN) if (TP + FN) else 0.0
    tnr = TN / (TN + FP) if (TN + FP) else 0.0
    return 0.5 * (tpr + tnr), tpr, tnr


# =============================
# 4) LOAD DATA
# =============================
df = pd.read_csv(FEATURES_CSV)

# Binary target
df["y"] = df["water_label"].map(lambda v: 1 if v in (3, 4) else 0)

exclude = {"MOF_name", "data_set", "water_label", "acid_label", "base_label", "boiling_label", "y"}
feature_cols_all = [c for c in df.columns if c not in exclude]

X_all = df[feature_cols_all].apply(pd.to_numeric, errors="coerce").values
y_all = df["y"].astype(int).values
data_set = df["data_set"].astype(str).values

if np.isnan(X_all).any():
    raise ValueError("Found NaNs in features. Please impute before splitting.")

print(f"Loaded: n={len(df)}, n_features(all)={X_all.shape[1]}, stable%={y_all.mean():.4f}")


# =============================
# 5) Compute difficulty bins once (model-free)
# =============================
scaler = StandardScaler()
Xz = scaler.fit_transform(X_all)

nn = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, metric="euclidean")
nn.fit(Xz)
_, idx = nn.kneighbors(Xz, return_distance=True)

neighbor_idx = idx[:, 1:]
neighbor_labels = y_all[neighbor_idx]
same_frac = (neighbor_labels == y_all.reshape(-1, 1)).mean(axis=1)
difficulty = 1.0 - same_frac

difficulty_bin = pd.qcut(difficulty, q=N_BINS, labels=False, duplicates="drop").astype(int)

# Strata
strata = (y_all.astype(str) + "_" + pd.Series(data_set).astype(str) + "_" + difficulty_bin.astype(str)).values


# =============================
# 6) Read selected RFA features (70)
# =============================
with open(RFA_FEATS_TXT, "r", encoding="utf-8") as f:
    selected_feats = [line.strip() for line in f if line.strip()]

X_sel_df = df[selected_feats].apply(pd.to_numeric, errors="coerce")
if np.isnan(X_sel_df.values).any():
    raise ValueError("Found NaNs in selected features. Please impute or investigate.")


# =============================
# 7) Multi-seed loop
# =============================
rows = []

for seed in SEEDS:
    # 7.1 create split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    train_idx, test_idx = next(sss.split(np.zeros(len(df)), strata))

    split = np.array(["train"] * len(df), dtype=object)
    split[test_idx] = "test"

    if SAVE_SPLITS:
        out_split = pd.DataFrame({"MOF_name": df["MOF_name"].astype(str), "split": split})
        out_split_path = os.path.join(SPLITS_DIR, f"split_seed_{seed}.csv")
        out_split.to_csv(out_split_path, index=False)

    # 7.2 adversarial AUC (distribution similarity)
    is_test = (split == "test").astype(int)
    adv = adversarial_auc(Xz, is_test=is_test, random_state=seed)

    # 7.3 train/test arrays (selected features)
    X_train = X_sel_df.iloc[train_idx]
    y_train = y_all[train_idx]
    X_test = X_sel_df.iloc[test_idx]
    y_test = y_all[test_idx]

    # model pipeline
    et_params = dict(ET_PARAMS)
    et_params["random_state"] = seed
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", ExtraTreesClassifier(**et_params))
    ])

    # 7.4 OOF probs + threshold on TRAIN only
    p_oof = oof_probs(model, X_train, y_train, random_state=seed)
    thr_info = best_threshold_min_fp_fn(y_train, p_oof, grid=THR_GRID)
    thr = thr_info["threshold"]

    # 7.5 Fit on full train -> test evaluation
    model.fit(X_train, y_train)
    p_test = model.predict_proba(X_test)[:, 1]

    test_auc = roc_auc_score(y_test, p_test)

    yhat = (p_test >= thr).astype(int)
    TN, FP, FN, TP = confusion_matrix(y_test, yhat, labels=[0, 1]).ravel()

    macro_f1, f1_stable, f1_unstable = macro_f1_from_cm(TN, FP, FN, TP)
    bal_acc, tpr, tnr = balanced_accuracy_from_cm(TN, FP, FN, TP)

    rows.append({
        "seed": seed,
        "adversarial_auc(train_vs_test)": float(adv),
        "thr_min_fp_fn(train_oof)": float(thr),
        "test_roc_auc": float(test_auc),
        "test_macro_f1": float(macro_f1),
        "test_f1_stable(1)": float(f1_stable),
        "test_f1_unstable(0)": float(f1_unstable),
        "test_balanced_accuracy": float(bal_acc),
        "test_TPR(stable_recall)": float(tpr),
        "test_TNR(unstable_specificity)": float(tnr),
        "TN": int(TN), "FP": int(FP), "FN": int(FN), "TP": int(TP),
        "FP_plus_FN": int(FP + FN),
        "stable_rate_train": float(y_train.mean()),
        "stable_rate_test": float(y_test.mean()),
        "difficulty_mean_train": float(difficulty[train_idx].mean()),
        "difficulty_mean_test": float(difficulty[test_idx].mean())
    })

results = pd.DataFrame(rows).sort_values("test_roc_auc", ascending=False)
results.to_csv(OUT_RESULTS, index=False)
print("\nSaved:", OUT_RESULTS)

# summary
def mean_std(series):
    return f"{series.mean():.4f} ± {series.std(ddof=1):.4f}"

summary = pd.DataFrame([{
    "n_seeds": len(SEEDS),
    "roc_auc(mean±std)": mean_std(results["test_roc_auc"]),
    "macro_f1(mean±std)": mean_std(results["test_macro_f1"]),
    "balanced_acc(mean±std)": mean_std(results["test_balanced_accuracy"]),
    "FP_plus_FN(mean±std)": mean_std(results["FP_plus_FN"]),
    "adversarial_auc(mean±std)": mean_std(results["adversarial_auc(train_vs_test)"]),
    "thr(mean±std)": mean_std(results["thr_min_fp_fn(train_oof)"])
}])
summary.to_csv(OUT_SUMMARY, index=False)
print("Saved:", OUT_SUMMARY)

print("\n=== QUICK SUMMARY ===")
print(summary.to_string(index=False))
print("\nTop 5 seeds by test ROC-AUC:")
print(results.head(5)[["seed", "test_roc_auc", "test_macro_f1", "test_balanced_accuracy", "FP_plus_FN", "adversarial_auc(train_vs_test)", "thr_min_fp_fn(train_oof)"]].to_string(index=False))


# =============================
# 8) OPTION (3): CLUSTERING on a representative seed
# Pick the median ROC-AUC seed (stable representative, not best/worst)
# =============================
rep_seed = results.sort_values("test_roc_auc").iloc[len(results)//2]["seed"]
print("\nRepresentative seed for clustering (median AUC):", rep_seed)

rep_split_path = os.path.join(SPLITS_DIR, f"split_seed_{int(rep_seed)}.csv")
rep_split = pd.read_csv(rep_split_path)
rep = df.merge(rep_split, on="MOF_name", how="inner")
train_mask_rep = rep["split"].astype(str).str.lower().str.startswith("train").values

X_train_rep = rep.loc[train_mask_rep, selected_feats].apply(pd.to_numeric, errors="coerce")
y_train_rep = rep.loc[train_mask_rep, "y"].astype(int).values

# Spearman correlation on TRAIN only
corr = X_train_rep.corr(method="spearman")

# distance = 1 - |corr|
dist = 1.0 - np.abs(corr.values)
np.fill_diagonal(dist, 0.0)
condensed = squareform(dist, checks=False)
Z = linkage(condensed, method="average")

# cluster ids
cluster_id = fcluster(Z, t=CLUSTER_CUTOFF, criterion="distance")
cluster_df = pd.DataFrame({"feature": selected_feats, "cluster": cluster_id})
cluster_df.to_csv(os.path.join(CLUST_DIR, "feature_clusters.csv"), index=False)
print("Saved:", os.path.join(CLUST_DIR, "feature_clusters.csv"))

# Clustermap unlabeled
sns.set_context("notebook")
g0 = sns.clustermap(
    corr,
    cmap="vlag",
    center=0.0,
    figsize=(12, 10),
    xticklabels=False,
    yticklabels=False
)
g0.fig.suptitle(f"Spearman Corr Clustermap (TRAIN) - unlabeled | seed={int(rep_seed)}", y=1.02)
out0 = os.path.join(CLUST_DIR, "corr_clustermap_unlabeled.png")
g0.savefig(out0, dpi=200)
plt.close(g0.fig)
print("Saved:", out0)

# Clustermap labeled (sparse ticks)
g1 = sns.clustermap(
    corr,
    cmap="vlag",
    center=0.0,
    figsize=(18, 16),
    xticklabels=True,
    yticklabels=True
)
xt = g1.ax_heatmap.get_xticklabels()
yt = g1.ax_heatmap.get_yticklabels()
for i, lab in enumerate(xt):
    lab.set_visible(i % LABEL_STEP == 0)
for i, lab in enumerate(yt):
    lab.set_visible(i % LABEL_STEP == 0)

plt.setp(g1.ax_heatmap.get_xticklabels(), rotation=90, fontsize=7)
plt.setp(g1.ax_heatmap.get_yticklabels(), rotation=0, fontsize=7)
g1.fig.suptitle(f"Spearman Corr Clustermap (TRAIN) - labeled | seed={int(rep_seed)}", y=1.02)
out1 = os.path.join(CLUST_DIR, "corr_clustermap_labeled.png")
g1.savefig(out1, dpi=200)
plt.close(g1.fig)
print("Saved:", out1)

# Class mean profiles (z-scored on TRAIN)
sc = StandardScaler()
Xz_rep = sc.fit_transform(X_train_rep.values)
Xz_rep_df = pd.DataFrame(Xz_rep, columns=selected_feats)

mean_unstable = Xz_rep_df[y_train_rep == 0].mean(axis=0)
mean_stable = Xz_rep_df[y_train_rep == 1].mean(axis=0)
means = pd.DataFrame([mean_unstable.values, mean_stable.values], index=["Unstable(0)", "Stable(1)"], columns=selected_feats)

g2 = sns.clustermap(
    means,
    cmap="vlag",
    center=0.0,
    figsize=(18, 4),
    xticklabels=True,
    yticklabels=True,
    col_linkage=Z  # use same feature clustering as corr
)
plt.setp(g2.ax_heatmap.get_xticklabels(), rotation=90, fontsize=7)
g2.fig.suptitle(f"Class mean profiles (z-scored on TRAIN) | seed={int(rep_seed)}", y=1.05)
out2 = os.path.join(CLUST_DIR, "class_means_clustermap.png")
g2.savefig(out2, dpi=200)
plt.close(g2.fig)
print("Saved:", out2)

print("\nDONE ✅")
print("Next: we will interpret which clusters dominate performance + which cluster aligns with FP errors.")
