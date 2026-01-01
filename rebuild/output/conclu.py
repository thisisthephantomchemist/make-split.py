# -*- coding: utf-8 -*-
"""
Interpretation pack:
1) Cluster selected RFA features via Spearman correlation (TRAIN only)
2) Save readable clustermap (labeled + unlabeled)
3) Compute permutation importance (TEST) for feature-level importance
4) Aggregate to cluster-level importance

Outputs:
- feature_groups.csv
- cluster_summary.csv
- permutation_importance_features.csv
- permutation_importance_clusters.csv
- clustermap_corr_labeled.png
- clustermap_corr_unlabeled.png
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.inspection import permutation_importance


# -----------------------------
# 1) Paths (EDIT THESE)
# -----------------------------
FEATURES_CSV  = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\features_and_labels.csv"
SPLIT_CSV     = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\output\split2080_best.csv"
RFA_FEATS_TXT = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\output\rfa_selected_features.txt"
OUT_DIR       = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\output\interpretation"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_GROUPS    = os.path.join(OUT_DIR, "feature_groups.csv")
OUT_CSUM      = os.path.join(OUT_DIR, "cluster_summary.csv")
OUT_PI_FEAT   = os.path.join(OUT_DIR, "permutation_importance_features.csv")
OUT_PI_CLUS   = os.path.join(OUT_DIR, "permutation_importance_clusters.csv")
OUT_CMAP_LAB  = os.path.join(OUT_DIR, "clustermap_corr_labeled.png")
OUT_CMAP_NOL  = os.path.join(OUT_DIR, "clustermap_corr_unlabeled.png")


# -----------------------------
# 2) Load data + split + selected features
# -----------------------------
df = pd.read_csv(FEATURES_CSV)
spl = pd.read_csv(SPLIT_CSV)

# Binary target: labels 1&2 -> 0 (unstable), labels 3&4 -> 1 (stable)
df["y"] = df["water_label"].map(lambda v: 1 if v in (3,4) else 0)

# detect split column
split_col = None
for c in spl.columns:
    if c.lower() in ("split", "set", "subset"):
        split_col = c
        break
if split_col is None:
    split_col = [c for c in spl.columns if ("split" in c.lower()) or ("set" in c.lower())][0]

m = df.merge(spl[["MOF_name", split_col]], on="MOF_name", how="inner").rename(columns={split_col: "split"})
train_mask = m["split"].astype(str).str.lower().str.startswith("train").values
test_mask  = m["split"].astype(str).str.lower().str.startswith("test").values

y = m["y"].astype(int).values

with open(RFA_FEATS_TXT, "r", encoding="utf-8") as f:
    feats = [line.strip() for line in f if line.strip()]

X = m[feats].apply(pd.to_numeric, errors="coerce")
X_train, y_train = X.loc[train_mask], y[train_mask]
X_test,  y_test  = X.loc[test_mask],  y[test_mask]

print("Selected features:", len(feats))
print("Train:", X_train.shape, "Test:", X_test.shape)


# -----------------------------
# 3) Feature type tagging: RAC vs Zeo++
#    (heuristic based on naming patterns you already have)
# -----------------------------
ZEO_LIKE = {"Di", "Df", "GPOV", "VSA", "ASA", "AV", "POAV", "PONAV", "Density", "PLD", "LCD"}

def feature_type(name: str) -> str:
    # Common Zeo++ scalar features (exact matches)
    if name in ZEO_LIKE:
        return "Zeo++"
    # Many RACs have patterns like mc-..., lc-..., f-..., chi, lig, etc.
    if re.search(r"(mc-|lc-|f-|chi|lig|D_mc|D_lc|_all$)", name):
        return "RAC"
    # Fallback: treat unknowns as "Other/Zeo-like"
    return "Other"

feat_type = [feature_type(f) for f in feats]


# -----------------------------
# 4) Correlation clustering (TRAIN only) using Spearman
# -----------------------------
# Spearman correlation matrix
corr = pd.DataFrame(X_train, columns=feats).corr(method="spearman")

# distance = 1 - |corr|  (group strongly related features together)
dist = 1.0 - np.abs(corr.values)
np.fill_diagonal(dist, 0.0)

# convert to condensed distance for linkage
condensed = squareform(dist, checks=False)
Z = linkage(condensed, method="average")

# Choose a cutoff to define clusters (tweakable)
# Smaller cutoff -> more clusters; larger cutoff -> fewer clusters
CUTOFF = 0.35
cluster_id = fcluster(Z, t=CUTOFF, criterion="distance")

groups_df = pd.DataFrame({
    "feature": feats,
    "type": feat_type,
    "cluster": cluster_id
}).sort_values(["cluster", "type", "feature"])

groups_df.to_csv(OUT_GROUPS, index=False)
print("Saved:", OUT_GROUPS)

# Summarize clusters
cluster_summary = []
for cid, sub in groups_df.groupby("cluster"):
    types = sub["type"].value_counts(normalize=True).to_dict()
    # pick a few representative feature names
    reps = sub["feature"].head(6).tolist()
    cluster_summary.append({
        "cluster": int(cid),
        "n_features": int(len(sub)),
        "pct_RAC": float(types.get("RAC", 0.0)),
        "pct_Zeo++": float(types.get("Zeo++", 0.0)),
        "pct_Other": float(types.get("Other", 0.0)),
        "example_features": ", ".join(reps)
    })

cluster_summary_df = pd.DataFrame(cluster_summary).sort_values("n_features", ascending=False)
cluster_summary_df.to_csv(OUT_CSUM, index=False)
print("Saved:", OUT_CSUM)


# -----------------------------
# 5) Clustermaps: unlabeled + labeled (readable)
# -----------------------------
sns.set_context("notebook")

# Unlabeled (structure)
g0 = sns.clustermap(
    corr,
    cmap="vlag",
    center=0.0,
    figsize=(12, 10),
    xticklabels=False,
    yticklabels=False
)
g0.fig.suptitle("Spearman correlation clustermap (TRAIN) - Unlabeled", y=1.02)
g0.savefig(OUT_CMAP_NOL, dpi=200)
plt.close(g0.fig)
print("Saved:", OUT_CMAP_NOL)

# Labeled (readable): label only every k-th tick to avoid clutter
# You can increase/decrease STEP depending on readability.
STEP = 2  # show every 2nd label (try 1 if you want all labels)
g1 = sns.clustermap(
    corr,
    cmap="vlag",
    center=0.0,
    figsize=(18, 16),
    xticklabels=True,
    yticklabels=True
)
# reduce tick density
xt = g1.ax_heatmap.get_xticklabels()
yt = g1.ax_heatmap.get_yticklabels()
for i, lab in enumerate(xt):
    lab.set_visible(i % STEP == 0)
for i, lab in enumerate(yt):
    lab.set_visible(i % STEP == 0)

plt.setp(g1.ax_heatmap.get_xticklabels(), rotation=90, fontsize=7)
plt.setp(g1.ax_heatmap.get_yticklabels(), rotation=0, fontsize=7)
g1.fig.suptitle("Spearman correlation clustermap (TRAIN) - Labeled", y=1.02)
g1.savefig(OUT_CMAP_LAB, dpi=200)
plt.close(g1.fig)
print("Saved:", OUT_CMAP_LAB)


# -----------------------------
# 6) Permutation importance (TEST) for feature-level + cluster-level importance
# -----------------------------
# Final model (ExtraTrees) — same as your final choice
final_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", ExtraTreesClassifier(
        n_estimators=800,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
        class_weight=None
    ))
])

# Fit on full TRAIN, evaluate on TEST (fixed model; OK for interpretation)
final_model.fit(X_train, y_train)
p_test = final_model.predict_proba(X_test)[:, 1]
auc_test = roc_auc_score(y_test, p_test)
print("TEST ROC-AUC (for interpretation run):", round(auc_test, 4))

# Permutation importance: scoring = roc_auc (threshold-free)
pi = permutation_importance(
    final_model, X_test, y_test,
    scoring="roc_auc",
    n_repeats=20,
    random_state=42,
    n_jobs=-1
)

pi_feat = pd.DataFrame({
    "feature": feats,
    "importance_mean": pi.importances_mean,
    "importance_std": pi.importances_std
}).merge(groups_df, on="feature", how="left") \
 .sort_values("importance_mean", ascending=False)

pi_feat.to_csv(OUT_PI_FEAT, index=False)
print("Saved:", OUT_PI_FEAT)

# Cluster-level importance = sum of mean importances within cluster
pi_cluster = pi_feat.groupby("cluster")["importance_mean"].sum().reset_index()
pi_cluster = pi_cluster.merge(cluster_summary_df[["cluster", "n_features", "pct_RAC", "pct_Zeo++"]], on="cluster", how="left")
pi_cluster = pi_cluster.sort_values("importance_mean", ascending=False)
pi_cluster.to_csv(OUT_PI_CLUS, index=False)
print("Saved:", OUT_PI_CLUS)

print("\nTop 5 clusters by total importance:")
print(pi_cluster.head(5).to_string(index=False))

print("\nTop 10 features by importance:")
print(pi_feat.head(10)[["feature","type","cluster","importance_mean","importance_std"]].to_string(index=False))
