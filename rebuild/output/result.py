# -*- coding: utf-8 -*-
"""
Outputs:
1) Confusion matrix on TEST (png + csv)
2) Feature-correlation clustermap for RFA-selected features (png)
3) Class-profile clustermap (Stable vs Unstable) (png)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -----------------------------
# 1) Paths (EDIT THESE)
# -----------------------------
FEATURES_CSV = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\features_and_labels.csv"
SPLIT_CSV    = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\output\split2080_best.csv"
RFA_FEATS_TXT= r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\output\rfa_selected_features.txt"

OUT_DIR      = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\output"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CM_PNG   = os.path.join(OUT_DIR, "confusion_matrix_test.png")
OUT_CM_CSV   = os.path.join(OUT_DIR, "confusion_matrix_test.csv")
OUT_CORR_PNG = os.path.join(OUT_DIR, "clustermap_feature_correlation.png")
OUT_PROF_PNG = os.path.join(OUT_DIR, "clustermap_class_profile.png")

# -----------------------------
# 2) Load data + split
# -----------------------------
df = pd.read_csv(FEATURES_CSV)
spl = pd.read_csv(SPLIT_CSV)

# Binary target: labels 1&2 -> 0 (unstable), labels 3&4 -> 1 (stable)
df["y"] = df["water_label"].map(lambda v: 1 if v in (3, 4) else 0)

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

# -----------------------------
# 3) Load RFA-selected features
# -----------------------------
with open(RFA_FEATS_TXT, "r", encoding="utf-8") as f:
    selected_features = [line.strip() for line in f if line.strip()]

X = m[selected_features].apply(pd.to_numeric, errors="coerce")

X_train, y_train = X.loc[train_mask], y[train_mask]
X_test,  y_test  = X.loc[test_mask],  y[test_mask]

print("Selected features:", len(selected_features))
print("Train:", X_train.shape, "Test:", X_test.shape)

# -----------------------------
# 4) Build final model (use YOUR best params here!)
# -----------------------------
# Replace these with your tuned params:
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

final_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(**best_params))
])

final_model.fit(X_train, y_train)

# Predict on TEST
probs_test = final_model.predict_proba(X_test)[:, 1]
# Choose threshold: use the CV-chosen threshold you reported (0.443333)
thr = 0.443333
y_pred = (probs_test >= thr).astype(int)

# -----------------------------
# 5) Confusion Matrix (TEST)
# -----------------------------
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])  # 0=unstable, 1=stable

cm_df = pd.DataFrame(cm, index=["True Unstable(0)", "True Stable(1)"],
                        columns=["Pred Unstable(0)", "Pred Stable(1)"])
cm_df.to_csv(OUT_CM_CSV, index=True)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Unstable(0)", "Stable(1)"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, values_format="d", colorbar=False)
ax.set_title(f"Confusion Matrix on TEST (thr={thr:.3f})")
plt.tight_layout()
plt.savefig(OUT_CM_PNG, dpi=200)
plt.close(fig)

print("Saved:", OUT_CM_PNG)
print("Saved:", OUT_CM_CSV)

# -----------------------------
# 6) Clustermap 1: Feature Correlation (selected features)
# -----------------------------
# Compute correlation on TRAIN only (scientifically cleaner)
corr = pd.DataFrame(X_train, columns=selected_features).corr(method="spearman")

sns.set_context("notebook")
g = sns.clustermap(
    corr,
    cmap="vlag",
    center=0.0,
    figsize=(12, 10),
    xticklabels=False,
    yticklabels=False
)
g.fig.suptitle("Clustermap: Spearman Correlation of RFA-Selected Features (TRAIN)", y=1.02)
g.savefig(OUT_CORR_PNG, dpi=200)
plt.close(g.fig)

print("Saved:", OUT_CORR_PNG)

# -----------------------------
# 7) Clustermap 2: Class Profile (Stable vs Unstable)
# -----------------------------
# Standardize features using TRAIN statistics (z-score), then average per class
Xtr = pd.DataFrame(X_train, columns=selected_features)
mu = Xtr.mean(axis=0)
sd = Xtr.std(axis=0).replace(0, 1.0)
Xtr_z = (Xtr - mu) / sd

prof_unstable = Xtr_z[y_train == 0].mean(axis=0)
prof_stable   = Xtr_z[y_train == 1].mean(axis=0)

prof = pd.DataFrame([prof_unstable, prof_stable],
                    index=["Unstable(0)", "Stable(1)"])

g2 = sns.clustermap(
    prof,
    cmap="vlag",
    center=0.0,
    figsize=(14, 3),
    col_cluster=True,
    row_cluster=False
)
g2.fig.suptitle("Clustermap: Class Mean Profiles (z-scored on TRAIN)", y=1.15)
g2.savefig(OUT_PROF_PNG, dpi=200)
plt.close(g2.fig)

print("Saved:", OUT_PROF_PNG)
