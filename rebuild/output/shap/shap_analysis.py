# -*- coding: utf-8 -*-
"""
SHAP interpretation for ExtraTrees (final model) on MOF water stability (2-class)

Outputs (in OUT_DIR):
- shap_summary_beeswarm.png
- shap_summary_bar.png
- shap_top_features.csv
- (optional) shap_values_test.csv
- shap_dependence_<top_feature>.png (for top N features)

IMPORTANT:
- Do NOT name this script "shap.py" (will shadow the shap library).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

import shap


# =============================
# 1) PATHS (EDIT THESE)
# =============================
FEATURES_CSV  = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\features_and_labels.csv"
SPLIT_CSV     = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\output\split2080_best.csv"
RFA_FEATS_TXT = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\output\rfa_selected_features.txt"

# Use your current folder:
OUT_DIR       = r"C:\Users\NR\Documents\mof_water_stability_ml\rebuild\output\shap"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_BEESWARM  = os.path.join(OUT_DIR, "shap_summary_beeswarm.png")
OUT_BAR       = os.path.join(OUT_DIR, "shap_summary_bar.png")
OUT_TOPCSV    = os.path.join(OUT_DIR, "shap_top_features.csv")
OUT_SHAPCSV   = os.path.join(OUT_DIR, "shap_values_test.csv")   # optional


# =============================
# 2) CONFIG
# =============================
RANDOM_STATE = 42

# Use at most this many TEST samples for SHAP (None = all test)
N_TEST_SAMPLES = None  # e.g., 150 for faster; None uses all 219

# Background samples from TRAIN for TreeExplainer (speed/stability)
N_BACKGROUND = 200

# Save full SHAP values as CSV? (can be large)
SAVE_SHAP_CSV = False

# Number of dependence plots for top features
N_DEP_PLOTS = 5


# =============================
# 3) HELPERS
# =============================
def detect_split_column(split_df: pd.DataFrame) -> str:
    """Try to find the split column name in split CSV."""
    for c in split_df.columns:
        if c.lower() in ("split", "set", "subset"):
            return c
    candidates = [c for c in split_df.columns if ("split" in c.lower()) or ("set" in c.lower())]
    if not candidates:
        raise ValueError("Could not detect split column in SPLIT_CSV.")
    return candidates[0]


def pick_class1_shap(shap_values):
    """
    Return SHAP values for class 1 (Stable) in a robust way.
    Handles common SHAP outputs:
    - list of 2 arrays: [ (n,p), (n,p) ]
    - array (n,p)
    - array (n,p,2)
    """
    if isinstance(shap_values, list):
        if len(shap_values) >= 2:
            return np.array(shap_values[1])
        return np.array(shap_values[0])

    arr = np.array(shap_values)
    if arr.ndim == 3 and arr.shape[-1] >= 2:
        return arr[:, :, 1]
    return arr


def pick_class1_expected_value(expected_value):
    """Pick expected value for class 1 if available."""
    if isinstance(expected_value, list) and len(expected_value) >= 2:
        return expected_value[1]
    ev = np.array(expected_value)
    if ev.ndim == 1 and ev.size >= 2:
        return ev[1]
    return expected_value


# =============================
# 4) LOAD DATA
# =============================
df = pd.read_csv(FEATURES_CSV)
spl = pd.read_csv(SPLIT_CSV)

# Binary target: labels 1&2 -> 0 (unstable), labels 3&4 -> 1 (stable)
df["y"] = df["water_label"].map(lambda v: 1 if v in (3, 4) else 0)

split_col = detect_split_column(spl)

m = df.merge(spl[["MOF_name", split_col]], on="MOF_name", how="inner").rename(columns={split_col: "split"})
train_mask = m["split"].astype(str).str.lower().str.startswith("train").values
test_mask  = m["split"].astype(str).str.lower().str.startswith("test").values

y = m["y"].astype(int).values

# Selected features
with open(RFA_FEATS_TXT, "r", encoding="utf-8") as f:
    feats = [line.strip() for line in f if line.strip()]

X = m[feats].apply(pd.to_numeric, errors="coerce")
X_train, y_train = X.loc[train_mask], y[train_mask]
X_test,  y_test  = X.loc[test_mask],  y[test_mask]

# Optional: subsample test for SHAP
if N_TEST_SAMPLES is not None and N_TEST_SAMPLES < len(X_test):
    X_test = X_test.sample(n=N_TEST_SAMPLES, random_state=RANDOM_STATE)
    y_test = y_test[X_test.index]

print(f"Train: {X_train.shape} | Test for SHAP: {X_test.shape} | n_features: {len(feats)}")


# =============================
# 5) FIT FINAL MODEL (ExtraTrees)
# =============================
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", ExtraTreesClassifier(
        n_estimators=800,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",
        class_weight=None
    ))
])

model.fit(X_train, y_train)
p_test = model.predict_proba(X_test)[:, 1]
auc_test = roc_auc_score(y_test, p_test)
print("TEST ROC-AUC (on SHAP subset):", round(auc_test, 4))


# =============================
# 6) PREP DATA FOR SHAP (after imputation)
# =============================
imputer = model.named_steps["imputer"]
clf = model.named_steps["clf"]

Xtr_imp = pd.DataFrame(imputer.transform(X_train), columns=feats, index=X_train.index)
Xte_imp = pd.DataFrame(imputer.transform(X_test),  columns=feats, index=X_test.index)

# Background subset from train
bg = Xtr_imp.sample(n=min(N_BACKGROUND, len(Xtr_imp)), random_state=RANDOM_STATE)


# =============================
# 7) BUILD EXPLAINER + COMPUTE SHAP
# =============================
explainer = shap.TreeExplainer(clf, data=bg)

shap_values = explainer.shap_values(Xte_imp)
shap_stable = pick_class1_shap(shap_values)     # expected (n_test, n_features)
base_value = pick_class1_expected_value(explainer.expected_value)

shap_stable = np.array(shap_stable)
if shap_stable.ndim != 2:
    raise ValueError(f"Unexpected SHAP shape after selection: {shap_stable.shape}. Expected (n_samples, n_features).")

print("SHAP computed. Shape:", shap_stable.shape)
print("Base value (class 1):", base_value)


# =============================
# 8) GLOBAL IMPORTANCE TABLE (mean |SHAP|)
# =============================
mean_abs = np.abs(shap_stable).mean(axis=0)
top_df = pd.DataFrame({
    "feature": feats,
    "mean_abs_shap": mean_abs
}).sort_values("mean_abs_shap", ascending=False)

top_df.to_csv(OUT_TOPCSV, index=False)
print("Saved:", OUT_TOPCSV)

# Optional: save full SHAP values
if SAVE_SHAP_CSV:
    sv = pd.DataFrame(shap_stable, columns=feats, index=Xte_imp.index)
    sv.insert(0, "MOF_name", m.loc[Xte_imp.index, "MOF_name"].values)
    sv.insert(1, "y_true", y_test)
    sv.to_csv(OUT_SHAPCSV, index=False)
    print("Saved:", OUT_SHAPCSV)


# =============================
# 9) SHAP SUMMARY PLOTS
# =============================
plt.figure()
shap.summary_plot(shap_stable, Xte_imp, show=False, max_display=25)
plt.tight_layout()
plt.savefig(OUT_BEESWARM, dpi=200)
plt.close()
print("Saved:", OUT_BEESWARM)

plt.figure()
shap.summary_plot(shap_stable, Xte_imp, plot_type="bar", show=False, max_display=25)
plt.tight_layout()
plt.savefig(OUT_BAR, dpi=200)
plt.close()
print("Saved:", OUT_BAR)


# =============================
# 10) DEPENDENCE PLOTS FOR TOP FEATURES
# =============================
top_feats = top_df["feature"].head(N_DEP_PLOTS).tolist()
for f in top_feats:
    out_png = os.path.join(OUT_DIR, f"shap_dependence_{f}.png")
    plt.figure()
    shap.dependence_plot(f, shap_stable, Xte_imp, show=False, interaction_index=None)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Saved:", out_png)

print("\nDone.")
print("Interpretation: SHAP > 0 pushes toward Stable(1); SHAP < 0 pushes toward Unstable(0).")
