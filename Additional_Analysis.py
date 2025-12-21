# ==============================
# ADDITIONAL ANALYSES
# (1) RSCE -> real-world degradation correlation (Spearman/Kendall + bootstrap CI + permutation p)
# (2) Failure-mode alignment summary across worlds
# (3) Optional: Temporal-split evaluation scaffold (if you have per-patient predictions + time)
# ==============================

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt

# ------------------------------
# 0) LOAD FILES
# ------------------------------
RSCE_PATH  = "Full_Dataset_mortality_RSCE_full_scores.csv"
WORLD_PATH = "Full_Dataset_mortality_world_metrics.csv"

assert os.path.exists(RSCE_PATH),  f"Missing: {RSCE_PATH}"
assert os.path.exists(WORLD_PATH), f"Missing: {WORLD_PATH}"

rsce = pd.read_csv(RSCE_PATH)
world = pd.read_csv(WORLD_PATH)

print("RSCE columns:", list(rsce.columns))
print("WORLD columns:", list(world.columns))
display(rsce.head())
display(world.head())

# ------------------------------
# 1) LIGHT SANITY & STANDARDIZE
# ------------------------------
def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# expected: model column
MODEL_COL_RSCE  = _find_col(rsce,  ["model", "Model", "MODEL"])
MODEL_COL_WORLD = _find_col(world, ["model", "Model", "MODEL"])

if MODEL_COL_RSCE is None or MODEL_COL_WORLD is None:
    raise ValueError("Cannot find model column in one of the CSVs.")

# expected: RSCE column (or RSCE_full)
RSCE_COL = _find_col(rsce, ["RSCE_full", "RSCE", "rsce_full", "rsce"])
if RSCE_COL is None:
    raise ValueError("Cannot find RSCE column in RSCE CSV (expected RSCE_full/RSCE).")

# expected: world column
WORLD_COL = _find_col(world, ["world", "World", "WORLD"])
if WORLD_COL is None:
    raise ValueError("Cannot find world column in WORLD metrics CSV (expected world).")

# expected metric columns (try to infer)
AUROC_COL  = _find_col(world, ["auroc", "AUROC", "roc_auc", "ROC_AUC"])
LOGLOSS_COL = _find_col(world, ["logloss", "LogLoss", "log_loss", "LOGLOSS"])
BRIER_COL  = _find_col(world, ["brier", "Brier", "brier_score", "BRIER"])
ECE_COL    = _find_col(world, ["ece", "ECE", "expected_calibration_error", "calibration_ece"])

print("Detected columns:",
      {"MODEL_RSCE": MODEL_COL_RSCE, "MODEL_WORLD": MODEL_COL_WORLD, "RSCE": RSCE_COL, "WORLD": WORLD_COL,
       "AUROC": AUROC_COL, "LOGLOSS": LOGLOSS_COL, "BRIER": BRIER_COL, "ECE": ECE_COL})

# Basic checks
rsce = rsce.rename(columns={MODEL_COL_RSCE: "model", RSCE_COL: "RSCE"})
world = world.rename(columns={MODEL_COL_WORLD: "model", WORLD_COL: "world"})

# keep only numeric metric cols that exist
metric_cols = [c for c in [AUROC_COL, LOGLOSS_COL, BRIER_COL, ECE_COL] if c is not None]
world_metrics = world[["model", "world"] + metric_cols].copy()

for c in metric_cols:
    world_metrics[c] = pd.to_numeric(world_metrics[c], errors="coerce")

rsce["RSCE"] = pd.to_numeric(rsce["RSCE"], errors="coerce")
rsce = rsce.dropna(subset=["RSCE"])

# ------------------------------
# 2) DEFINE BASELINE WORLD (WA_real) AUTOMATICALLY
# ------------------------------
# We try to choose a "clean-ish baseline" world.
# Priority: WA_real -> WA -> world containing "WA"
def choose_baseline_world(world_series):
    ws = world_series.astype(str).unique().tolist()
    for pref in ["WA_real", "WA", "WorldA", "worldA"]:
        if pref in ws:
            return pref
    # fallback: pick first world that contains 'WA'
    for w in ws:
        if "WA" in w:
            return w
    # otherwise, just the most frequent world
    return world_series.astype(str).value_counts().index[0]

BASE_WORLD = choose_baseline_world(world_metrics["world"])
print("Baseline world:", BASE_WORLD)

# ------------------------------
# 3) BUILD "DEGRADATION" FEATURES PER MODEL
# ------------------------------
# For each model, compute:
# - baseline metrics at BASE_WORLD
# - mean drop of AUROC over other worlds (AUROC_baseline - AUROC_world)
# - mean drift in LogLoss/Brier/ECE over other worlds (world - baseline)
# - worst-case (max) drift as a stress indicator
#
# NOTE: if a metric is "lower is better" (LogLoss/Brier/ECE), drift = world - baseline (positive = worse).
# If a metric is "higher is better" (AUROC), drop = baseline - world (positive = worse).

def compute_degradation_table(df, base_world, auroc=None, logloss=None, brier=None, ece=None):
    out_rows = []
    models = sorted(df["model"].unique())
    worlds = sorted(df["world"].astype(str).unique())

    non_base_worlds = [w for w in worlds if w != base_world]
    if len(non_base_worlds) == 0:
        raise ValueError("Only one world present; cannot compute degradation.")

    for m in models:
        sub = df[df["model"] == m].set_index("world")

        if base_world not in sub.index:
            continue

        row = {"model": m}

        # baseline values
        if auroc is not None:
            row["AUROC_base"] = sub.loc[base_world, auroc]
        if logloss is not None:
            row["LogLoss_base"] = sub.loc[base_world, logloss]
        if brier is not None:
            row["Brier_base"] = sub.loc[base_world, brier]
        if ece is not None:
            row["ECE_base"] = sub.loc[base_world, ece]

        # degradation aggregates
        def _agg(series):
            vals = series.dropna().values
            if len(vals) == 0:
                return np.nan, np.nan
            return float(np.mean(vals)), float(np.max(vals))

        if auroc is not None:
            drops = []
            for w in non_base_worlds:
                if w in sub.index:
                    drops.append(sub.loc[base_world, auroc] - sub.loc[w, auroc])
            drops = pd.Series(drops, dtype=float)
            row["AUROC_drop_mean"], row["AUROC_drop_worst"] = _agg(drops)

        if logloss is not None:
            drifts = []
            for w in non_base_worlds:
                if w in sub.index:
                    drifts.append(sub.loc[w, logloss] - sub.loc[base_world, logloss])
            drifts = pd.Series(drifts, dtype=float)
            row["LogLoss_drift_mean"], row["LogLoss_drift_worst"] = _agg(drifts)

        if brier is not None:
            drifts = []
            for w in non_base_worlds:
                if w in sub.index:
                    drifts.append(sub.loc[w, brier] - sub.loc[base_world, brier])
            drifts = pd.Series(drifts, dtype=float)
            row["Brier_drift_mean"], row["Brier_drift_worst"] = _agg(drifts)

        if ece is not None:
            drifts = []
            for w in non_base_worlds:
                if w in sub.index:
                    drifts.append(sub.loc[w, ece] - sub.loc[base_world, ece])
            drifts = pd.Series(drifts, dtype=float)
            row["ECE_drift_mean"], row["ECE_drift_worst"] = _agg(drifts)

        out_rows.append(row)

    return pd.DataFrame(out_rows)

deg = compute_degradation_table(
    world_metrics,
    base_world=BASE_WORLD,
    auroc=AUROC_COL,
    logloss=LOGLOSS_COL,
    brier=BRIER_COL,
    ece=ECE_COL
)

display(deg.sort_values(by=[c for c in ["AUROC_drop_mean","LogLoss_drift_mean","ECE_drift_mean"] if c in deg.columns][0]))

# merge with RSCE score per model
merged = deg.merge(rsce[["model","RSCE"]], on="model", how="inner")
display(merged.sort_values("RSCE", ascending=False))

# ------------------------------
# 4) CORRELATION + BOOTSTRAP CI + PERMUTATION TEST
# ------------------------------
def bootstrap_corr(x, y, corr_fn, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    n = len(x)
    if n < 4:
        return np.nan, (np.nan, np.nan), np.nan

    # point estimate
    r0 = corr_fn(x, y)

    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(corr_fn(x[idx], y[idx]))
    boots = np.asarray(boots, float)
    lo, hi = np.quantile(boots[np.isfinite(boots)], [0.025, 0.975])

    return float(r0), (float(lo), float(hi)), boots

def perm_test_corr(x, y, corr_fn, n_perm=20000, seed=123):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    n = len(x)
    if n < 4:
        return np.nan
    r0 = corr_fn(x, y)
    cnt = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        rp = corr_fn(x, yp)
        if np.isfinite(rp) and abs(rp) >= abs(r0):
            cnt += 1
    return (cnt + 1) / (n_perm + 1)

def spearman_only(x, y):
    r, _ = spearmanr(x, y)
    return r

def kendall_only(x, y):
    r, _ = kendalltau(x, y)
    return r

# Define "outcomes" we want RSCE to predict (lower is better: drops/drifts)
targets = []
for c in ["AUROC_drop_mean", "AUROC_drop_worst",
          "LogLoss_drift_mean", "LogLoss_drift_worst",
          "Brier_drift_mean", "Brier_drift_worst",
          "ECE_drift_mean", "ECE_drift_worst"]:
    if c in merged.columns:
        targets.append(c)

results = []
for t in targets:
    x = merged["RSCE"].values
    y = merged[t].values

    sp, sp_ci, _ = bootstrap_corr(x, y, spearman_only, n_boot=5000)
    sp_p = perm_test_corr(x, y, spearman_only, n_perm=20000)

    kd, kd_ci, _ = bootstrap_corr(x, y, kendall_only, n_boot=5000)
    kd_p = perm_test_corr(x, y, kendall_only, n_perm=20000)

    results.append({
        "target": t,
        "spearman_r": sp, "spearman_CI95": sp_ci, "spearman_perm_p": sp_p,
        "kendall_tau": kd, "kendall_CI95": kd_ci, "kendall_perm_p": kd_p,
        "n_models": int(np.sum(np.isfinite(x) & np.isfinite(y)))
    })

corr_df = pd.DataFrame(results).sort_values(by="spearman_r")
display(corr_df)

# Save correlation summary for paper appendix
corr_df.to_csv("RSCE_vs_realworld_degradation_correlation.csv", index=False)
print("Saved: RSCE_vs_realworld_degradation_correlation.csv")

# ------------------------------
# 5) PUBLICATION-STYLE SCATTER PLOTS (RSCE vs degradation)
# ------------------------------
def scatter_with_labels(df, xcol, ycol, title, fname):
    d = df[[xcol, ycol, "model"]].dropna()
    plt.figure(figsize=(7,5))
    plt.scatter(d[xcol], d[ycol])
    for _, r in d.iterrows():
        plt.annotate(r["model"], (r[xcol], r[ycol]), fontsize=9, xytext=(4,4), textcoords="offset points")
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.show()

for t in targets:
    scatter_with_labels(
        merged,
        xcol="RSCE",
        ycol=t,
        title=f"RSCE vs {t} (baseline={BASE_WORLD})",
        fname=f"scatter_RSCE_vs_{t}.png"
    )

print("Saved scatter plots: scatter_RSCE_vs_*.png")

# ------------------------------
# 6) FAILURE-MODE ALIGNMENT TABLE (World-level summary)
# ------------------------------
# This table helps you write the paper narrative:
# - Which worlds cause the largest AUROC drop / calibration drift for each model family
# - Which world is the "worst-case" for each model

def worst_worlds_by_metric(df, base_world, auroc=None, logloss=None, brier=None, ece=None):
    rows = []
    for m in sorted(df["model"].unique()):
        sub = df[df["model"] == m].set_index("world")
        if base_world not in sub.index:
            continue
        others = [w for w in sub.index.astype(str).tolist() if w != base_world]
        if len(others) == 0:
            continue

        row = {"model": m}

        if auroc is not None:
            drops = {w: float(sub.loc[base_world, auroc] - sub.loc[w, auroc]) for w in others if pd.notna(sub.loc[w, auroc])}
            if len(drops) > 0:
                ww = max(drops, key=drops.get)
                row["worst_world_AUROC_drop"] = ww
                row["AUROC_drop_worst"] = drops[ww]

        if ece is not None:
            drifts = {w: float(sub.loc[w, ece] - sub.loc[base_world, ece]) for w in others if pd.notna(sub.loc[w, ece])}
            if len(drifts) > 0:
                ww = max(drifts, key=drifts.get)
                row["worst_world_ECE_drift"] = ww
                row["ECE_drift_worst"] = drifts[ww]

        if logloss is not None:
            drifts = {w: float(sub.loc[w, logloss] - sub.loc[base_world, logloss]) for w in others if pd.notna(sub.loc[w, logloss])}
            if len(drifts) > 0:
                ww = max(drifts, key=drifts.get)
                row["worst_world_LogLoss_drift"] = ww
                row["LogLoss_drift_worst"] = drifts[ww]

        if brier is not None:
            drifts = {w: float(sub.loc[w, brier] - sub.loc[base_world, brier]) for w in others if pd.notna(sub.loc[w, brier])}
            if len(drifts) > 0:
                ww = max(drifts, key=drifts.get)
                row["worst_world_Brier_drift"] = ww
                row["Brier_drift_worst"] = drifts[ww]

        rows.append(row)

    return pd.DataFrame(rows)

worst_tbl = worst_worlds_by_metric(world_metrics, BASE_WORLD, AUROC_COL, LOGLOSS_COL, BRIER_COL, ECE_COL)
worst_tbl = worst_tbl.merge(rsce[["model","RSCE"]], on="model", how="left").sort_values("RSCE", ascending=False)
display(worst_tbl)

worst_tbl.to_csv("worst_worlds_by_metric.csv", index=False)
print("Saved: worst_worlds_by_metric.csv")

print("Done")
