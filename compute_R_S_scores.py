"""
compute_R_S_scores.py

Central scoring engine for the synthetic-only benchmark.

This script:
- Loads per-world Model Zoo results:
    - WorldA_Ideal.csv
    - worldB_hospital_skew_results.csv
    - worldC_noise_missing_outliers_results.csv
    - worldD_distribution_shift_results.csv
    - worldE_wbv_error_results.csv
    - worldF_strong_nonlinear_results.csv
- Computes:
    - Reliability Score (R): on base worlds A & B
    - Robustness Score (S): under perturbations C, D, E, F

1) Reliability Score (R)
   - Computed only on "base worlds": A (ideal) and B (realistic hospital).
   - Uses AUROC, PR-AUC, and Brier score.
   - For each world, metrics are normalized across models:
        auc_norm   = (AUROC - min_AUROC) / (max_AUROC - min_AUROC)
        prauc_norm = (PR-AUC - min_PR-AUC) / (max_PR-AUC - min_PR-AUC)
        brier_norm = (max_Brier - Brier) / (max_Brier - min_Brier)
     (so that larger is better for all three)
   - Per-world reliability:
        R_world = (w_auc * auc_norm + w_prauc * prauc_norm + w_brier * brier_norm)
                  / (w_auc + w_prauc + w_brier)
   - Final R for each model is the mean of R_world over World A and B.

2) Robustness Score (S)
   - Measures how much performance drops under perturbations:
        - World C: noise + missing + outliers
        - World D: distribution shift
        - World E: surrogate WBV error
        - World F: strong nonlinear world
   - For each world, we define a "clean" baseline:
        - C: scenario = "noise_only" and noise_rel_std = 0.0
        - D: shift_severity = 0.0
        - E: wbv_bias = 0.0
        - F: nonlinear_alpha = 0.0
   - For each model and each perturbation level in that world, we compute
     relative drops in AUROC and Brier score:
        d_auc   = max(0, (AUROC_base - AUROC_perturbed) / AUROC_base)
        d_brier = max(0, (Brier_perturbed - Brier_base) / Brier_base)
     and average them:
        d_mean = (d_auc + d_brier) / 2
   - For each model, S is:
        S = 1 - mean(d_mean over all perturbation points in C, D, E, F)
     so higher S = more robust (smaller performance drop).

Output:
- scores_R_S.csv with columns:
    model, R, S, RS_mean
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------

DATA_DIR = Path(".")  # directory where the CSV files live
OUTPUT_CSV = DATA_DIR / "scores_R_S.csv"

WORLD_A_FILE = DATA_DIR / "WorldA_Ideal.csv"
WORLD_B_FILE = DATA_DIR / "worldB_hospital_skew_results.csv"
WORLD_C_FILE = DATA_DIR / "worldC_noise_missing_outliers_results.csv"
WORLD_D_FILE = DATA_DIR / "worldD_distribution_shift_results.csv"
WORLD_E_FILE = DATA_DIR / "worldE_wbv_error_results.csv"
WORLD_F_FILE = DATA_DIR / "worldF_strong_nonlinear_results.csv"

# weights for R
W_AUC = 0.4
W_PRAUC = 0.3
W_BRIER = 0.3


# ----------------------------------------------------
# RELIABILITY (R)
# ----------------------------------------------------

def _normalize_world_for_R(
    df: pd.DataFrame,
    w_auc: float,
    w_prauc: float,
    w_brier: float,
) -> pd.DataFrame:
    """
    Normalize AUROC, PR-AUC, and Brier across models for a single world
    and compute per-world R_world for each model.
    """
    eps = 1e-8
    df = df.copy()

    auc_min, auc_max = df["roc_auc"].min(), df["roc_auc"].max()
    pr_min, pr_max = df["pr_auc"].min(), df["pr_auc"].max()
    brier_min, brier_max = df["brier"].min(), df["brier"].max()

    df["auc_norm"] = (df["roc_auc"] - auc_min) / max(auc_max - auc_min, eps)
    df["prauc_norm"] = (df["pr_auc"] - pr_min) / max(pr_max - pr_min, eps)
    df["brier_norm"] = (brier_max - df["brier"]) / max(brier_max - brier_min, eps)

    df["R_world"] = (
        w_auc * df["auc_norm"]
        + w_prauc * df["prauc_norm"]
        + w_brier * df["brier_norm"]
    ) / (w_auc + w_prauc + w_brier)

    return df[["model", "R_world"]]


def compute_R(
    worldA: pd.DataFrame,
    worldB: pd.DataFrame,
    w_auc: float = W_AUC,
    w_prauc: float = W_PRAUC,
    w_brier: float = W_BRIER,
) -> pd.DataFrame:
    """
    Compute Reliability Score R for each model from World A and B.
    """
    A = _normalize_world_for_R(worldA, w_auc, w_prauc, w_brier)
    B = _normalize_world_for_R(worldB, w_auc, w_prauc, w_brier)

    merged = A.merge(B, on="model", suffixes=("_A", "_B"))
    merged["R"] = merged[["R_world_A", "R_world_B"]].mean(axis=1)
    return merged[["model", "R"]]


# ----------------------------------------------------
# ROBUSTNESS (S)
# ----------------------------------------------------

def compute_S(
    worldC: pd.DataFrame,
    worldD: pd.DataFrame,
    worldE: pd.DataFrame,
    worldF: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Robustness Score S for each model from Worlds C, D, E, F.
    """
    eps = 1e-8
    drops = []

    # WORLD C: noise + missing + outliers
    baseC = worldC[
        (worldC["scenario"] == "noise_only") & (worldC["noise_rel_std"] == 0.0)
    ]
    for _, row in baseC.iterrows():
        model = row["model"]
        auc_base = float(row["roc_auc"])
        brier_base = float(row["brier"])

        perturbed = worldC[
            (worldC["model"] == model)
            & ~(
                (worldC["scenario"] == "noise_only")
                & (worldC["noise_rel_std"] == 0.0)
            )
        ]

        for _, pr in perturbed.iterrows():
            auc_p = float(pr["roc_auc"])
            brier_p = float(pr["brier"])

            d_auc = max(0.0, (auc_base - auc_p) / max(auc_base, eps))
            d_brier = max(0.0, (brier_p - brier_base) / max(brier_base, eps))
            d_mean = (d_auc + d_brier) / 2.0
            drops.append((model, "C", d_mean))

    # WORLD D: distribution shift
    baseD = worldD[worldD["shift_severity"] == 0.0]
    for _, row in baseD.iterrows():
        model = row["model"]
        auc_base = float(row["roc_auc"])
        brier_base = float(row["brier"])

        perturbed = worldD[
            (worldD["model"] == model) & (worldD["shift_severity"] > 0.0)
        ]

        for _, pr in perturbed.iterrows():
            auc_p = float(pr["roc_auc"])
            brier_p = float(pr["brier"])

            d_auc = max(0.0, (auc_base - auc_p) / max(auc_base, eps))
            d_brier = max(0.0, (brier_p - brier_base) / max(brier_base, eps))
            d_mean = (d_auc + d_brier) / 2.0
            drops.append((model, "D", d_mean))

    # WORLD E: surrogate WBV error
    baseE = worldE[worldE["wbv_bias"] == 0.0]
    for _, row in baseE.iterrows():
        model = row["model"]
        auc_base = float(row["roc_auc"])
        brier_base = float(row["brier"])

        perturbed = worldE[(worldE["model"] == model) & (worldE["wbv_bias"] > 0.0)]

        for _, pr in perturbed.iterrows():
            auc_p = float(pr["roc_auc"])
            brier_p = float(pr["brier"])

            d_auc = max(0.0, (auc_base - auc_p) / max(auc_base, eps))
            d_brier = max(0.0, (brier_p - brier_base) / max(brier_base, eps))
            d_mean = (d_auc + d_brier) / 2.0
            drops.append((model, "E", d_mean))

    # WORLD F: strong nonlinear world
    baseF = worldF[worldF["nonlinear_alpha"] == 0.0]
    for _, row in baseF.iterrows():
        model = row["model"]
        auc_base = float(row["roc_auc"])
        brier_base = float(row["brier"])

        perturbed = worldF[
            (worldF["model"] == model) & (worldF["nonlinear_alpha"] > 0.0)
        ]

        for _, pr in perturbed.iterrows():
            auc_p = float(pr["roc_auc"])
            brier_p = float(pr["brier"])

            d_auc = max(0.0, (auc_base - auc_p) / max(auc_base, eps))
            d_brier = max(0.0, (brier_p - brier_base) / max(brier_base, eps))
            d_mean = (d_auc + d_brier) / 2.0
            drops.append((model, "F", d_mean))

    if not drops:
        raise RuntimeError("No perturbation drops computed; check input CSVs.")

    drops_df = pd.DataFrame(drops, columns=["model", "world", "d_mean"])
    agg = drops_df.groupby("model")["d_mean"].mean().reset_index()
    agg["S"] = 1.0 - agg["d_mean"].clip(lower=0.0)
    return agg[["model", "S"]]


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------

def main() -> None:
    worldA = pd.read_csv(WORLD_A_FILE)
    worldB = pd.read_csv(WORLD_B_FILE)
    worldC = pd.read_csv(WORLD_C_FILE)
    worldD = pd.read_csv(WORLD_D_FILE)
    worldE = pd.read_csv(WORLD_E_FILE)
    worldF = pd.read_csv(WORLD_F_FILE)

    R_df = compute_R(worldA, worldB)
    S_df = compute_S(worldC, worldD, worldE, worldF)

    scores = R_df.merge(S_df, on="model", how="inner")
    scores["RS_mean"] = (scores["R"] + scores["S"]) / 2.0
    scores = scores.sort_values("RS_mean", ascending=False)

    scores.to_csv(OUTPUT_CSV, index=False)

    print("=== R & S scores ===")
    print(scores)
    print()
    print(f"Scores saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
