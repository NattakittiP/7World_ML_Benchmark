"""
compute_R_S_C_scores.py

คำนวณ:
- R: Reliability (World A + B, normalize AUROC/PR-AUC/Brier)
- S: Robustness (performance drop under perturbations C,D,E,F)
- C: Calibration Stability (Brier-only drop under perturbations C,D,E,F)

Input:
- WorldA_Ideal.csv
- worldB_hospital_skew_results.csv
- worldC_noise_missing_outliers_results.csv
- worldD_distribution_shift_results.csv
- worldE_wbv_error_results.csv
- worldF_strong_nonlinear_results.csv

Output:
- scores_R_S_C.csv
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

# ------------------ CONFIG ------------------

DATA_DIR = Path(".")
OUT_CSV = DATA_DIR / "scores_R_S_C.csv"

WORLD_A_FILE = DATA_DIR / "WorldA_Ideal.csv"
WORLD_B_FILE = DATA_DIR / "worldB_hospital_skew_results.csv"
WORLD_C_FILE = DATA_DIR / "worldC_noise_missing_outliers_results.csv"
WORLD_D_FILE = DATA_DIR / "worldD_distribution_shift_results.csv"
WORLD_E_FILE = DATA_DIR / "worldE_wbv_error_results.csv"
WORLD_F_FILE = DATA_DIR / "worldF_strong_nonlinear_results.csv"

W_AUC = 0.4
W_PRAUC = 0.3
W_BRIER = 0.3


# ------------------ RELIABILITY R ------------------

def _normalize_for_R(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-8
    df = df.copy()

    auc_min, auc_max = df["roc_auc"].min(), df["roc_auc"].max()
    pr_min, pr_max = df["pr_auc"].min(), df["pr_auc"].max()
    b_min, b_max = df["brier"].min(), df["brier"].max()

    df["auc_norm"]   = (df["roc_auc"] - auc_min) / max(auc_max - auc_min, eps)
    df["prauc_norm"] = (df["pr_auc"]   - pr_min) / max(pr_max - pr_min, eps)
    df["brier_norm"] = (b_max - df["brier"])    / max(b_max - b_min,   eps)

    df["R_world"] = (
        W_AUC   * df["auc_norm"]
        + W_PRAUC * df["prauc_norm"]
        + W_BRIER * df["brier_norm"]
    ) / (W_AUC + W_PRAUC + W_BRIER)

    return df[["model", "R_world"]]


def compute_R(worldA: pd.DataFrame, worldB: pd.DataFrame) -> pd.DataFrame:
    A = _normalize_for_R(worldA)
    B = _normalize_for_R(worldB)
    merged = A.merge(B, on="model", suffixes=("_A", "_B"))
    merged["R"] = merged[["R_world_A", "R_world_B"]].mean(axis=1)
    return merged[["model", "R"]]


# ------------------ ROBUSTNESS S ------------------

def compute_S_and_C(
    worldC: pd.DataFrame,
    worldD: pd.DataFrame,
    worldE: pd.DataFrame,
    worldF: pd.DataFrame,
) -> pd.DataFrame:
    eps = 1e-8
    rows = []

    # helper: add drops
    def add_drops(model, world_tag, auc_base, brier_base, auc_p, brier_p):
        d_auc   = max(0.0, (auc_base   - auc_p)     / max(auc_base, eps))
        d_brier = max(0.0, (brier_p    - brier_base)/ max(brier_base, eps))
        return d_auc, d_brier

    drops_auc = []
    drops_brier = []

    # WORLD C
    baseC = worldC[(worldC["scenario"] == "noise_only") & (worldC["noise_rel_std"] == 0.0)]
    for _, r in baseC.iterrows():
        m = r["model"]
        auc_base = float(r["roc_auc"])
        brier_base = float(r["brier"])

        pert = worldC[
            (worldC["model"] == m)
            & ~((worldC["scenario"] == "noise_only") & (worldC["noise_rel_std"] == 0.0))
        ]
        for _, p in pert.iterrows():
            auc_p   = float(p["roc_auc"])
            brier_p = float(p["brier"])
            d_auc, d_brier = add_drops(m, "C", auc_base, brier_base, auc_p, brier_p)
            drops_auc.append((m, "C", d_auc))
            drops_brier.append((m, "C", d_brier))

    # WORLD D
    baseD = worldD[worldD["shift_severity"] == 0.0]
    for _, r in baseD.iterrows():
        m = r["model"]
        auc_base = float(r["roc_auc"])
        brier_base = float(r["brier"])
        pert = worldD[(worldD["model"] == m) & (worldD["shift_severity"] > 0.0)]
        for _, p in pert.iterrows():
            auc_p = float(p["roc_auc"])
            brier_p = float(p["brier"])
            d_auc, d_brier = add_drops(m, "D", auc_base, brier_base, auc_p, brier_p)
            drops_auc.append((m, "D", d_auc))
            drops_brier.append((m, "D", d_brier))

    # WORLD E
    baseE = worldE[worldE["wbv_bias"] == 0.0]
    for _, r in baseE.iterrows():
        m = r["model"]
        auc_base = float(r["roc_auc"])
        brier_base = float(r["brier"])
        pert = worldE[(worldE["model"] == m) & (worldE["wbv_bias"] > 0.0)]
        for _, p in pert.iterrows():
            auc_p = float(p["roc_auc"])
            brier_p = float(p["brier"])
            d_auc, d_brier = add_drops(m, "E", auc_base, brier_base, auc_p, brier_p)
            drops_auc.append((m, "E", d_auc))
            drops_brier.append((m, "E", d_brier))

    # WORLD F
    baseF = worldF[worldF["nonlinear_alpha"] == 0.0]
    for _, r in baseF.iterrows():
        m = r["model"]
        auc_base = float(r["roc_auc"])
        brier_base = float(r["brier"])
        pert = worldF[(worldF["model"] == m) & (worldF["nonlinear_alpha"] > 0.0)]
        for _, p in pert.iterrows():
            auc_p = float(p["roc_auc"])
            brier_p = float(p["brier"])
            d_auc, d_brier = add_drops(m, "F", auc_base, brier_base, auc_p, brier_p)
            drops_auc.append((m, "F", d_auc))
            drops_brier.append((m, "F", d_brier))

    if not drops_auc:
        raise RuntimeError("No drops computed; check world CSVs.")

    df_auc   = pd.DataFrame(drops_auc,   columns=["model", "world", "d_auc"])
    df_brier = pd.DataFrame(drops_brier, columns=["model", "world", "d_brier"])

    agg_auc   = df_auc.groupby("model")["d_auc"].mean().reset_index()
    agg_brier = df_brier.groupby("model")["d_brier"].mean().reset_index()

    agg = agg_auc.merge(agg_brier, on="model")
    # S = 1 - mean(drop_auc, drop_brier)
    agg["S"] = 1.0 - ((agg["d_auc"] + agg["d_brier"]) / 2.0).clip(lower=0.0)
    # C = 1 - mean(drop_brier)
    agg["C"] = 1.0 - agg["d_brier"].clip(lower=0.0)

    return agg[["model", "S", "C"]]


# ------------------ MAIN ------------------

def main():
    worldA = pd.read_csv(WORLD_A_FILE)
    worldB = pd.read_csv(WORLD_B_FILE)
    worldC = pd.read_csv(WORLD_C_FILE)
    worldD = pd.read_csv(WORLD_D_FILE)
    worldE = pd.read_csv(WORLD_E_FILE)
    worldF = pd.read_csv(WORLD_F_FILE)

    R_df = compute_R(worldA, worldB)
    SC_df = compute_S_and_C(worldC, worldD, worldE, worldF)

    scores = R_df.merge(SC_df, on="model", how="inner")
    scores["RS_mean"]  = (scores["R"] + scores["S"]) / 2.0
    scores["RSC_mean"] = (scores["R"] + scores["S"] + scores["C"]) / 3.0

    scores = scores.sort_values("RSC_mean", ascending=False)
    scores.to_csv(OUT_CSV, index=False)

    print("=== R, S, C scores ===")
    print(scores)
    print(f"\nSaved to: {OUT_CSV}")


if __name__ == "__main__":
    main()
