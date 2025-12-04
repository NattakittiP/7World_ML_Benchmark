# rsce_rank_stability_worlds.py
# Rank-order stability of flagship models across all Worlds (A–G)

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OUT_DIR = Path("RSCE_world_rank")

WORLD_FILES = {
    "WorldA": Path("WorldA_Ideal.csv"),
    "WorldB": Path("worldB_hospital_skew_results.csv"),
    "WorldC": Path("worldC_noise_missing_outliers_results.csv"),
    "WorldD": Path("worldD_distribution_shift_results.csv"),
    "WorldE": Path("worldE_wbv_error_results.csv"),
    "WorldF": Path("worldF_strong_nonlinear_results.csv"),
    "WorldG": Path("worldG_label_noise_results.csv"),
}

FLAGSHIP_MODELS = [
    "ExtraTrees",
    "RandomForest",
    "GradientBoosting",
    "XGBoost",
    "LightGBM",
    "MLP",
    "SVC_RBF",
    "GaussianNB",
    "Logistic_L1",
    "Logistic_L2",
]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_world_results() -> pd.DataFrame:
    rows = []
    for wname, path in WORLD_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"{wname} file not found: {path}")

        df = pd.read_csv(path)

        # ใช้เฉพาะ flagship models
        df = df[df["model"].isin(FLAGSHIP_MODELS)].copy()
        if df.empty:
            continue

        # rank จาก roc_auc (ใหญ่ดีกว่า)
        df = df.sort_values("roc_auc", ascending=False).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)

        for _, row in df.iterrows():
            rows.append(
                {
                    "world": wname,
                    "model": row["model"],
                    "roc_auc": float(row["roc_auc"]),
                    "rank": int(row["rank"]),
                }
            )

    return pd.DataFrame(rows)


def compute_spearman_matrix(rank_df: pd.DataFrame) -> pd.DataFrame:
    # pivot: index = model, columns = world, values = rank
    pivot = rank_df.pivot_table(
        index="model", columns="world", values="rank"
    )

    worlds = list(pivot.columns)
    n = len(worlds)

    rho_mat = np.zeros((n, n))
    for i, wi in enumerate(worlds):
        for j, wj in enumerate(worlds):
            r1 = pivot[wi]
            r2 = pivot[wj]
            # ดรอป model ที่ missing ใน world ใด world หนึ่ง
            mask = (~r1.isna()) & (~r2.isna())
            if mask.sum() < 2:
                rho = np.nan
            else:
                rho, _ = spearmanr(r1[mask], r2[mask])
            rho_mat[i, j] = rho

    rho_df = pd.DataFrame(rho_mat, index=worlds, columns=worlds)
    return rho_df


def plot_spearman_heatmap(rho_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(rho_df.to_numpy(), vmin=-1.0, vmax=1.0)
    plt.colorbar(label="Spearman ρ (rank correlation)")
    worlds = rho_df.index.tolist()
    plt.xticks(np.arange(len(worlds)), worlds, rotation=45, ha="right")
    plt.yticks(np.arange(len(worlds)), worlds)
    plt.title("Rank-order stability across Worlds (roc_auc)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)


def plot_worldA_vs_others(rho_df: pd.DataFrame, out_path: Path) -> None:
    if "WorldA" not in rho_df.index:
        return
    base = "WorldA"
    others = [w for w in rho_df.index if w != base]

    vals = rho_df.loc[base, others].to_numpy()
    x = np.arange(len(others))

    plt.figure(figsize=(6, 4))
    plt.bar(x, vals)
    plt.xticks(x, others, rotation=30, ha="right")
    plt.ylabel("Spearman ρ vs WorldA")
    plt.ylim(-1.0, 1.0)
    plt.title("Rank stability relative to WorldA (roc_auc)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)


def main():
    ensure_dir(OUT_DIR)

    rank_df = load_world_results()
    rank_csv = OUT_DIR / "world_rankings.csv"
    rank_df.to_csv(rank_csv, index=False)
    print("Saved:", rank_csv)

    rho_df = compute_spearman_matrix(rank_df)
    rho_csv = OUT_DIR / "world_rank_spearman.csv"
    rho_df.to_csv(rho_csv)
    print("Saved:", rho_csv)

    heatmap_png = OUT_DIR / "world_rank_spearman_heatmap.png"
    plot_spearman_heatmap(rho_df, heatmap_png)

    bar_png = OUT_DIR / "world_rank_spearman_vs_WorldA.png"
    plot_worldA_vs_others(rho_df, bar_png)


if __name__ == "__main__":
    main()
