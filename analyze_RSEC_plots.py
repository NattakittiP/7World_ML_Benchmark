"""
analyze_RSEC_plots.py

Merge:
- scores_R_S_C.csv
- scores_E_all.csv

สร้าง:
- scores_RSEC.csv
และ plot:
- 2D scatter plots (R vs S, R vs E, S vs E, C vs E)
- Bar plots for each metric
- Heatmap of R,S,C,E
- 3D scatter R,S,E
- Radar plot (spider) for selected models
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


DATA_DIR = Path(".")
RSC_FILE = DATA_DIR / "scores_R_S_C.csv"
E_FILE = DATA_DIR / "scores_E_all.csv"
OUT_MERGED = DATA_DIR / "scores_RSEC.csv"
PLOT_DIR = DATA_DIR / "plots_RSEC"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    ensure_dir(PLOT_DIR)

    rsc = pd.read_csv(RSC_FILE)
    e = pd.read_csv(E_FILE)

    # merge on model
    df = rsc.merge(e[["model", "E"]], on="model", how="left")
    df["RSEC_mean"] = df[["R", "S", "C", "E"]].mean(axis=1, skipna=True)
    df = df.sort_values("RSEC_mean", ascending=False)
    df.to_csv(OUT_MERGED, index=False)
    print("Merged R,S,C,E:")
    print(df)

    # ------------- Scatter plots -------------
    def scatter_plot(x, y, fname, xlabel, ylabel):
        plt.figure(figsize=(6, 5))
        for _, row in df.iterrows():
            plt.scatter(row[x], row[y])
            plt.text(row[x] + 0.002, row[y] + 0.002, row["model"], fontsize=8)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / fname, dpi=300)
        plt.close()

    scatter_plot("R", "S", "scatter_R_vs_S.png", "R (Reliability)", "S (Robustness)")
    scatter_plot("R", "E", "scatter_R_vs_E.png", "R (Reliability)", "E (Explainability)")
    scatter_plot("S", "E", "scatter_S_vs_E.png", "S (Robustness)", "E (Explainability)")
    scatter_plot("C", "E", "scatter_C_vs_E.png", "C (Calibration Stability)", "E (Explainability)")

    # ------------- Bar plots -------------
    def bar_plot_metric(metric, fname):
        plt.figure(figsize=(8, 4))
        plt.bar(df["model"], df[metric])
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / fname, dpi=300)
        plt.close()

    for m in ["R", "S", "C", "E", "RSEC_mean"]:
        bar_plot_metric(m, f"bar_{m}.png")

    # ------------- Heatmap -------------
    metrics = ["R", "S", "C", "E"]
    data = df[metrics].to_numpy()
    plt.figure(figsize=(6, 6))
    im = plt.imshow(data, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.yticks(range(len(df)), df["model"])
    plt.xticks(range(len(metrics)), metrics)
    plt.title("R,S,C,E Heatmap")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "heatmap_R_S_C_E.png", dpi=300)
    plt.close()

    # ------------- 3D scatter (R,S,E) -------------
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for _, row in df.iterrows():
        ax.scatter(row["R"], row["S"], row["E"])
        ax.text(row["R"], row["S"], row["E"], row["model"], fontsize=8)
    ax.set_xlabel("R")
    ax.set_ylabel("S")
    ax.set_zlabel("E")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "scatter_3D_R_S_E.png", dpi=300)
    plt.close()

    # ------------- Radar plot -------------
    # normalize metrics per column to [0,1] for radar
    radar_df = df[["model", "R", "S", "C", "E"]].copy()
    for col in ["R", "S", "C", "E"]:
        col_min, col_max = radar_df[col].min(), radar_df[col].max()
        if col_max > col_min:
            radar_df[col] = (radar_df[col] - col_min) / (col_max - col_min)
        else:
            radar_df[col] = 0.5

    labels = ["R", "S", "C", "E"]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close loop

    # radar for all models (หนึ่งรูป)
    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    for _, row in radar_df.iterrows():
        vals = row[labels].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, label=row["model"], alpha=0.8)
        ax.fill(angles, vals, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"])
    plt.legend(bbox_to_anchor=(1.2, 1.0))
    plt.title("R,S,C,E Radar Plot")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "radar_R_S_C_E_all_models.png", dpi=300)
    plt.close()

    print(f"\nAll plots saved under: {PLOT_DIR}")


if __name__ == "__main__":
    main()
