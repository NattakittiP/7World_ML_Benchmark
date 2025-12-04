# rsce_cluster_map.py
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram

RSC_PATH = Path("scores_R_S_C.csv")
E_PATH = Path("scores_E_all.csv")
OUT_DIR = Path("plots_RSCE")

W_R, W_S, W_C, W_E = 0.4, 0.3, 0.2, 0.1


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_and_compute_rsce() -> pd.DataFrame:
    rsc = pd.read_csv(RSC_PATH)
    e = pd.read_csv(E_PATH)[["model", "E"]]
    df = pd.merge(rsc, e, on="model", how="inner")

    # normalize
    for col in ["R", "S", "C", "E"]:
        mn, mx = df[col].min(), df[col].max()
        if mx > mn:
            df[col + "_norm"] = (df[col] - mn) / (mx - mn)
        else:
            df[col + "_norm"] = 0.0

    df["RSCE"] = (
        W_R * df["R_norm"]
        + W_S * df["S_norm"]
        + W_C * df["C_norm"]
        + W_E * df["E_norm"]
    )

    df = df.sort_values("RSCE", ascending=False).reset_index(drop=True)
    return df


def plot_cluster_map(df: pd.DataFrame) -> None:
    ensure_dir(OUT_DIR)

    # ใช้ normalized metrics สำหรับ clustering
    features = ["R_norm", "S_norm", "C_norm", "E_norm"]
    X = df[features].to_numpy()

    # hierarchical clustering (Ward)
    Z = linkage(X, method="ward")
    order = leaves_list(Z)

    df_ord = df.iloc[order].reset_index(drop=True)
    X_ord = df_ord[["R", "S", "C", "E", "RSCE"]].to_numpy()

    # สร้าง figure: ซ้าย dendrogram, ขวา heatmap
    fig = plt.figure(figsize=(8, 6))
    # axes ขวา (heatmap)
    ax_heat = fig.add_axes([0.3, 0.1, 0.6, 0.8])
    im = ax_heat.imshow(X_ord, aspect="auto")

    ax_heat.set_xticks(np.arange(X_ord.shape[1]))
    ax_heat.set_xticklabels(["R", "S", "C", "E", "RSCE"])
    ax_heat.set_yticks(np.arange(len(df_ord)))
    ax_heat.set_yticklabels(df_ord["model"].tolist())
    ax_heat.set_title("RSCE Cluster Map")

    # แสดงค่าใน cell (optional)
    for i in range(X_ord.shape[0]):
        for j in range(X_ord.shape[1]):
            ax_heat.text(
                j,
                i,
                f"{X_ord[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
            )

    # colorbar
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Metric value", rotation=270, labelpad=10)

    # axes ซ้าย (dendrogram)
    ax_dend = fig.add_axes([0.05, 0.1, 0.2, 0.8])
    dendrogram(Z, orientation="right", labels=df["model"].tolist(), ax=ax_dend)
    ax_dend.set_yticklabels([])  # ไม่ให้ label ซ้ำกับ heatmap
    ax_dend.set_xticks([])
    ax_dend.set_title("Model clusters")

    out_path = OUT_DIR / "RSCE_cluster_map.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)


def main():
    df = load_and_compute_rsce()
    plot_cluster_map(df)


if __name__ == "__main__":
    main()
