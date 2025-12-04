# rsce_family_radar.py
# Radar chart ของ RSCE family centroids

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CENTROID_PATH = Path("RSCE_family_centroids.csv")
OUT_PATH = Path("RSCE_family_radar.png")


def load_centroids() -> pd.DataFrame:
    if not CENTROID_PATH.exists():
        raise FileNotFoundError(f"{CENTROID_PATH} not found")
    df = pd.read_csv(CENTROID_PATH)

    # เลือกเฉพาะคอลัมน์ที่ต้องใช้
    keep = ["family", "R_norm", "S_norm", "C_norm", "E_norm", "RSCE"]
    df = df[keep]
    return df


def plot_family_radar(df: pd.DataFrame) -> None:
    metrics = ["R_norm", "S_norm", "C_norm", "E_norm", "RSCE"]
    n_metrics = len(metrics)

    # มุมบนแกน polar
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])  # ปิดวง

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    for _, row in df.iterrows():
        vals = row[metrics].to_numpy(dtype=float)
        vals = np.concatenate([vals, vals[:1]])  # ปิดวง

        ax.plot(angles, vals, label=row["family"])
        ax.fill(angles, vals, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["R", "S", "C", "E", "RSCE"])
    ax.set_yticks(np.linspace(0.0, 1.0, 5))
    ax.set_yticklabels([f"{v:.2f}" for v in np.linspace(0.0, 1.0, 5)])
    ax.set_ylim(0.0, 1.05)

    ax.set_title("Normalized R–S–C–E–RSCE by model family", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=300)
    plt.close(fig)
    print("Saved:", OUT_PATH)


def main():
    df = load_centroids()
    plot_family_radar(df)


if __name__ == "__main__":
    main()
