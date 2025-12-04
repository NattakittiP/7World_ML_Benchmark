# plot_RSCE_dashboard.py
# RSCE Barplot, Radar Chart, 3D Cube, Heatmap
# ใช้ RSCE แบบ Option C:
# RSCE = 0.4*R_norm + 0.3*S_norm + 0.2*C_norm + 0.1*E_norm

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

RSC_PATH = Path("scores_R_S_C.csv")
E_PATH = Path("scores_E_all.csv")
OUT_DIR = Path("plots_RSCE")

# น้ำหนักสำหรับ RSCE Option C
W_R = 0.4
W_S = 0.3
W_C = 0.2
W_E = 0.1


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------
# LOAD & MERGE
# ---------------------------------------------------

def load_and_compute_rsce() -> pd.DataFrame:
    if not RSC_PATH.exists():
        raise FileNotFoundError(f"{RSC_PATH} not found")
    if not E_PATH.exists():
        raise FileNotFoundError(f"{E_PATH} not found")

    rsc = pd.read_csv(RSC_PATH)
    e = pd.read_csv(E_PATH)

    # เอาเฉพาะ column model + E
    e = e[["model", "E"]].copy()

    df = pd.merge(rsc, e, on="model", how="inner")

    # min-max normalize
    for col in ["R", "S", "C", "E"]:
        mn, mx = df[col].min(), df[col].max()
        if mx > mn:
            df[col + "_norm"] = (df[col] - mn) / (mx - mn)
        else:
            df[col + "_norm"] = 0.0

    # RSCE Option C
    df["RSCE"] = (
        W_R * df["R_norm"]
        + W_S * df["S_norm"]
        + W_C * df["C_norm"]
        + W_E * df["E_norm"]
    )

    # เรียงจาก RSCE มาก → น้อย
    df = df.sort_values("RSCE", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------
# PLOTS
# ---------------------------------------------------

def plot_rsce_bar(df: pd.DataFrame) -> None:
    ensure_dir(OUT_DIR)
    models = df["model"].tolist()
    rsce = df["RSCE"].to_numpy()

    plt.figure(figsize=(6, 4))
    plt.bar(models, rsce)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("RSCE (Option C)")
    plt.title("RSCE by Model")
    plt.tight_layout()
    out_path = OUT_DIR / "RSCE_barplot.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)


def plot_rsce_radar(df: pd.DataFrame, top_k: int = 6) -> None:
    """
    Radar chart ของ normalized R,S,C,E
    โดยดึงเฉพาะ top_k models ตาม RSCE (เพื่อไม่ให้กราฟรกเกินไป)
    """
    ensure_dir(OUT_DIR)

    metrics = ["R_norm", "S_norm", "C_norm", "E_norm"]
    labels_metrics = ["R", "S", "C", "E"]

    k = min(top_k, len(df))
    sub = df.head(k)

    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])  # ปิดวงให้ครบ

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    for _, row in sub.iterrows():
        values = row[metrics].to_numpy()
        values = np.concatenate([values, values[:1]])
        ax.plot(angles, values, linewidth=1)
        ax.fill(angles, values, alpha=0.1)
        # label ชื่อ model ไว้ใน legend
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_metrics)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Normalized R–S–C–E (Top models by RSCE)")
    ax.legend(sub["model"].tolist(), bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.tight_layout()
    out_path = OUT_DIR / "RSCE_radar.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)


def plot_rsce_3d_cube(df: pd.DataFrame) -> None:
    """
    3D scatter: R_norm, S_norm, C_norm (cube)
    ขนาด/ตำแหน่งสะท้อน reliability ของแต่ละ model
    """
    ensure_dir(OUT_DIR)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    x = df["R_norm"].to_numpy()
    y = df["S_norm"].to_numpy()
    z = df["C_norm"].to_numpy()

    ax.scatter(x, y, z, s=50)

    for xi, yi, zi, name in zip(x, y, z, df["model"]):
        ax.text(xi, yi, zi, name)

    ax.set_xlabel("R_norm")
    ax.set_ylabel("S_norm")
    ax.set_zlabel("C_norm")
    ax.set_title("RSCE Cube: (R_norm, S_norm, C_norm)")
    plt.tight_layout()
    out_path = OUT_DIR / "RSCE_3D_cube.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)


def plot_rsce_heatmap(df: pd.DataFrame) -> None:
    """
    Heatmap ของ metrics หลัก: R,S,C,E,RSCE
    """
    ensure_dir(OUT_DIR)

    metrics = ["R", "S", "C", "E", "RSCE"]
    data = df[metrics].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(data, aspect="auto")

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["model"].tolist())
    ax.set_title("R–S–C–E–RSCE Heatmap")

    # แสดงค่าใน cell (optional ถ้าอยากอ่านง่าย)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = f"{data[i, j]:.2f}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=6,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_path = OUT_DIR / "RSCE_heatmap.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)


def main():
    df = load_and_compute_rsce()
    print("Merged & RSCE-computed table:")
    print(df[["model", "R", "S", "C", "E", "RSCE"]])

    plot_rsce_bar(df)
    plot_rsce_radar(df)
    plot_rsce_3d_cube(df)
    plot_rsce_heatmap(df)

    print("\nAll RSCE plots saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
