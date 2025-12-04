# rsce_centroid_plot.py
# RSCE family centroids + plots

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

RSC_PATH = Path("scores_R_S_C.csv")
E_PATH = Path("scores_E_all.csv")
OUT_DIR = Path("plots_RSCE")
OUT_CSV = OUT_DIR / "RSCE_family_centroids.csv"

# RSCE weights (Option C)
W_R, W_S, W_C, W_E = 0.4, 0.3, 0.2, 0.1


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------
# LOAD + RSCE
# ---------------------------------------------------

def load_and_compute_rsce() -> pd.DataFrame:
    if not RSC_PATH.exists():
        raise FileNotFoundError(f"{RSC_PATH} not found")
    if not E_PATH.exists():
        raise FileNotFoundError(f"{E_PATH} not found")

    rsc = pd.read_csv(RSC_PATH)
    e = pd.read_csv(E_PATH)[["model", "E"]]

    df = pd.merge(rsc, e, on="model", how="inner")

    # normalize R,S,C,E
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

    df = df.sort_values("RSCE", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------
# FAMILY DEFINITION
# ---------------------------------------------------

def assign_family(model_name: str) -> str:
    """Map model name -> family label."""
    if model_name in ["ExtraTrees", "RandomForest"]:
        return "Tree-based (bagging)"
    if model_name in ["XGBoost", "GradientBoosting", "LightGBM"]:
        return "Boosting"
    if model_name in ["Logistic_L1", "Logistic_L2"]:
        return "Linear models"
    if model_name == "SVC_RBF":
        return "Kernel SVM"
    if model_name == "MLP":
        return "Neural network"
    if model_name == "GaussianNB":
        return "Bayes"
    # fallback (กรณีเผื่อมี model อื่นในอนาคต)
    return "Other"


def compute_family_centroids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["family"] = df["model"].apply(assign_family)

    # metrics ที่จะใช้ทำ centroid
    metrics_norm = ["R_norm", "S_norm", "C_norm", "E_norm", "RSCE"]

    grp = df.groupby("family", as_index=False)[metrics_norm].mean()

    # นับจำนวนโมเดลในแต่ละ family ไว้ด้วย
    counts = df.groupby("family", as_index=False)["model"].count()
    counts = counts.rename(columns={"model": "n_models"})
    grp = grp.merge(counts, on="family", how="left")

    # เรียงตาม RSCE
    grp = grp.sort_values("RSCE", ascending=False).reset_index(drop=True)
    return grp


# ---------------------------------------------------
# PLOTS
# ---------------------------------------------------

def plot_family_rsce_bar(centroids: pd.DataFrame) -> None:
    ensure_dir(OUT_DIR)

    fam = centroids["family"].tolist()
    rsce = centroids["RSCE"].to_numpy()
    n_models = centroids["n_models"].to_numpy()

    x = np.arange(len(fam))

    plt.figure(figsize=(7, 4))
    plt.bar(x, rsce)
    for i, (rs, n) in enumerate(zip(rsce, n_models)):
        plt.text(i, rs + 0.02, f"n={n}", ha="center", va="bottom", fontsize=8)

    plt.xticks(x, fam, rotation=30, ha="right")
    plt.ylabel("RSCE (family centroid, Option C)")
    plt.title("RSCE by model family")
    plt.ylim(0.0, min(1.1, rsce.max() + 0.15))
    plt.tight_layout()

    out_path = OUT_DIR / "RSCE_family_bar.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)


def plot_family_centroid_scatter(centroids: pd.DataFrame) -> None:
    """2D scatter: R_norm vs S_norm, ขนาดจุด ~ RSCE, label = family."""
    ensure_dir(OUT_DIR)

    x = centroids["R_norm"].to_numpy()
    y = centroids["S_norm"].to_numpy()
    rsce = centroids["RSCE"].to_numpy()
    fam = centroids["family"].tolist()

    # ทำขนาดจุดตาม RSCE (scale ให้ดูสวย)
    sizes = 300 * (0.3 + rsce)  # ให้ทุกจุดมีขนาดขั้นต่ำ

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=sizes, alpha=0.8)

    for xi, yi, name in zip(x, y, fam):
        plt.text(xi, yi, name, ha="center", va="center", fontsize=9)

    plt.xlabel("R_norm (robustness)")
    plt.ylabel("S_norm (score stability)")
    plt.title("Family centroids in R–S space (size ∝ RSCE)")
    plt.xlim(0.0, 1.05)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    out_path = OUT_DIR / "RSCE_family_centroid_scatter.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main():
    df = load_and_compute_rsce()
    centroids = compute_family_centroids(df)

    ensure_dir(OUT_DIR)
    centroids.to_csv(OUT_CSV, index=False)
    print("Saved family centroid table to:", OUT_CSV)
    print("\nFamily centroids:")
    print(centroids)

    plot_family_rsce_bar(centroids)
    plot_family_centroid_scatter(centroids)


if __name__ == "__main__":
    main()
