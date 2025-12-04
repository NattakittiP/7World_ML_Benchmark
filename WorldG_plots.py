# worldG_plots.py
# Visualization for World G (label-noise robustness)
# ใช้เฉพาะ worldG_label_noise_results.csv ไม่ต้องเทรนโมเดลใหม่

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = Path("worldG_label_noise_results.csv")
OUT_DIR = Path("plots_WorldG")

FOCUS_MODELS = [
    "ExtraTrees",
    "RandomForest",
    "GradientBoosting",
    "Logistic_L1",
    "MLP",
    "SVC_RBF",
]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def plot_auc_vs_noise(df: pd.DataFrame) -> None:
    """
    Figure G1: AUC vs random symmetric noise (G1, RSLN)
    """
    df_r = df[df["noise_type"] == "RSLN"].copy()
    if df_r.empty:
        print("No RSLN rows found in CSV; skip AUC vs noise plot.")
        return

    ensure_dir(OUT_DIR)

    plt.figure(figsize=(6, 4))
    for model in FOCUS_MODELS:
        sub = df_r[df_r["model"] == model]
        if sub.empty:
            continue
        # noise_rate อาจเป็น NaN ถ้าเขียนผิด แต่ใน G1 เรามีค่า 0.05 และ 0.10
        x = sub["noise_rate"].to_numpy()
        y = sub["roc_auc"].to_numpy()
        order = np.argsort(x)
        plt.plot(
            x[order],
            y[order],
            marker="o",
            label=model,
        )

    plt.xlabel("Random symmetric label noise rate")
    plt.ylabel("ROC AUC")
    plt.title("World G1 (Random Symmetric Label Noise): AUC vs noise")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "G1_auc_vs_noise.png", dpi=300)
    plt.close()
    print("Saved:", OUT_DIR / "G1_auc_vs_noise.png")


def plot_brier_vs_noise(df: pd.DataFrame) -> None:
    """
    Figure G2: Brier score vs random symmetric noise (G1, RSLN)
    """
    df_r = df[df["noise_type"] == "RSLN"].copy()
    if df_r.empty:
        print("No RSLN rows found in CSV; skip Brier vs noise plot.")
        return

    ensure_dir(OUT_DIR)

    plt.figure(figsize=(6, 4))
    for model in FOCUS_MODELS:
        sub = df_r[df_r["model"] == model]
        if sub.empty:
            continue
        x = sub["noise_rate"].to_numpy()
        y = sub["brier"].to_numpy()
        order = np.argsort(x)
        plt.plot(
            x[order],
            y[order],
            marker="o",
            label=model,
        )

    plt.xlabel("Random symmetric label noise rate")
    plt.ylabel("Brier score")
    plt.title("World G1 (Random Symmetric Label Noise): Brier vs noise")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "G1_brier_vs_noise.png", dpi=300)
    plt.close()
    print("Saved:", OUT_DIR / "G1_brier_vs_noise.png")


def plot_class_conditional_bar(df: pd.DataFrame) -> None:
    """
    Figure G3: AUC under class-conditional noise (G2, CCLN)
    """
    df_c = df[df["noise_type"] == "CCLN"].copy()
    if df_c.empty:
        print("No CCLN rows found in CSV; skip class-conditional plot.")
        return

    ensure_dir(OUT_DIR)

    # เอาเฉพาะโมเดลที่เราสนใจ
    df_c = df_c[df_c["model"].isin(FOCUS_MODELS)]

    x = np.arange(len(df_c))
    labels = df_c["model"].tolist()
    aucs = df_c["roc_auc"].to_numpy()
    briers = df_c["brier"].to_numpy()

    # AUC bar
    plt.figure(figsize=(6, 4))
    plt.bar(x, aucs)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("ROC AUC")
    plt.title("World G2 (Class-Conditional Label Noise): AUC by model")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "G2_auc_bar.png", dpi=300)
    plt.close()
    print("Saved:", OUT_DIR / "G2_auc_bar.png")

    # Brier bar
    plt.figure(figsize=(6, 4))
    plt.bar(x, briers)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Brier score")
    plt.title("World G2 (Class-Conditional Label Noise): Brier by model")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "G2_brier_bar.png", dpi=300)
    plt.close()
    print("Saved:", OUT_DIR / "G2_brier_bar.png")


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"{CSV_PATH} not found")

    df = pd.read_csv(CSV_PATH)

    print("worldG_label_noise_results.csv loaded with shape:", df.shape)

    plot_auc_vs_noise(df)
    plot_brier_vs_noise(df)
    plot_class_conditional_bar(df)

    print("\nAll World G plots saved under:", OUT_DIR)


if __name__ == "__main__":
    main()
