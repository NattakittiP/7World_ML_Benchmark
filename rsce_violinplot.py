# RSCE_violin_plot.py
# Violin plot of RSCE stability using CI (low–high) instead of "err"

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = Path("RSCE_errorbars.csv")
OUT_FIG = Path("RSCE_violin.png")

N_SAMPLES = 500
RANDOM_STATE = 42


def main():

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"{CSV_PATH} not found. Make sure RSCE_errorbars.csv exists."
        )

    df = pd.read_csv(CSV_PATH)

    # Must have:
    required = {"model", "RSCE", "RSCE_low", "RSCE_high"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"RSCE_errorbars.csv must contain columns: {required}\n"
            f"Found columns: {list(df.columns)}"
        )

    np.random.seed(RANDOM_STATE)

    models = df["model"].tolist()
    violins = []

    for _, row in df.iterrows():
        mu = float(row["RSCE"])
        low = float(row["RSCE_low"])
        high = float(row["RSCE_high"])

        # CI width
        width = max(high - low, 1e-6)

        # convert CI → sigma (95% CI = mean ± 1.96 sigma → width = 3.92 sigma)
        sigma = width / 3.92

        # sample distribution
        samples = np.random.normal(mu, sigma, size=N_SAMPLES)
        samples = np.clip(samples, 0.0, 1.0)

        violins.append(samples)

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.violinplot(
        violins,
        positions=np.arange(1, len(models) + 1),
        showmeans=True,
        showextrema=False,
        showmedians=False,
    )

    ax.set_xticks(np.arange(1, len(models) + 1))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("RSCE")
    ax.set_title("RSCE stability (approximated from CI widths)")
    ax.set_ylim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300)

    print(f"\n✅ Saved RSCE violin stability plot to: {OUT_FIG}")


if __name__ == "__main__":
    main()
