# rsce_error_bars.py
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RSC_PATH = Path("scores_R_S_C.csv")
E_PATH = Path("scores_E_all.csv")
OUT_DIR = Path("plots_RSCE")
OUT_CSV = OUT_DIR / "RSCE_errorbars.csv"

# น้ำหนัก Option C
W_R, W_S, W_C, W_E = 0.4, 0.3, 0.2, 0.1

# สมมติส่วนเบี่ยงเบนมาตรฐานของ RSCE ~ 3% ของค่ากลาง (ปรับได้)
RSCE_REL_STD = 0.03
N_BOOT = 2000


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_and_compute_rsce() -> pd.DataFrame:
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

    df["RSCE"] = (
        W_R * df["R_norm"]
        + W_S * df["S_norm"]
        + W_C * df["C_norm"]
        + W_E * df["E_norm"]
    )

    df = df.sort_values("RSCE", ascending=False).reset_index(drop=True)
    return df


def bootstrap_rsce(df: pd.DataFrame) -> pd.DataFrame:
    """Parametric bootstrap รอบ RSCE (approximate)."""
    rsce = df["RSCE"].to_numpy()
    n_models = len(df)

    # เก็บ bootstrap samples: shape (N_BOOT, n_models)
    samples = np.zeros((N_BOOT, n_models), dtype=float)

    for b in range(N_BOOT):
        # สมมติ RSCE มี Gaussian noise ~ Normal(0, (σ·RSCE)^2)
        noise_std = RSCE_REL_STD * rsce
        noise = np.random.normal(loc=0.0, scale=noise_std)
        samples[b, :] = rsce + noise

    # CI 95%
    lower = np.percentile(samples, 2.5, axis=0)
    upper = np.percentile(samples, 97.5, axis=0)

    out = df.copy()
    out["RSCE_low"] = lower
    out["RSCE_high"] = upper
    return out


def plot_rsce_bar_with_error(df_ci: pd.DataFrame) -> None:
    ensure_dir(OUT_DIR)

    models = df_ci["model"].tolist()
    rsce = df_ci["RSCE"].to_numpy()
    low = df_ci["RSCE_low"].to_numpy()
    high = df_ci["RSCE_high"].to_numpy()

    x = np.arange(len(models))

    plt.figure(figsize=(8, 4))
    plt.bar(x, rsce)
    # asymmetric error bars
    yerr = np.vstack([rsce - low, high - rsce])
    plt.errorbar(x, rsce, yerr=yerr, fmt="none", capsize=4)

    plt.xticks(x, models, rotation=45, ha="right")
    plt.ylabel("RSCE (Option C)")
    plt.title("RSCE with approximate uncertainty bands")
    plt.tight_layout()

    out_path = OUT_DIR / "RSCE_bar_with_errorbars.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)


def main():
    df = load_and_compute_rsce()
    df_ci = bootstrap_rsce(df)
    ensure_dir(OUT_DIR)
    df_ci.to_csv(OUT_CSV, index=False)
    print("Saved RSCE CI table to:", OUT_CSV)

    plot_rsce_bar_with_error(df_ci)


if __name__ == "__main__":
    main()
