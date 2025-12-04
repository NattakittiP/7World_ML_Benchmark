"""
preprocess.py  —  Unified Version (Config + Pipeline in ONE file)

This script:
1. Loads the raw synthetic cohort CSV.
2. Creates binary target "HighTGResponder" from TG4h (>= 75th percentile).
3. Stratified train/test split.
4. Builds preprocessing pipeline:
     - OneHotEncoder for categorical
     - StandardScaler for numeric
     - EXCLUDES TG4h and TCR from features
5. Saves:
     - preprocessor.joblib
     - train_arrays.npz / test_arrays.npz
     - summary.json

Run:
    python preprocess.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CONFIG (in the same file)
# ============================================================

class CFG:
    RAW_DATA_PATH = Path("Baseline.csv")

    OUTPUT_DIR = Path("preprocessed_outputs")
    PREPROCESSOR_PATH = OUTPUT_DIR / "preprocessor.joblib"
    TRAIN_ARRAYS_PATH = OUTPUT_DIR / "train_arrays.npz"
    TEST_ARRAYS_PATH = OUTPUT_DIR / "test_arrays.npz"
    SUMMARY_JSON_PATH = OUTPUT_DIR / "summary.json"

    TARGET_COL = "HighTGResponder"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # Categorical
    CATEGORICAL_COLS = ["Sex"]

    # Numerical (NO TG4h, NO TCR)
    NUMERICAL_COLS = [
        "Age",
        "Hematocrit",
        "TotalProtein",
        "WBV",
        "TG0h",
        "HDL",
        "LDL",
        "BMI",
    ]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def ensure_output_dir() -> None:
    CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"ERROR: Cannot find {path}")
    return pd.read_csv(path)


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    HighTGResponder = 1 if TG4h >= 75th percentile.
    TG4h is used ONLY for target creation, not as feature.
    """
    if "TG4h" not in df.columns:
        raise KeyError("Column TG4h not found in raw data.")

    threshold = df["TG4h"].quantile(0.75)

    df = df.copy()
    df[CFG.TARGET_COL] = (df["TG4h"] >= threshold).astype(int)
    return df


def validate_feature_columns(df: pd.DataFrame) -> None:
    missing = []
    for col in CFG.CATEGORICAL_COLS + CFG.NUMERICAL_COLS:
        if col not in df.columns:
            missing.append(col)
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")


def do_train_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = df[CFG.TARGET_COL]
    df_train, df_test = train_test_split(
        df,
        test_size=CFG.TEST_SIZE,
        stratify=y,
        random_state=CFG.RANDOM_STATE,
    )
    return df_train, df_test


def build_preprocessor() -> ColumnTransformer:
    cat_tf = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    num_tf = StandardScaler()

    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_tf, CFG.CATEGORICAL_COLS),
            ("num", num_tf, CFG.NUMERICAL_COLS),
        ],
        remainder="drop",
    )
    return pre


def fit_and_transform(
    preprocessor: ColumnTransformer,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    X_train = df_train.drop(columns=[CFG.TARGET_COL])
    y_train = df_train[CFG.TARGET_COL].values

    X_test = df_test.drop(columns=[CFG.TARGET_COL])
    y_test = df_test[CFG.TARGET_COL].values

    preprocessor.fit(X_train)
    X_train_p = preprocessor.transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    return X_train_p, X_test_p, y_train, y_test


def build_summary(
    df: pd.DataFrame, df_train: pd.DataFrame, df_test: pd.DataFrame
) -> Dict[str, Any]:

    tgt = CFG.TARGET_COL

    summary = {
        "n_total": len(df),
        "n_train": len(df_train),
        "n_test": len(df_test),
        "target_distribution": {
            "total": df[tgt].value_counts(normalize=True).to_dict(),
            "train": df_train[tgt].value_counts(normalize=True).to_dict(),
            "test": df_test[tgt].value_counts(normalize=True).to_dict(),
        },
        "features_used": CFG.CATEGORICAL_COLS + CFG.NUMERICAL_COLS,
        "excluded_features": ["TG4h", "TCR", "ID"],
    }
    return summary


def save_outputs(
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test,
    summary: Dict[str, Any],
):
    dump(preprocessor, CFG.PREPROCESSOR_PATH)

    np.savez_compressed(CFG.TRAIN_ARRAYS_PATH, X=X_train, y=y_train)
    np.savez_compressed(CFG.TEST_ARRAYS_PATH, X=X_test, y=y_test)

    with open(CFG.SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ensure_output_dir()

    df_raw = load_raw(CFG.RAW_DATA_PATH)
    df = create_target(df_raw)

    validate_feature_columns(df)

    df_train, df_test = do_train_test_split(df)

    pre = build_preprocessor()

    X_train, X_test, y_train, y_test = fit_and_transform(
        pre, df_train, df_test
    )

    summary = build_summary(df, df_train, df_test)
    save_outputs(pre, X_train, X_test, y_train, y_test, summary)

    print("--------------------------------------------------")
    print("✅ PREPROCESSING COMPLETED")
    print("Train shape:", X_train.shape, "Labels:", y_train.shape)
    print("Test shape :", X_test.shape, "Labels:", y_test.shape)
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()