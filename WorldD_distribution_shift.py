"""
world_d_distribution_shift_full_standalone.py

WORLD D: DISTRIBUTION SHIFT WORLD

Train on original Baseline cohort.
Test on shifted distributions (older, higher BMI, worse lipids).

Shift levels:
    0.0 = no shift (baseline)
    1.0 = moderate shift
    2.0 = severe shift

Output:
    preprocessed_outputs/worldD_distribution_shift_results.csv
"""

from __future__ import annotations

import warnings
from typing import Dict, Any, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    log_loss,
)

# Optional models
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


# =========================
# CONFIG
# =========================

DATA_PATH = Path("Baseline.csv")
OUTPUT_DIR = Path("preprocessed_outputs")
OUTPUT_CSV = OUTPUT_DIR / "worldD_distribution_shift_results.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_COL = "HighTGResponder"

CATEGORICAL_COLS = ["Sex"]
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

# severity 0 = none, 1 = moderate, 2 = severe
SHIFT_LEVELS = [0.0, 1.0, 2.0]

SELECTED_MODELS = [
    "Logistic_L1",
    "Logistic_L2",
    "RandomForest",
    "ExtraTrees",
    "GradientBoosting",
    "XGBoost",
    "LightGBM",
    "GaussianNB",
    "MLP",
    "SVC_RBF",
]


# =========================
# BASIC UTILS
# =========================

def load_raw_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    if "TG4h" not in df.columns:
        raise KeyError("Column 'TG4h' not found in Baseline.csv")
    q75 = df["TG4h"].quantile(0.75)
    df = df.copy()
    df[TARGET_COL] = (df["TG4h"] >= q75).astype(int)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in CATEGORICAL_COLS + NUMERICAL_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def build_preprocessor() -> ColumnTransformer:
    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    num = StandardScaler()
    return ColumnTransformer(
        transformers=[
            ("cat", cat, CATEGORICAL_COLS),
            ("num", num, NUMERICAL_COLS),
        ],
        remainder="drop",
    )


def compute_metrics(y_true, y_prob, y_pred) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_prob)),
    }


# =========================
# MODEL ZOO
# =========================

def get_model_zoo() -> Dict[str, Tuple[Any, Dict]]:
    zoo: Dict[str, Tuple[Any, Dict]] = {}

    zoo["Logistic_L2"] = (
        LogisticRegression(max_iter=500, solver="lbfgs", penalty="l2"),
        {"C": [0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
    )

    zoo["Logistic_L1"] = (
        LogisticRegression(max_iter=500, solver="liblinear", penalty="l1"),
        {"C": [0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
    )

    zoo["RandomForest"] = (
        RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        {"max_depth": [None, 5, 10], "min_samples_split": [2, 5]},
    )

    zoo["ExtraTrees"] = (
        ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE),
        {"max_depth": [None, 5, 10], "min_samples_split": [2, 5]},
    )

    zoo["GradientBoosting"] = (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
    )

    zoo["GaussianNB"] = (GaussianNB(), {})

    zoo["MLP"] = (
        MLPClassifier(max_iter=500, random_state=RANDOM_STATE),
        {"hidden_layer_sizes": [(32,), (64,)], "alpha": [1e-4, 1e-3]},
    )

    zoo["SVC_RBF"] = (
        SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        {"C": [0.5, 1.0, 5.0], "gamma": ["scale", "auto"], "class_weight": [None, "balanced"]},
    )

    if HAS_XGB:
        zoo["XGBoost"] = (
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
            ),
            {"max_depth": [3, 5], "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0]},
        )

    if HAS_LGBM:
        zoo["LightGBM"] = (
            LGBMClassifier(random_state=RANDOM_STATE),
            {"n_estimators": [200, 400], "learning_rate": [0.05, 0.1], "num_leaves": [31, 63]},
        )

    return zoo


def run_model_zoo(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    selected: List[str],
) -> pd.DataFrame:
    zoo = get_model_zoo()
    models = {m: zoo[m] for m in selected if m in zoo}

    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, (estimator, param_grid) in models.items():
        print(f"    Running model: {name}")

        try:
            if param_grid:
                gs = GridSearchCV(
                    estimator,
                    param_grid,
                    scoring="roc_auc",
                    cv=cv,
                    n_jobs=-1,
                    refit=True,
                )
                gs.fit(X_train, y_train)
                best = gs.best_estimator_
                best_cv_auc = float(gs.best_score_)
                params = gs.best_params_
            else:
                estimator.fit(X_train, y_train)
                best = estimator
                gs = GridSearchCV(
                    estimator,
                    {},
                    scoring="roc_auc",
                    cv=cv,
                    refit=True,
                )
                gs.fit(X_train, y_train)
                best_cv_auc = float(gs.best_score_)
                params = {}

            y_prob = best.predict_proba(X_test)[:, 1]
            y_pred = best.predict(X_test)

            row = {
                "model": name,
                "best_cv_roc_auc": best_cv_auc,
                "best_params": params,
            }
            row.update(compute_metrics(y_test, y_prob, y_pred))
            results.append(row)

        except Exception as e:
            warnings.warn(f"Model {name} failed: {e}")

    return pd.DataFrame(results)


# =========================
# DISTRIBUTION SHIFT LOGIC
# =========================

def apply_distribution_shift_inplace(df_test: pd.DataFrame, severity: float) -> None:
    """
    Apply distribution shift ONLY on test set:
        - Age: + 5 * severity years
        - BMI: + 1.5 * severity
        - TG0h, LDL: multiplicative increase
        - HDL: multiplicative decrease
    """
    if severity <= 0.0:
        return

    if "Age" in df_test.columns:
        df_test["Age"] = df_test["Age"] + 5.0 * severity

    if "BMI" in df_test.columns:
        df_test["BMI"] = df_test["BMI"] + 1.5 * severity

    if "TG0h" in df_test.columns:
        df_test["TG0h"] = df_test["TG0h"] * (1.0 + 0.10 * severity)

    if "LDL" in df_test.columns:
        df_test["LDL"] = df_test["LDL"] * (1.0 + 0.10 * severity)

    if "HDL" in df_test.columns:
        df_test["HDL"] = df_test["HDL"] * max(0.0, (1.0 - 0.05 * severity))


# =========================
# MAIN
# =========================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_data(DATA_PATH)
    df = create_target(df_raw)
    validate_columns(df)

    # Fixed train/test split
    df_train_base, df_test_base = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET_COL],
        random_state=RANDOM_STATE,
    )
    train_idx, test_idx = df_train_base.index, df_test_base.index

    all_results = []

    for severity in SHIFT_LEVELS:
        print(f"\n====== WORLD D — shift_severity = {severity:.1f} ======")

        df_train = df.loc[train_idx].copy()
        df_test = df.loc[test_idx].copy()

        # apply shift ONLY on test
        apply_distribution_shift_inplace(df_test, severity)

        pre = build_preprocessor()

        X_train = df_train.drop(columns=[TARGET_COL])
        y_train = df_train[TARGET_COL].to_numpy()

        X_test = df_test.drop(columns=[TARGET_COL])
        y_test = df_test[TARGET_COL].to_numpy()

        pre.fit(X_train)
        X_train_p = pre.transform(X_train)
        X_test_p = pre.transform(X_test)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_df = run_model_zoo(X_train_p, y_train, X_test_p, y_test, SELECTED_MODELS)

        res_df["shift_severity"] = severity
        all_results.append(res_df)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(OUTPUT_CSV, index=False)

    print("\nDONE — World D results saved to:", OUTPUT_CSV)
    print(final_df.head(10))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
