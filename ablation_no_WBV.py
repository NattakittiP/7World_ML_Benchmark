# ablation_no_WBV.py
# Evaluate top models when WBV is removed from the feature set (World A only).

from __future__ import annotations
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    log_loss,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

DATA_PATH = Path("Baseline.csv")
OUT_CSV = Path("ablation_no_WBV_results.csv")
OUT_TXT = Path("Ablation_no_WBV.txt")

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "HighTGResponder"

CATEGORICAL_COLS = ["Sex"]
NUMERICAL_COLS_FULL = [
    "Age",
    "Hematocrit",
    "TotalProtein",
    "WBV",
    "TG0h",
    "HDL",
    "LDL",
    "BMI",
]

# สำหรับ ablation นี้ ตัด WBV ออก
NUMERICAL_COLS_NOWBV = [
    c for c in NUMERICAL_COLS_FULL if c != "WBV"
]


# ---------------- Data ----------------

def load_and_prepare():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found")

    df = pd.read_csv(DATA_PATH)
    if "TG4h" not in df.columns:
        raise KeyError("Baseline.csv must contain TG4h")

    q75 = df["TG4h"].quantile(0.75)
    df[TARGET_COL] = (df["TG4h"] >= q75).astype(int)

    X = df[CATEGORICAL_COLS + NUMERICAL_COLS_NOWBV]
    y = df[TARGET_COL].to_numpy()

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    num = StandardScaler()

    pre = ColumnTransformer(
        [
            ("cat", cat, CATEGORICAL_COLS),
            ("num", num, NUMERICAL_COLS_NOWBV),
        ],
        remainder="drop",
    )

    pre.fit(X_train_df)

    X_train = pre.transform(X_train_df)
    X_test = pre.transform(X_test_df)

    return X_train, X_test, y_train, y_test


# ---------------- Models ----------------

def get_models() -> Dict[str, object]:
    models = {}

    models["Logistic_L1"] = LogisticRegression(
        max_iter=500,
        solver="liblinear",
        penalty="l1",
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=2,
        random_state=RANDOM_STATE,
    )

    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=2,
        random_state=RANDOM_STATE,
    )

    models["GradientBoosting"] = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
    )

    models["MLP"] = MLPClassifier(
        hidden_layer_sizes=(32,),
        alpha=1e-3,
        max_iter=500,
        random_state=RANDOM_STATE,
    )

    models["SVC_RBF"] = SVC(
        kernel="rbf",
        probability=True,
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    return models


# ---------------- Metrics ----------------

def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict:
    eps = 1e-15
    p1 = np.clip(proba[:, 1], eps, 1 - eps)
    y_pred = (p1 >= 0.5).astype(int)

    roc = roc_auc_score(y_true, p1)
    pr = average_precision_score(y_true, p1)
    brier = brier_score_loss(y_true, p1)
    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, proba, labels=[0, 1])

    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "brier": brier,
        "accuracy": acc,
        "log_loss": ll,
    }


# ---------------- Main ----------------

def main():
    X_train, X_test, y_train, y_test = load_and_prepare()
    models = get_models()

    rows = []
    lines = []

    for mname, model in models.items():
        clf = model
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        metrics = evaluate(y_test, proba)

        row = {"model": mname}
        row.update(metrics)
        rows.append(row)

        lines.append(
            f"{mname:12s} | AUC={metrics['roc_auc']:.3f}, "
            f"PR={metrics['pr_auc']:.3f}, "
            f"Brier={metrics['brier']:.3f}, "
            f"Acc={metrics['accuracy']:.3f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    with OUT_TXT.open("w", encoding="utf-8") as f:
        f.write("Ablation: Removing WBV from feature set (World A)\n\n")
        f.write("\n".join(lines))

    print(f"\nSaved CSV to {OUT_CSV}")
    print(f"Saved text summary to {OUT_TXT}")


if __name__ == "__main__":
    main()
