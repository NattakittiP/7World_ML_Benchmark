# ablation_no_calibration.py
# Compare top models with vs without probability calibration on World A (Baseline).

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict

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

from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

DATA_PATH = Path("Baseline.csv")
OUT_CSV = Path("ablation_no_calibration_results.csv")
OUT_TXT = Path("Ablation_no_calibration.txt")

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


# ---------------- Data ----------------

def load_and_prepare():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found")

    df = pd.read_csv(DATA_PATH)
    if "TG4h" not in df.columns:
        raise KeyError("Baseline.csv must contain TG4h")

    q75 = df["TG4h"].quantile(0.75)
    df[TARGET_COL] = (df["TG4h"] >= q75).astype(int)

    X = df[CATEGORICAL_COLS + NUMERICAL_COLS]
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
            ("num", num, NUMERICAL_COLS),
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


# ---------------- ECE ----------------

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Simple Expected Calibration Error with equal-width bins in [0,1].
    """
    eps = 1e-15
    p1 = np.clip(y_prob, eps, 1 - eps)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p1, bins) - 1  # 0..n_bins-1

    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        frac = np.mean(mask)
        avg_conf = np.mean(p1[mask])
        avg_acc = np.mean(y_true[mask])
        ece += frac * abs(avg_conf - avg_acc)
    return float(ece)


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
    ece = compute_ece(y_true, p1, n_bins=10)

    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "brier": brier,
        "accuracy": acc,
        "log_loss": ll,
        "ece": ece,
    }


# ---------------- Main ----------------

def main():
    X_train, X_test, y_train, y_test = load_and_prepare()
    models = get_models()

    rows = []
    lines = []

    for mname, base_model in models.items():
        # RAW (no calibration)
        raw_clf = base_model
        raw_clf.fit(X_train, y_train)
        proba_raw = raw_clf.predict_proba(X_test)
        m_raw = evaluate(y_test, proba_raw)

        rows.append(
            {
                "model": mname,
                "variant": "raw",
                **m_raw,
            }
        )

        lines.append(
            f"{mname:12s} [RAW]  AUC={m_raw['roc_auc']:.3f}, "
            f"PR={m_raw['pr_auc']:.3f}, Brier={m_raw['brier']:.3f}, "
            f"ECE={m_raw['ece']:.3f}"
        )

       # CALIBRATED (isotonic)
        calib = CalibratedClassifierCV(
            estimator=base_model,   # <- ใช้ estimator แทน base_estimator
            cv=5,
            method="isotonic",
        )
        calib.fit(X_train, y_train)
        proba_cal = calib.predict_proba(X_test)
        m_cal = evaluate(y_test, proba_cal)

        rows.append(
            {
                "model": mname,
                "variant": "calibrated_isotonic",
                **m_cal,
            }
        )

        lines.append(
            f"{mname:12s} [CAL] AUC={m_cal['roc_auc']:.3f}, "
            f"PR={m_cal['pr_auc']:.3f}, Brier={m_cal['brier']:.3f}, "
            f"ECE={m_cal['ece']:.3f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    with OUT_TXT.open("w", encoding="utf-8") as f:
        f.write("Ablation: No calibration vs isotonic calibration (World A)\n\n")
        f.write("\n".join(lines))

    print(f"\nSaved CSV to {OUT_CSV}")
    print(f"Saved text summary to {OUT_TXT}")


if __name__ == "__main__":
    main()
