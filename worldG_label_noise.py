# worldG_label_noise.py
# Label-noise robustness world (World G)
# - G1: Random symmetric label noise at 5% and 10%
# - G2: Class-conditional label noise (15% for positives, 5% for negatives)

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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

DATA_PATH = Path("Baseline.csv")
OUT_CSV = Path("worldG_label_noise_results.csv")
OUT_TXT = Path("World G result.txt")

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


# ------------------------------------------
# Data & Preprocessing
# ------------------------------------------

def load_and_prepare() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found")

    df = pd.read_csv(DATA_PATH)

    if "TG4h" not in df.columns:
        raise KeyError("Baseline.csv must contain TG4h column")

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

    return X_train, X_test, y_train, y_test, pre


# ------------------------------------------
# Model zoo for World G
# ------------------------------------------

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

    models["Logistic_L2"] = LogisticRegression(
        max_iter=500,
        solver="lbfgs",
        penalty="l2",
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

    models["AdaBoost"] = AdaBoostClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
    )

    models["GaussianNB"] = GaussianNB()

    models["KNN"] = KNeighborsClassifier(
        n_neighbors=11,
        weights="distance",
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


# ------------------------------------------
# Label noise generators
# ------------------------------------------

def apply_random_symmetric_noise(
    y: np.ndarray,
    noise_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Flip labels at random (0 <-> 1) with probability = noise_rate."""
    y_noisy = y.copy()
    n = len(y_noisy)
    n_flip = int(noise_rate * n)
    if n_flip <= 0:
        return y_noisy

    idx = rng.choice(n, size=n_flip, replace=False)
    y_noisy[idx] = 1 - y_noisy[idx]
    return y_noisy


def apply_class_conditional_noise(
    y: np.ndarray,
    pos_flip_rate: float,
    neg_flip_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Flip labels with different rates by class:
    - flip 1 -> 0 with pos_flip_rate
    - flip 0 -> 1 with neg_flip_rate
    """
    y_noisy = y.copy()
    idx_pos = np.where(y_noisy == 1)[0]
    idx_neg = np.where(y_noisy == 0)[0]

    n_flip_pos = int(pos_flip_rate * len(idx_pos))
    n_flip_neg = int(neg_flip_rate * len(idx_neg))

    if n_flip_pos > 0:
        flip_pos = rng.choice(idx_pos, size=n_flip_pos, replace=False)
        y_noisy[flip_pos] = 0

    if n_flip_neg > 0:
        flip_neg = rng.choice(idx_neg, size=n_flip_neg, replace=False)
        y_noisy[flip_neg] = 1

    return y_noisy


# ------------------------------------------
# Metric helper
# ------------------------------------------

def evaluate_model(y_true: np.ndarray, proba: np.ndarray) -> dict:
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


# ------------------------------------------
# Main
# ------------------------------------------

def main():
    X_train, X_test, y_train_clean, y_test, pre = load_and_prepare()
    models = get_models()

    rng = np.random.default_rng(RANDOM_STATE)

    rows = []
    txt_lines = []

    scenarios = []

    # G1: random symmetric 5% and 10%
    for rate in [0.05, 0.10]:
        scenarios.append(
            {
                "scenario": "G1_random_symmetric",
                "noise_type": "RSLN",
                "noise_rate": rate,
                "pos_flip_rate": rate,
                "neg_flip_rate": rate,
            }
        )

    # G2: class-conditional (1->0 = 15%, 0->1 = 5%)
    scenarios.append(
        {
            "scenario": "G2_class_conditional",
            "noise_type": "CCLN",
            "noise_rate": None,
            "pos_flip_rate": 0.15,
            "neg_flip_rate": 0.05,
        }
    )

    for sc in scenarios:
        scenario_name = sc["scenario"]
        noise_type = sc["noise_type"]
        rate = sc["noise_rate"]
        pos_rate = sc["pos_flip_rate"]
        neg_rate = sc["neg_flip_rate"]

        txt_lines.append(f"\n=== Scenario: {scenario_name} ({noise_type}) ===")

        if noise_type == "RSLN":
            y_train_noisy = apply_random_symmetric_noise(
                y_train_clean,
                noise_rate=rate,
                rng=rng,
            )
            txt_lines.append(
                f"Random symmetric noise: flip ~{rate*100:.1f}% of all labels."
            )
        else:
            y_train_noisy = apply_class_conditional_noise(
                y_train_clean,
                pos_flip_rate=pos_rate,
                neg_flip_rate=neg_rate,
                rng=rng,
            )
            txt_lines.append(
                f"Class-conditional noise: flip 1->0 ~{pos_rate*100:.1f}%, 0->1 ~{neg_rate*100:.1f}%."
            )

        actual_flip = np.mean(y_train_noisy != y_train_clean)
        txt_lines.append(f"Actual flipped fraction in train set: {actual_flip*100:.2f}%")

        for mname, model in models.items():
            clf = model
            clf.random_state = getattr(clf, "random_state", RANDOM_STATE)

            clf.fit(X_train, y_train_noisy)
            proba = clf.predict_proba(X_test)
            metrics = evaluate_model(y_test, proba)

            row = {
                "scenario": scenario_name,
                "noise_type": noise_type,
                "noise_rate": rate,
                "pos_flip_rate": pos_rate,
                "neg_flip_rate": neg_rate,
                "actual_flip_fraction": actual_flip,
                "model": mname,
            }
            row.update(metrics)
            rows.append(row)

            txt_lines.append(
                f"{mname:15s} | AUC={metrics['roc_auc']:.3f}, "
                f"PR-AUC={metrics['pr_auc']:.3f}, "
                f"Brier={metrics['brier']:.3f}, "
                f"Acc={metrics['accuracy']:.3f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    with OUT_TXT.open("w", encoding="utf-8") as f:
        f.write("WORLD G: Label-noise robustness results\n")
        f.write("\n".join(txt_lines))

    print("\nSaved:")
    print(f"- CSV : {OUT_CSV}")
    print(f"- Text: {OUT_TXT}")


if __name__ == "__main__":
    main()
