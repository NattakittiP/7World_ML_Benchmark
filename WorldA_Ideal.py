"""
model_zoo.py — FINAL VERSION (Fixed for sklearn 1.4+)

Fully working version with:
- BaggingClassifier(estimator=...) fix
- Removed GridSearchCV(param_grid={}) bug
- More robust exception handling
"""

from __future__ import annotations
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    log_loss,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

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


RANDOM_STATE = 42
N_JOBS = -1
CV_FOLDS = 5

TRAIN_PATH = Path("train_arrays.npz")
TEST_PATH = Path("test_arrays.npz")
RESULTS_CSV = Path("model_zoo_results.csv")


# ======================================================================
# Load Train/Test Data
# ======================================================================
def load_npz_data(train_path: Path, test_path: Path):
    if not train_path.exists():
        raise FileNotFoundError(f"Train NPZ not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test NPZ not found: {test_path}")

    train = np.load(train_path)
    test = np.load(test_path)

    return train["X"], train["y"], test["X"], test["y"]


# ======================================================================
# Model Zoo
# ======================================================================
def get_model_zoo() -> Dict[str, Tuple[Any, Dict]]:
    models: Dict[str, Tuple[Any, Dict[str, List[Any]]]] = {}

    # Logistic Regression (L2)
    models["Logistic_L2"] = (
        LogisticRegression(
            solver="lbfgs",
            penalty="l2",
            max_iter=500,
            random_state=RANDOM_STATE,
        ),
        {
            "C": [0.1, 1.0, 10.0],
            "class_weight": [None, "balanced"],
        },
    )

    # Logistic Regression (L1)
    models["Logistic_L1"] = (
        LogisticRegression(
            solver="liblinear",
            penalty="l1",
            max_iter=500,
            random_state=RANDOM_STATE,
        ),
        {
            "C": [0.1, 1.0, 10.0],
            "class_weight": [None, "balanced"],
        },
    )

    # KNN
    models["KNN"] = (
        KNeighborsClassifier(),
        {
            "n_neighbors": [3, 5, 11],
            "weights": ["uniform", "distance"],
        },
    )

    # GaussianNB (NO GRID SEARCH)
    models["GaussianNB"] = (
        GaussianNB(),
        {},   # <-- handle separately
    )

    # Decision Tree
    models["DecisionTree"] = (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {
            "max_depth": [None, 3, 5, 10],
            "min_samples_split": [2, 5, 10],
        },
    )

    # Random Forest
    models["RandomForest"] = (
        RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=N_JOBS
        ),
        {
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5, 10],
            "class_weight": [None, "balanced_subsample"],
        },
    )

    # Extra Trees
    models["ExtraTrees"] = (
        ExtraTreesClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=N_JOBS
        ),
        {
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5, 10],
            "class_weight": [None, "balanced"],
        },
    )

    # Gradient Boosting
    models["GradientBoosting"] = (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3],
        },
    )

    # AdaBoost
    models["AdaBoost"] = (
        AdaBoostClassifier(random_state=RANDOM_STATE),
        {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.05, 0.1, 0.5],
        },
    )

    # Bagging (Decision Tree base)
    models["Bagging_DT"] = (
        BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),  # FIXED
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),
        {
            "n_estimators": [50, 100, 200],
            "max_samples": [0.5, 0.8, 1.0],
        },
    )

    # MLP
    models["MLP"] = (
        MLPClassifier(random_state=RANDOM_STATE, max_iter=500),
        {
            "hidden_layer_sizes": [(32,), (64,), (64, 32)],
            "alpha": [1e-4, 1e-3],
        },
    )

    # SVM (RBF)
    models["SVC_RBF"] = (
        SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        {
            "C": [0.5, 1.0, 5.0],
            "gamma": ["scale", "auto"],
            "class_weight": [None, "balanced"],
        },
    )

    # XGBoost
    if HAS_XGB:
        models["XGBoost"] = (
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                use_label_encoder=False,
                n_jobs=N_JOBS,
            ),
            {
                "n_estimators": [200, 400],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        )

    # LightGBM
    if HAS_LGBM:
        models["LightGBM"] = (
            LGBMClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS),
            {
                "n_estimators": [200, 400],
                "num_leaves": [31, 63],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
            },
        )

    return models


# ======================================================================
# Compute Metrics
# ======================================================================
def compute_metrics(y_true, y_prob, y_pred):
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_prob)),
    }


# ======================================================================
# Run Model Zoo
# ======================================================================
def run_model_zoo(X_train, y_train, X_test, y_test):

    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    results = []
    models = get_model_zoo()

    for name, (estimator, grid) in models.items():
        print(f"\n===== Running model: {name} =====")

        try:
            # -----------------------------------------------------------------
            # Case 1: Hyperparameter tuning (normal case)
            # -----------------------------------------------------------------
            if len(grid) > 0:
                gs = GridSearchCV(
                    estimator=estimator,
                    param_grid=grid,
                    scoring="roc_auc",
                    cv=cv,
                    n_jobs=N_JOBS,
                    refit=True,
                    verbose=0,
                )
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
                best_cv = float(gs.best_score_)
                best_params = gs.best_params_

            # -----------------------------------------------------------------
            # Case 2: NO GRID (GaussianNB)
            # -----------------------------------------------------------------
            else:
                estimator.fit(X_train, y_train)
                best_model = estimator
                best_cv = float("nan")
                best_params = {}

            # Evaluate
            y_prob = best_model.predict_proba(X_test)[:, 1]
            y_pred = best_model.predict(X_test)
            metrics = compute_metrics(y_test, y_prob, y_pred)

            row = {
                "model": name,
                "best_cv_roc_auc": best_cv,
                "best_params": best_params,
            }
            row.update(metrics)
            results.append(row)

        except Exception as e:
            warnings.warn(f"{name} FAILED: {e}")
            continue

    return pd.DataFrame(results)


# ======================================================================
# Main
# ======================================================================
def main():
    X_train, y_train, X_test, y_test = load_npz_data(TRAIN_PATH, TEST_PATH)

    print("Data shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test : {X_test.shape}, y_test : {y_test.shape}")

    df = run_model_zoo(X_train, y_train, X_test, y_test)
    df.sort_values("roc_auc", ascending=False, inplace=True)

    df.to_csv(RESULTS_CSV, index=False)
    print("\n✅ MODEL ZOO FINISHED")
    print(f"Results saved to: {RESULTS_CSV}")
    print(df)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()