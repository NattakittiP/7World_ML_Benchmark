# compute_E_All_score.py
# E-score for all flagship models (World A vs F)
# ExtraTrees, RandomForest, GradientBoosting, XGBoost, LightGBM,
# MLP, SVC_RBF, GaussianNB, Logistic_L1, Logistic_L2

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, wasserstein_distance

import shap


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

DATA_PATH = Path("Baseline.csv")
OUT_CSV = Path("scores_E_all.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "HighTGResponder"

CATEGORICAL_COLS = ["Sex"]
NUMERICAL_COLS = [
    "Age", "Hematocrit", "TotalProtein", "WBV",
    "TG0h", "HDL", "LDL", "BMI",
]

# โมเดลหลักที่เราต้องการ E-score ให้ครบ 10 ตัว
FLAGSHIP_MODELS = [
    "ExtraTrees",
    "RandomForest",
    "GradientBoosting",
    "XGBoost",
    "LightGBM",
    "MLP",
    "SVC_RBF",
    "GaussianNB",
    "Logistic_L1",
    "Logistic_L2",
]

# XGBoost optional
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# LightGBM optional
try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except Exception:
    HAS_LGB = False


# ---------------------------------------------------------------------
# DATA & WORLD F
# ---------------------------------------------------------------------

def load_baseline() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found")
    return pd.read_csv(DATA_PATH)


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary target HighTGResponder from TG4h (>= 75th percentile)."""
    df = df.copy()
    if "TG4h" not in df.columns:
        raise KeyError("Column 'TG4h' missing in Baseline.csv")
    q75 = df["TG4h"].quantile(0.75)
    df[TARGET_COL] = (df["TG4h"] >= q75).astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    num = StandardScaler()
    pre = ColumnTransformer(
        [
            ("cat", cat, CATEGORICAL_COLS),
            ("num", num, NUMERICAL_COLS),
        ],
        remainder="drop",
    )
    return pre


# ฟังก์ชัน non-linear เหมือน World F เดิม
def strong_nonlinear_g(col: str, x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    if col == "Age":
        return ((x - 50.0) ** 2) / 5.0 + 40.0
    if col == "TG0h":
        return (np.log1p(np.clip(x, 0.0, None)) ** 3) * 50.0
    if col == "BMI":
        return ((x - 25.0) ** 3) / 5.0 + 25.0
    if col == "HDL":
        return np.sqrt(np.clip(x, 0.0, None)) * 10.0
    if col == "LDL":
        return (np.log1p(np.clip(x, 0.0, None)) ** 2) * 30.0
    return x


def apply_strong_nonlinear_world(df: pd.DataFrame, alpha: float) -> None:
    """Apply strong non-linear distortion (World F) in-place."""
    if alpha <= 0.0:
        return
    cols = ["Age", "TG0h", "BMI", "HDL", "LDL"]
    for col in cols:
        if col not in df.columns:
            continue
        x = df[col].to_numpy(float)
        gx = strong_nonlinear_g(col, x)
        df[col] = (1.0 - alpha) * x + alpha * gx


# ---------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------

def get_models():
    models = {}

    # Logistic L1
    models["Logistic_L1"] = LogisticRegression(
        max_iter=500,
        solver="liblinear",
        penalty="l1",
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    # Logistic L2
    models["Logistic_L2"] = LogisticRegression(
        max_iter=500,
        solver="liblinear",
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=2,
        random_state=RANDOM_STATE,
    )

    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=2,
        random_state=RANDOM_STATE,
    )

    models["GradientBoosting"] = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
    )

    models["GaussianNB"] = GaussianNB()

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

    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )

    if HAS_LGB:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
        )

    return models


# ---------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------

def compute_shap_values(model, X: np.ndarray) -> np.ndarray:
    """
    Return SHAP values for positive class (shape: [n_samples, n_features])
    ใช้ KernelExplainer สำหรับทุกโมเดล เพื่อความเสถียร
    """
    # จำกัดจำนวนตัวอย่างเพื่อให้เร็วขึ้น
    n_sample = min(200, X.shape[0])
    X_eval = X[:n_sample]

    # background ใช้ ~50 ตัวอย่างแรก
    n_bg = min(50, X.shape[0])
    background = X[:n_bg]

    # ให้ฟังก์ชันคืน probability ของ positive class
    def f(x):
        proba = model.predict_proba(x)
        # proba shape: (n_samples, 2) → เอา column ของ class=1
        return proba[:, 1]

    explainer = shap.KernelExplainer(f, background)
    shap_vals = explainer.shap_values(X_eval)  # binary: ได้ array [n_sample, n_features]

    # เผื่อบางเวอร์ชัน shap คืน list (แต่ปกติ KernelExplainer + scalar output → ไม่ list)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]

    return np.array(shap_vals)



def compute_E_for_model(model_name, model, X_test_A, X_test_F):
    """Compute GEST, LEC, ADP (raw) for one model."""
    n_sample = min(200, X_test_A.shape[0])
    XA = X_test_A[:n_sample]
    XF = X_test_F[:n_sample]

    shap_A = compute_shap_values(model, XA)
    shap_F = compute_shap_values(model, XF)

    n = min(shap_A.shape[0], shap_F.shape[0])
    shap_A = shap_A[:n]
    shap_F = shap_F[:n]

    # GEST
    mean_A = np.mean(np.abs(shap_A), axis=0)
    mean_F = np.mean(np.abs(shap_F), axis=0)
    rank_A = np.argsort(mean_A)
    rank_F = np.argsort(mean_F)
    rho, _ = spearmanr(rank_A, rank_F)
    GEST = float(rho)

    # LEC
    cos_mat = cosine_similarity(shap_A, shap_F)
    LEC = float(np.mean(cos_mat))

    # ADP_raw
    adp = float(wasserstein_distance(mean_A, mean_F))

    return GEST, LEC, adp



# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print("Loading Baseline...")
    df = load_baseline()
    df = create_target(df)

    X = df[CATEGORICAL_COLS + NUMERICAL_COLS]
    y = df[TARGET_COL].to_numpy()

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    pre = build_preprocessor()
    pre.fit(X_train_df)

    # World A
    X_train_A = pre.transform(X_train_df)
    X_test_A = pre.transform(X_test_df)

    # World F (strong nonlinear)
    X_test_F_df = X_test_df.copy()
    apply_strong_nonlinear_world(X_test_F_df, alpha=1.0)
    X_test_F = pre.transform(X_test_F_df)

    models = get_models()

    results = []
    adps = []

    print("Training models on World A and computing SHAP-based E-score (A vs F)...")

    for name in FLAGSHIP_MODELS:
        if name not in models:
            print(f"[WARN] {name} not available (e.g., XGBoost/LightGBM missing). Skipping.")
            continue

        clf = models[name]
        print(f"\n=== Model: {name} ===")
        clf.fit(X_train_A, y_train)

        try:
            GEST, LEC, adp = compute_E_for_model(name, clf, X_test_A, X_test_F)
            print(f"GEST={GEST:.3f}, LEC={LEC:.3f}, ADP_raw={adp:.4f}")
            results.append((name, GEST, LEC, adp))
            adps.append(adp)
        except Exception as e:
            print(f"[ERROR] SHAP failed for {name}: {e}")

    if not results:
        raise RuntimeError("No E-scores computed.")

    adp_arr = np.array(adps)
    adp_max = adp_arr.max() if adp_arr.size > 0 else 1.0

    rows = []
    for (name, GEST, LEC, adp) in results:
        ADP_norm = 1.0 - (adp / adp_max if adp_max > 0 else 0.0)
        E = 0.5 * GEST + 0.3 * LEC + 0.2 * ADP_norm
        rows.append(
            {
                "model": name,
                "GEST": GEST,
                "LEC": LEC,
                "ADP_norm": ADP_norm,
                "E": E,
            }
        )

    df_scores = pd.DataFrame(rows)
    df_scores = df_scores.sort_values("E", ascending=False)
    df_scores.to_csv(OUT_CSV, index=False)

    print("\n=== E-scores (World A vs F) ===")
    print(df_scores)
    print(f"\nSaved to: {OUT_CSV}")


if __name__ == "__main__":
    main()
