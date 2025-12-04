"""
compute_E_scores.py

Explainability Stability (E-score) via SHAP
เปรียบเทียบ World A (baseline) vs World F (strong nonlinear)

สิ่งที่สคริปต์นี้ทำ:
1) โหลด Baseline.csv
2) สร้าง target HighTGResponder = 1 ถ้า TG4h >= 75th percentile
3) train/test split แบบ stratified, random_state=42, test_size=0.2
4) สร้าง preprocessor:
   - OneHotEncoder(Sex) + StandardScaler(numerical)
5) Train "flagship models" บน World A:
   - Logistic_L1
   - RandomForest
   - ExtraTrees
   - XGBoost (ถ้ามี)
   - GradientBoosting
   - MLP
   - SVC_RBF
6) คำนวณ SHAP บน:
   - World A (ข้อมูลเดิม)
   - World F (ข้อมูลเดียวกัน แต่ผ่าน strong nonlinear warp)
7) คำนวณ:
   - GEST: Spearman rank corr. ของ mean(|SHAP|) ระหว่าง A vs F
   - LEC: cosine similarity เฉลี่ยระหว่าง SHAP vectors A vs F
   - ADP: Wasserstein distance ของ mean(|SHAP|), แปลงเป็น score 0–1
   - E = 0.5*GEST + 0.3*LEC + 0.2*ADP_norm
8) เซฟเป็น scores_E.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, wasserstein_distance

import shap


# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------

DATA_PATH = Path("Baseline.csv")
OUTPUT_CSV = Path("scores_E.csv")

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

FLAGSHIP_MODELS = [
    "ExtraTrees",
    "RandomForest",
    "XGBoost",
    "GradientBoosting",
    "MLP",
    "SVC_RBF",
    "Logistic_L1",
]

# ลอง import XGBoost ถ้ามี
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# ----------------------------------------------------
# DATA & WORLDS
# ----------------------------------------------------

def load_baseline(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Baseline.csv not found at {path}")
    df = pd.read_csv(path)
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    if "TG4h" not in df.columns:
        raise KeyError("Column 'TG4h' not found in Baseline.csv")
    df = df.copy()
    q75 = df["TG4h"].quantile(0.75)
    df[TARGET_COL] = (df["TG4h"] >= q75).astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    num = StandardScaler()
    pre = ColumnTransformer(
        transformers=[
            ("cat", cat, CATEGORICAL_COLS),
            ("num", num, NUMERICAL_COLS),
        ],
        remainder="drop",
    )
    return pre


# -------- World F strong nonlinear (copy จาก world_f_strong_nonlinear) --------

def strong_nonlinear_g(col: str, x: np.ndarray) -> np.ndarray:
    """
    ฟังก์ชัน g(x) ที่จงใจให้โค้งแรง / U-shape / log-power
    เพื่อทำให้ linear ยากขึ้น
    """
    x = x.astype(float)

    if col == "Age":
        # U-shape รอบอายุ 50
        return ((x - 50.0) ** 2) / 5.0 + 40.0

    if col == "TG0h":
        # log-based power → compress low, complex high
        return (np.log1p(np.clip(x, 0.0, None)) ** 3) * 50.0

    if col == "BMI":
        # cubic around 25
        return ((x - 25.0) ** 3) / 5.0 + 25.0

    if col == "HDL":
        # sqrt warp
        return np.sqrt(np.clip(x, 0.0, None)) * 10.0

    if col == "LDL":
        # log-power
        return (np.log1p(np.clip(x, 0.0, None)) ** 2) * 30.0

    # คอลัมน์อื่นไม่ warp
    return x


def apply_strong_nonlinear_world_inplace(df: pd.DataFrame, alpha: float) -> None:
    """
    x_new = (1 - alpha) * x + alpha * g(x) บนคอลัมน์ที่เลือก
    """
    if alpha <= 0.0:
        return

    cols_to_warp = ["Age", "TG0h", "BMI", "HDL", "LDL"]

    for col in cols_to_warp:
        if col not in df.columns:
            continue
        x = df[col].to_numpy(float)
        gx = strong_nonlinear_g(col, x)
        x_new = (1.0 - alpha) * x + alpha * gx
        df[col] = x_new


# ----------------------------------------------------
# MODEL ZOO (FLAGSHIPS)
# ----------------------------------------------------

def get_flagship_models() -> Dict[str, any]:
    models = {}

    models["Logistic_L1"] = LogisticRegression(
        max_iter=500,
        solver="liblinear",
        penalty="l1",
        random_state=RANDOM_STATE,
        C=1.0,
        class_weight="balanced",
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

    return models


# ----------------------------------------------------
# SHAP UTILITIES
# ----------------------------------------------------

def compute_shap_values(model, X: np.ndarray) -> np.ndarray:
    """
    เลือก SHAP explainer ตามชนิด model แบบอัตโนมัติ
    """
    cname = model.__class__.__name__

    if cname in ["RandomForestClassifier", "ExtraTreesClassifier",
                 "GradientBoostingClassifier"]:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        # class index 1
        sv = sv[1] if isinstance(sv, list) else sv

    elif cname in ["XGBClassifier"]:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        sv = sv[1] if isinstance(sv, list) else sv

    elif cname in ["LogisticRegression"]:
        explainer = shap.LinearExplainer(model, X, feature_dependence="independent")
        sv = explainer.shap_values(X)

    elif cname in ["SVC", "MLPClassifier"]:
        # ใช้ KernelExplainer (ช้า) แต่เราจะลด sample ลง
        background = X[:50]
        explainer = shap.KernelExplainer(model.predict_proba, background)
        # เราจะ compute เฉพาะ subset 200 ตัว
        sv = explainer.shap_values(X[:200])[1]
        # เพื่อความสม่ำเสมอ เราจะ return เฉพาะ 200 ตัวแรก
        return np.array(sv)

    else:
        raise NotImplementedError(f"SHAP explainer not implemented for {cname}")

    return np.array(sv)


# ----------------------------------------------------
# E-SCORE COMPUTATION
# ----------------------------------------------------

def compute_E_for_model(
    model_name: str,
    model,
    X_test_A: np.ndarray,
    X_test_F: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    คำนวณ GEST, LEC, ADP, และ E สำหรับโมเดลเดียว
    """

    # เพื่อประหยัดเวลา: sample จาก test 300 ตัว → ใช้ 200 ตัวแรกสำหรับ SHAP
    n_sample = min(200, X_test_A.shape[0])
    XA = X_test_A[:n_sample]
    XF = X_test_F[:n_sample]

    shap_A = compute_shap_values(model, XA)
    shap_F = compute_shap_values(model, XF)

    # เผื่อกรณี KernelExplainer ของ SVC/MLP ทำเฉพาะ 200 ตัวแล้ว
    nA = shap_A.shape[0]
    nF = shap_F.shape[0]
    n = min(nA, nF)
    shap_A = shap_A[:n]
    shap_F = shap_F[:n]

    # 1) GEST: Spearman corr. ของ mean |SHAP|
    mean_A = np.mean(np.abs(shap_A), axis=0)
    mean_F = np.mean(np.abs(shap_F), axis=0)

    # rank โดยใช้ argsort (ค่าเล็กสุด rank ต่ำสุด)
    rank_A = np.argsort(mean_A)
    rank_F = np.argsort(mean_F)
    rho, _ = spearmanr(rank_A, rank_F)
    GEST = float(rho)

    # 2) LEC: cosine similarity เฉลี่ยของ local SHAP vectors
    cos_mat = cosine_similarity(shap_A, shap_F)
    LEC = float(np.mean(cos_mat))

    # 3) ADP: Wasserstein distance ของ mean(|SHAP|)
    adp = wasserstein_distance(mean_A, mean_F)
    # ตอน normalize จะทำภายหลังเมื่อรวมทุกโมเดล

    return GEST, LEC, adp


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------

def main():
    print("Loading baseline data...")
    df = load_baseline(DATA_PATH)
    df = create_target(df)

    # train/test split
    X = df[CATEGORICAL_COLS + NUMERICAL_COLS]
    y = df[TARGET_COL].to_numpy()

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # สร้าง preprocessor และ transform World A
    pre = build_preprocessor()
    pre.fit(X_train_df)

    X_train_A = pre.transform(X_train_df)
    X_test_A = pre.transform(X_test_df)

    # สร้าง World F: เอา test_df มาวาร์ปแบบ nonlinear แล้ว transform
    X_test_F_df = X_test_df.copy()
    apply_strong_nonlinear_world_inplace(X_test_F_df, alpha=1.0)
    X_test_F = pre.transform(X_test_F_df)

    models = get_flagship_models()

    results = []
    adp_raw_list = []

    print("Training models on World A and computing SHAP-based stability (World A vs F)...")

    # ฝึกและคำนวณ GEST, LEC, ADP ต่อโมเดล
    for name in FLAGSHIP_MODELS:
        if name not in models:
            print(f"[WARN] Model {name} is not available (e.g. XGBoost not installed). Skipping.")
            continue

        clf = models[name]
        print(f"\n=== Model: {name} ===")
        clf.fit(X_train_A, y_train)

        try:
            GEST, LEC, adp = compute_E_for_model(name, clf, X_test_A, X_test_F)
            results.append((name, GEST, LEC, adp))
            adp_raw_list.append(adp)
            print(f"GEST={GEST:.3f}, LEC={LEC:.3f}, ADP_raw={adp:.4f}")
        except Exception as e:
            print(f"[ERROR] SHAP computation failed for {name}: {e}")

    if not results:
        raise RuntimeError("No E-scores computed; all models failed SHAP step.")

    # Normalize ADP across models → ADP_norm (0-1, ยิ่งสูงแปลว่า drift น้อย)
    adp_raw_arr = np.array(adp_raw_list)
    adp_max = adp_raw_arr.max() if adp_raw_arr.size > 0 else 1.0

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

    df_scores.to_csv(OUTPUT_CSV, index=False)
    print("\n=== Explainability Stability Scores (A vs F) ===")
    print(df_scores)
    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
