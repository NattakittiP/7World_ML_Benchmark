"""
shap_visualizations.py

ระดับ: ศาสตราจารย์ลงมาทำเอง

สิ่งที่สคริปต์นี้ทำ:

1) โหลด Baseline.csv
2) สร้าง target HighTGResponder = 1 ถ้า TG4h >= 75th percentile
3) train/test split (stratified) + preprocessing:
   - OneHotEncoder สำหรับ Sex
   - StandardScaler สำหรับ numerical features
4) เทรนโมเดลหลัก (World A):
   - ExtraTrees
   - RandomForest
   - GradientBoosting
   - XGBoost (ถ้ามี)
   - Logistic_L1
   - MLP
   - SVC_RBF

5) สร้าง World F (strong nonlinear physiology) บน test set
6) สำหรับแต่ละโมเดล:
   - คำนวณ SHAP บน World A (X_test_A)
   - คำนวณ SHAP บน World F (X_test_F)
   - สร้าง plot:

   6.1 Global importance (mean |SHAP|):
       - summary_plot (bar) World A
       - summary_plot (bar) World F
       - barplot เปรียบเทียบ mean |SHAP| (A vs F) ต่อ feature

   6.2 Beeswarm plot:
       - summary_plot (dot) World A
       - summary_plot (dot) World F

   6.3 Dependence plot (feature-level):
       - TG0h, BMI, WBV สำหรับ World A
       - TG0h, BMI, WBV สำหรับ World F

7) เซฟทุกภาพเข้าโฟลเดอร์:
   plots_SHAP/<model_name>/...

หมายเหตุ:
- SHAP สำหรับ MLP, SVC_RBF ใช้ KernelExplainer → อาจช้า
- จำกัด sample สำหรับ SHAP เพื่อให้รันได้จริงใน Colab
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------

DATA_PATH = Path("Baseline.csv")
OUTPUT_DIR = Path("plots_SHAP")

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

# จำกัดจำนวน sample สำหรับ SHAP
MAX_SAMPLES_TREE = 300   # tree-based, logistic
MAX_SAMPLES_KERNEL = 150 # MLP, SVC (KernelExplainer)


# XGBoost (optional)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# ----------------------------------------------------
# UTILITIES
# ----------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_and_prepare_baseline() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, list, pd.DataFrame]:
    """
    โหลด Baseline.csv, สร้าง target, และ preprocess
    คืนค่า:
    - X_train, X_test (หลัง preprocess)
    - y_train, y_test
    - preprocessor (ColumnTransformer)
    - feature_names (list ของชื่อฟีเจอร์หลัง encode)
    - X_test_df_raw (DataFrame ของฟีเจอร์ดิบใน test set, ใช้สำหรับ world F)
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Baseline.csv not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if "TG4h" not in df.columns:
        raise KeyError("Column 'TG4h' not found in Baseline.csv")

    # สร้าง target
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

    # Preprocessor
    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    num = StandardScaler()

    pre = ColumnTransformer(
        transformers=[
            ("cat", cat, CATEGORICAL_COLS),
            ("num", num, NUMERICAL_COLS),
        ],
        remainder="drop",
    )

    pre.fit(X_train_df)

    X_train = pre.transform(X_train_df)
    X_test = pre.transform(X_test_df)

    # ฟีเจอร์เนมหลัง encode
    cat_names = pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_COLS).tolist()
    feature_names = cat_names + NUMERICAL_COLS

    return X_train, X_test, y_train, y_test, pre, feature_names, X_test_df


# ------- Strong Nonlinear World (World F) -------

def strong_nonlinear_g(col: str, x: np.ndarray) -> np.ndarray:
    """
    ฟังก์ชัน g(x) สำหรับสร้าง nonlinear warp ให้ยากขึ้นสำหรับ linear models
    """
    x = x.astype(float)

    if col == "Age":
        # U-shape รอบอายุ 50
        return ((x - 50.0) ** 2) / 5.0 + 40.0

    if col == "TG0h":
        # log-power
        return (np.log1p(np.clip(x, 0.0, None)) ** 3) * 50.0

    if col == "BMI":
        # cubic around 25
        return ((x - 25.0) ** 3) / 5.0 + 25.0

    if col == "HDL":
        return np.sqrt(np.clip(x, 0.0, None)) * 10.0

    if col == "LDL":
        return (np.log1p(np.clip(x, 0.0, None)) ** 2) * 30.0

    return x


def apply_strong_nonlinear_world_inplace(df: pd.DataFrame, alpha: float = 1.0) -> None:
    """
    ปรับค่าใน df ให้เป็น world F:
    x_new = (1-alpha)*x + alpha*g(x)

    ทำ inplace บน df
    """
    if alpha <= 0.0:
        return

    cols_to_warp = ["Age", "TG0h", "BMI", "HDL", "LDL"]

    for col in cols_to_warp:
        if col not in df.columns:
            continue
        x = df[col].to_numpy(float)
        gx = strong_nonlinear_g(col, x)
        df[col] = (1.0 - alpha) * x + alpha * gx


# ------- Model Zoo for SHAP Visualization -------

def get_models_for_shap() -> Dict[str, object]:
    """
    สร้างชุดโมเดลที่เราจะใช้สำหรับ SHAP visualization
    """
    models = {}

    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=2,
        random_state=RANDOM_STATE,
    )

    models["RandomForest"] = RandomForestClassifier(
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

    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )

    models["Logistic_L1"] = LogisticRegression(
        max_iter=500,
        solver="liblinear",
        penalty="l1",
        C=1.0,
        class_weight="balanced",
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


# ------- SHAP Explainer Selection -------

def compute_shap_values(
    model,
    X: np.ndarray,
    model_name: str,
    max_samples: int,
) -> np.ndarray:
    """
    เลือก explainer ตามประเภทของโมเดล แล้วคำนวณ SHAP values บน X[:max_samples]

    คืนค่า:
    - shap_values: array (n_samples, n_features) สำหรับ class 1
    """
    cname = model.__class__.__name__

    # จำกัดจำนวนตัวอย่าง
    n = min(max_samples, X.shape[0])
    X_eval = X[:n]

    # ----- เลือก explainer ตามประเภทโมเดล -----

    # Tree-based
    if cname in ["ExtraTreesClassifier", "RandomForestClassifier", "GradientBoostingClassifier"]:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_eval)

    # XGBoost
    elif cname == "XGBClassifier":
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_eval)

    # Logistic Regression  (ใช้ค่า default = interventional)
    elif cname == "LogisticRegression":
        # ใช้ background เล็ก ๆ สำหรับ masker ให้ stable ขึ้น
        bg_size = min(200, X.shape[0])
        background = X[:bg_size]
        explainer = shap.LinearExplainer(model, background)  # ไม่ส่ง feature_perturbation แล้ว
        sv = explainer.shap_values(X_eval)

    # Kernel-based (MLP, SVC)
    elif cname in ["SVC", "MLPClassifier"]:
        bg_size = min(50, X.shape[0])
        background = X[:bg_size]
        explainer = shap.KernelExplainer(model.predict_proba, background)
        sv = explainer.shap_values(X_eval)  # list per class

    else:
        raise NotImplementedError(f"SHAP explainer not implemented for model type: {cname}")

    # ----- ทำให้เป็น array 2D (n_samples, n_features) เสมอ -----

    # binary-class → sv มักเป็น list [class0, class1]
    if isinstance(sv, list):
        sv = sv[1]   # เอา class 1 ตามที่เราใช้คำนวณทุก metric

    shap_values = np.array(sv)

    # ถ้าเป็น 3D: (n_samples, n_features, n_outputs) → เอา output สุดท้าย (class 1)
    if shap_values.ndim == 3:
        shap_values = shap_values[..., -1]

    # ถ้าเผลอเป็น 1D → บังคับให้เป็นคอลัมน์เดียว
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(-1, 1)

    return shap_values, X_eval





# ------- Plot Helpers -------

def summary_bar_plot(
    shap_values: np.ndarray,
    X_eval: np.ndarray,
    feature_names: list,
    out_path: Path,
    title: str,
):
    shap.summary_plot(
        shap_values,
        X_eval,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    fig = plt.gcf()
    plt.title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def summary_beeswarm_plot(
    shap_values: np.ndarray,
    X_eval: np.ndarray,
    feature_names: list,
    out_path: Path,
    title: str,
):
    shap.summary_plot(
        shap_values,
        X_eval,
        feature_names=feature_names,
        plot_type="dot",
        show=False,
    )
    fig = plt.gcf()
    plt.title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def dependence_plot(
    feature_name: str,
    shap_values: np.ndarray,
    X_eval: np.ndarray,
    feature_names: list,
    out_path: Path,
    title: str,
):
    shap.dependence_plot(
        feature_name,
        shap_values,
        X_eval,
        feature_names=feature_names,
        show=False,
    )
    fig = plt.gcf()
    plt.title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def compare_mean_abs_shap_bar(
    shap_A: np.ndarray,
    shap_F: np.ndarray,
    feature_names: list,
    out_path: Path,
    title: str,
):
    mean_A = np.mean(np.abs(shap_A), axis=0)
    mean_F = np.mean(np.abs(shap_F), axis=0)

    x = np.arange(len(feature_names))

    width = 0.38
    plt.figure(figsize=(max(8, len(feature_names) * 0.4), 4))
    plt.bar(x - width/2, mean_A, width, label="World A")
    plt.bar(x + width/2, mean_F, width, label="World F")

    plt.xticks(x, feature_names, rotation=60, ha="right", fontsize=8)
    plt.ylabel("Mean |SHAP|")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------

def main():
    ensure_dir(OUTPUT_DIR)

    print("Loading and preprocessing Baseline.csv ...")
    X_train_A, X_test_A, y_train, y_test, pre, feature_names, X_test_raw_df = load_and_prepare_baseline()

    # สร้าง World F บน test set (raw), แล้ว transform ด้วย preprocessor เดิม
    X_test_F_df = X_test_raw_df.copy()
    apply_strong_nonlinear_world_inplace(X_test_F_df, alpha=1.0)
    X_test_F = pre.transform(X_test_F_df)

    models = get_models_for_shap()

    shap.initjs()  # สำหรับ notebook; ใน script ไม่ critical แต่ไม่เป็นไร

    for model_name, model in models.items():
        print(f"\n=== Training & SHAP visualization for model: {model_name} ===")

        # เทรนบน World A
        model.fit(X_train_A, y_train)

        # เลือก max_samples ตามประเภทโมเดล
        if model_name in ["MLP", "SVC_RBF"]:
            max_samples = MAX_SAMPLES_KERNEL
        else:
            max_samples = MAX_SAMPLES_TREE

        # SHAP บน World A
        print(f"  Computing SHAP on World A (baseline) ...")
        shap_A, X_eval_A = compute_shap_values(
            model,
            X_test_A,
            model_name=model_name,
            max_samples=max_samples,
        )

        # SHAP บน World F
        print(f"  Computing SHAP on World F (strong nonlinear) ...")
        shap_F, X_eval_F = compute_shap_values(
            model,
            X_test_F,
            model_name=model_name,
            max_samples=max_samples,
        )

        # โฟลเดอร์ย่อยของโมเดล
        model_dir = OUTPUT_DIR / model_name
        ensure_dir(model_dir)

        # ---------- Global bar plots ----------
        summary_bar_plot(
            shap_A,
            X_eval_A,
            feature_names,
            model_dir / f"{model_name}_WorldA_bar.png",
            title=f"{model_name} - Global SHAP importance (World A)",
        )

        summary_bar_plot(
            shap_F,
            X_eval_F,
            feature_names,
            model_dir / f"{model_name}_WorldF_bar.png",
            title=f"{model_name} - Global SHAP importance (World F)",
        )

        compare_mean_abs_shap_bar(
            shap_A,
            shap_F,
            feature_names,
            model_dir / f"{model_name}_A_vs_F_mean_abs_shap.png",
            title=f"{model_name} - Mean |SHAP|: World A vs World F",
        )

        # ---------- Beeswarm plots ----------
        summary_beeswarm_plot(
            shap_A,
            X_eval_A,
            feature_names,
            model_dir / f"{model_name}_WorldA_beeswarm.png",
            title=f"{model_name} - SHAP beeswarm (World A)",
        )

        summary_beeswarm_plot(
            shap_F,
            X_eval_F,
            feature_names,
            model_dir / f"{model_name}_WorldF_beeswarm.png",
            title=f"{model_name} - SHAP beeswarm (World F)",
        )

        # ---------- Dependence plots ----------
        # ฟีเจอร์ที่สำคัญเชิงสรีรวิทยา
        key_feats = ["TG0h", "BMI", "WBV"]

        for feat in key_feats:
            if feat not in feature_names:
                continue

            # World A
            dependence_plot(
                feat,
                shap_A,
                X_eval_A,
                feature_names,
                model_dir / f"{model_name}_WorldA_dependence_{feat}.png",
                title=f"{model_name} - SHAP dependence on {feat} (World A)",
            )

            # World F
            dependence_plot(
                feat,
                shap_F,
                X_eval_F,
                feature_names,
                model_dir / f"{model_name}_WorldF_dependence_{feat}.png",
                title=f"{model_name} - SHAP dependence on {feat} (World F)",
            )

        print(f"  -> Plots saved in: {model_dir}")

    print(f"\nAll SHAP plots generated under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()