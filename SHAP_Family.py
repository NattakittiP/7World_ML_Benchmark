# shap_family_aggregation.py
# Aggregated SHAP feature importance at model-family level

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

import shap

# Optional models
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except Exception:
    HAS_LGB = False

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

DATA_PATH = Path("Baseline.csv")
OUT_DIR = Path("SHAP_family")

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "HighTGResponder"

CATEGORICAL_COLS = ["Sex"]
NUMERICAL_COLS = [
    "Age", "Hematocrit", "TotalProtein", "WBV",
    "TG0h", "HDL", "LDL", "BMI",
]

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


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# DATA
# -------------------------------------------------------------------

def load_baseline() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found")
    df = pd.read_csv(DATA_PATH)
    if "TG4h" not in df.columns:
        raise KeyError("TG4h not found in Baseline.csv")
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


def get_feature_names(pre: ColumnTransformer):
    cat_names = pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_COLS)
    num_names = np.array(NUMERICAL_COLS)
    return np.concatenate([cat_names, num_names])


# -------------------------------------------------------------------
# MODELS
# -------------------------------------------------------------------

def get_models():
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
        solver="liblinear",
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=RANDOM_STATE,
    )

    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=10,
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
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
        )

    return models


# -------------------------------------------------------------------
# FAMILY MAPPING
# -------------------------------------------------------------------

def assign_family(model_name: str) -> str:
    if model_name in ["ExtraTrees", "RandomForest"]:
        return "Tree-based (bagging)"
    if model_name in ["XGBoost", "GradientBoosting", "LightGBM"]:
        return "Boosting"
    if model_name in ["Logistic_L1", "Logistic_L2"]:
        return "Linear models"
    if model_name == "SVC_RBF":
        return "Kernel SVM"
    if model_name == "MLP":
        return "Neural network"
    if model_name == "GaussianNB":
        return "Bayes"
    return "Other"


# -------------------------------------------------------------------
# SHAP
# -------------------------------------------------------------------

def compute_shap_values(model, X, model_name: str) -> np.ndarray:
    cname = model.__class__.__name__

    # ------- เลือก explainer ตามประเภทโมเดล -------
    if cname in [
        "RandomForestClassifier",
        "ExtraTreesClassifier",
        "GradientBoostingClassifier",
        "XGBClassifier",
        "LGBMClassifier",
    ]:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)

    elif cname == "LogisticRegression":
        # ใช้ background เล็ก ๆ เป็น masker
        bg_size = min(200, X.shape[0])
        background = X[:bg_size]
        explainer = shap.LinearExplainer(model, background)
        sv = explainer.shap_values(X)

    elif cname in ["SVC", "MLPClassifier", "GaussianNB"]:
        background = X[:50]
        explainer = shap.KernelExplainer(model.predict_proba, background)
        sv = explainer.shap_values(X[:200])  # list per class

    else:
        raise NotImplementedError(f"No SHAP explainer for {cname}")

    # ------- ทำให้เป็น array 2D (n_samples, n_features) เสมอ -------

    # binary-class → sv มักจะเป็น list [class0, class1]
    if isinstance(sv, list):
        sv = sv[1]   # ใช้ class 1 ตามที่คุณใช้ทุกที่

    shap_values = np.array(sv)

    # ถ้าเป็น 3D: (n_samples, n_features, n_outputs) → เอา output สุดท้าย (class 1)
    if shap_values.ndim == 3:
        shap_values = shap_values[..., -1]

    # ถ้าเผลอออกมาเป็น 1D → บังคับเป็นคอลัมน์เดียว
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(-1, 1)

    return shap_values



# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    ensure_dir(OUT_DIR)

    print("Loading baseline data...")
    df = load_baseline()
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

    X_train = pre.transform(X_train_df)
    X_test = pre.transform(X_test_df)

    feature_names = get_feature_names(pre)

    models = get_models()
    rows = []

    print("Training models + computing mean|SHAP| per feature...")
    for name in FLAGSHIP_MODELS:
        if name not in models:
            print(f"[WARN] {name} not available, skip.")
            continue

        clf = models[name]
        print(f"\n=== {name} ===")
        clf.fit(X_train, y_train)

        # ใช้ sample subset เพื่อประหยัดเวลา
        n_sample = min(300, X_test.shape[0])
        X_sample = X_test[:n_sample]

        try:
            shap_vals = compute_shap_values(clf, X_sample, name)
        except Exception as e:
            print(f"[ERROR] SHAP failed for {name}: {e}")
            continue

        mean_abs = np.mean(np.abs(shap_vals), axis=0)

        for fname, val in zip(feature_names, mean_abs):
            rows.append(
                {
                    "model": name,
                    "family": assign_family(name),
                    "feature": fname,
                    "mean_abs_shap": float(val),
                }
            )

    df_model = pd.DataFrame(rows)
    model_csv = OUT_DIR / "shap_feature_importance_by_model.csv"
    df_model.to_csv(model_csv, index=False)
    print("\nSaved:", model_csv)

    # ------------------ Family aggregation ------------------
    df_family = (
        df_model
        .groupby(["family", "feature"], as_index=False)["mean_abs_shap"]
        .mean()
    )

    # normalize ภายในแต่ละ family ให้ sum = 1
    df_family["family_total"] = df_family.groupby("family")["mean_abs_shap"].transform("sum")
    df_family["importance_norm"] = df_family["mean_abs_shap"] / df_family["family_total"]

    fam_csv = OUT_DIR / "shap_family_importance.csv"
    df_family.to_csv(fam_csv, index=False)
    print("Saved:", fam_csv)

    # ------------------ Plot top-k features per family ------------------
    top_k = 5
    for fam in sorted(df_family["family"].unique()):
        sub = df_family[df_family["family"] == fam].sort_values(
            "importance_norm", ascending=False
        ).head(top_k)

        plt.figure(figsize=(6, 4))
        x = np.arange(len(sub))
        vals = sub["importance_norm"].to_numpy()
        plt.bar(x, vals)
        plt.xticks(x, sub["feature"], rotation=30, ha="right")
        plt.ylabel("Normalized mean |SHAP|")
        plt.title(f"Top-{top_k} features for {fam}")
        plt.tight_layout()

        out_png = OUT_DIR / f"SHAP_family_{fam.replace(' ', '_').replace('(', '').replace(')', '')}_top{top_k}.png"
        plt.savefig(out_png, dpi=300)
        plt.close()
        print("Saved:", out_png)


if __name__ == "__main__":
    main()
