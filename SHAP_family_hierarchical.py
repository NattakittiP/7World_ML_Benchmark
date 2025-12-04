# shap_family_hierarchical.py
# Family-level hierarchical SHAP map (features x families)

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

import shap
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

DATA_PATH = Path("Baseline.csv")
OUT_MATRIX_CSV = Path("SHAP_family_matrix.csv")
OUT_FIG = Path("SHAP_family_hierarchical_map.png")

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

# model → family
MODEL_FAMILY = {
    "ExtraTrees": "Tree-based (bagging)",
    "RandomForest": "Tree-based (bagging)",
    "XGBoost": "Boosting",
    "GradientBoosting": "Boosting",
    "LightGBM": "Boosting",
    "SVC_RBF": "Kernel SVM",
    "MLP": "Neural network",
    "GaussianNB": "Bayes",
    "Logistic_L1": "Linear models",
    "Logistic_L2": "Linear models",
}

# Optional XGBoost / LightGBM
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


# ---------------------------------------------------------------------
# DATA & PREPROCESSOR
# ---------------------------------------------------------------------

def load_baseline() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found")
    return pd.read_csv(DATA_PATH)


def create_target(df: pd.DataFrame) -> pd.DataFrame:
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


# ---------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------

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
    """Return SHAP values for positive class as 2D array (n_samples, n_features)."""
    cname = model.__class__.__name__

    # ---------- เลือก explainer ตามประเภทโมเดล ----------
    # Tree-based (RF, ET, GB, XGB, LGBM)
    if cname in [
        "RandomForestClassifier",
        "ExtraTreesClassifier",
        "GradientBoostingClassifier",
        "XGBClassifier",
        "LGBMClassifier",
    ]:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)

    # Linear (Logistic)
    elif cname == "LogisticRegression":
        # ใช้ background เล็ก ๆ ทำ masker
        bg_size = min(200, X.shape[0])
        background = X[:bg_size]
        explainer = shap.LinearExplainer(model, background)  # ไม่ใส่ feature_dependence แล้ว
        sv = explainer.shap_values(X)

    # Kernel / NN / Bayes
    elif cname in ["SVC", "MLPClassifier", "GaussianNB"]:
        bg_size = min(50, X.shape[0])
        background = X[:bg_size]
        explainer = shap.KernelExplainer(model.predict_proba, background)
        sv = explainer.shap_values(X[:200])  # list per class

    else:
        raise NotImplementedError(f"No SHAP explainer for {cname}")

    # ---------- ทำให้เป็น 2D (n_samples, n_features) เสมอ ----------

    # binary-class → sv มักเป็น list [class0, class1]
    if isinstance(sv, list):
        sv = sv[1]  # เอาเฉพาะ class 1

    shap_values = np.array(sv)

    # ถ้าเป็น 3D: (n_samples, n_features, n_outputs) → เอา output สุดท้าย (class 1)
    if shap_values.ndim == 3:
        shap_values = shap_values[..., -1]

    # ถ้าเผลอได้ 1D → บังคับเป็นคอลัมน์เดียว
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(-1, 1)

    return shap_values



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

    X_train = pre.transform(X_train_df)
    X_test = pre.transform(X_test_df)

    feature_names = pre.get_feature_names_out()
    # ตัด prefix "cat__" / "num__"
    feature_names = [name.split("__")[-1] for name in feature_names]

    models = get_models()

    n_feat = X_train.shape[1]
    families = sorted(set(MODEL_FAMILY.values()))
    family_sum = {fam: np.zeros(n_feat, dtype=float) for fam in families}
    family_count = {fam: 0 for fam in families}

    print("Training models and aggregating SHAP by family...")

    for name in FLAGSHIP_MODELS:
        if name not in models:
            print(f"[WARN] {name} unavailable, skipping.")
            continue

        fam = MODEL_FAMILY.get(name)
        if fam is None:
            continue

        clf = models[name]
        print(f"\n=== Model: {name} (Family: {fam}) ===")
        clf.fit(X_train, y_train)

        # subset เพื่อความเร็ว
        n_sample = min(300, X_test.shape[0])
        X_sample = X_test[:n_sample]

        try:
            shap_vals = compute_shap_values(clf, X_sample)
        except Exception as e:
            print(f"[ERROR] SHAP failed for {name}: {e}")
            continue

        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        family_sum[fam] += mean_abs
        family_count[fam] += 1

    # สร้าง matrix feature x family
    data = {}
    for fam in families:
        if family_count[fam] == 0:
            continue
        vec = family_sum[fam] / family_count[fam]
        vec = vec / (vec.sum() + 1e-12)  # normalize
        data[fam] = vec

    mat = pd.DataFrame(data, index=feature_names)
    mat.to_csv(OUT_MATRIX_CSV)
    print(f"\nSaved family SHAP matrix to: {OUT_MATRIX_CSV}")

    # Hierarchical clustermap
    sns.set(context="talk")
    g = sns.clustermap(
        mat,
        cmap="viridis",
        linewidths=0.5,
        figsize=(12, 10),
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Normalized mean |SHAP|"},
    )
    plt.title("Family-level hierarchical SHAP map", pad=100)
    plt.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    print(f"Saved hierarchical SHAP map to: {OUT_FIG}")


if __name__ == "__main__":
    main()
