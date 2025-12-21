# ============================================================
# HOSPITAL MORTALITY RSCE MINI-BENCHMARK
# (โครงสร้างเหมือน NHANES RSCE code ของคุณ)
# RSCE = 0.4R + 0.3S + 0.2C + 0.1E
# ============================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.impute import SimpleImputer

import shap
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ------------------------
# 1) LOAD DATA + TARGET & FEATURES
# ------------------------
DATA_PATH = "analytic_dataset_mortality_all_admissions.csv"
df = pd.read_csv(DATA_PATH)

# label เป้าหมาย: in-hospital mortality (0/1)
TARGET_COL = "label_mortality"
assert TARGET_COL in df.columns, "label_mortality not found in dataset."
df = df.dropna(subset=[TARGET_COL]).copy()
df[TARGET_COL] = df[TARGET_COL].astype(int)

# ตัดคอลัมน์ที่ไม่ควรเป็น feature (ID + leakage)
drop_cols = []
for c in ["label_mortality", "hadm_id", "subject_id",
          "discharge_location", "anchor_year", "anchor_year_group"]:
    if c in df.columns:
        drop_cols.append(c)

X_full = df.drop(columns=drop_cols)
y = df[TARGET_COL].values

# แยก numeric / categorical (ให้ใกล้เคียง NHANES style)
NUM_COLS = [c for c in X_full.columns if pd.api.types.is_numeric_dtype(X_full[c])]
CAT_COLS = [c for c in X_full.columns if c not in NUM_COLS]

print("Numeric cols:", NUM_COLS)
print("Categorical cols:", CAT_COLS)

X = X_full[NUM_COLS + CAT_COLS]

# ------------------------
# 2) TRAIN / TEST SPLIT
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

# ------------------------
# 3) PREPROCESSOR (แก้เวอร์ชันนี้แทนของเดิม)
# ------------------------
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),   # <--- เติมอันนี้เข้าไป
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),  # <--- เติมอันนี้ด้วย
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUM_COLS),
        ("cat", categorical_transformer, CAT_COLS)
    ],
    remainder="drop"
)

def make_pipeline(clf):
    return Pipeline([("preprocess", preprocess), ("clf", clf)])

PIPELINES = {name: make_pipeline(model) for name, model in MODELS.items()}

# ------------------------
# 4) MODEL ZOO (แบบเดียวกับ NHANES)
# ------------------------
MODELS = {
    "Logistic_L2": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=RANDOM_STATE),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=400, learning_rate=0.05, random_state=RANDOM_STATE),
    "SVC_RBF": SVC(kernel="rbf", probability=True, C=3.0, random_state=RANDOM_STATE),
    "MLP": MLPClassifier(hidden_layer_sizes=(128,64), max_iter=1000, random_state=RANDOM_STATE),
    "GaussianNB": GaussianNB()
}

def make_pipeline(clf):
    return Pipeline([("preprocess", preprocess), ("clf", clf)])

PIPELINES = {name: make_pipeline(model) for name, model in MODELS.items()}

# ------------------------
# 5) WORLD PERTURBATIONS (WA_real – WG_real)
#     โครงคล้าย NHANES: noise, missingness, shift,
#     surrogate corruption, nonlinear, label noise
# ------------------------
def add_gaussian_noise(X_df, std_fraction=0.20):
    Xn = X_df.copy()
    for col in NUM_COLS:
        std = X_df[col].std()
        noise = np.random.normal(0, std_fraction * (std if std > 0 else 1.0), len(X_df))
        Xn[col] = X_df[col] + noise
    return Xn

def add_outliers(X_df, frac=0.02):
    X_out = X_df.copy()
    n = len(X_out)
    for col in NUM_COLS:
        idx = np.random.choice(n, max(1, int(frac * n)), replace=False)
        std = X_out[col].std()
        if std == 0:
            std = 1.0
        noise = np.random.standard_t(df=3, size=len(idx)) * (3 * std)
        X_out.loc[X_out.index[idx], col] = X_out.loc[X_out.index[idx], col] + noise
    return X_out

def induce_missingness(X_df, base_p=0.10, extra_p=0.15):
    """
    MCAR + MAR ผูกกับ label โดยประมาณ:
    - แถวที่ y=1 (เสียชีวิต) มีโอกาส missing มากกว่านิดหน่อย
    ทำใน test set ยากหน่อยเพราะไม่มี y ใน function -> ใช้ proxy = median split จาก lab เด่น ๆ
    เพื่อให้โครงสร้างคล้าย NHANES (MAR anchored)
    """
    Xmiss = X_df.copy()
    Xmiss[NUM_COLS] = Xmiss[NUM_COLS].astype(float)

    # MCAR
    mask = np.random.rand(*Xmiss[NUM_COLS].shape) < base_p
    Xmiss.loc[:, NUM_COLS] = Xmiss[NUM_COLS].mask(mask)

    # simple MAR: ถ้า column ใดมี variance เยอะ ใช้เป็น anchor
    if len(NUM_COLS) > 0:
        anchor_col = NUM_COLS[0]
        q75 = Xmiss[anchor_col].quantile(0.75)
        high_idx = Xmiss[anchor_col] >= q75
        for col in NUM_COLS:
            mask2 = (np.random.rand(len(Xmiss)) < extra_p) & high_idx
            Xmiss.loc[mask2, col] = np.nan

    return Xmiss

def distribution_shift(X_df, shift_scale=0.30):
    X_shift = X_df.copy()
    # shift เฉพาะ numeric
    for col in NUM_COLS:
        std = X_shift[col].std()
        if std == 0:
            std = 1.0
        X_shift[col] = X_shift[col] + shift_scale * std
    return X_shift

def corrupt_surrogate_features(X_df, gamma=0.5):
    """
    เลือก numeric บางตัวเป็น surrogate แล้วบิดเหมือน TCR ใน NHANES
    """
    X_cor = X_df.copy()
    # เอา lab สัก 3 ตัวสุ่มเป็น surrogate
    if len(NUM_COLS) > 0:
        np.random.seed(RANDOM_STATE)
        k = min(3, len(NUM_COLS))
        sur_cols = np.random.choice(NUM_COLS, size=k, replace=False)
        for col in sur_cols:
            sur = X_cor[col].astype(float)
            delta_sys = 0.1 * sur.mean()
            noise = np.random.normal(0, 0.2 * sur.std() if sur.std() > 0 else 1.0, size=len(sur))
            X_cor[col] = (1 - gamma)*sur + gamma*(sur + delta_sys + noise)
    return X_cor

def nonlinear_distortion_real(X_df, alpha=0.6):
    X_non = X_df.copy()
    # warp numeric บางส่วน
    for col in NUM_COLS:
        x = X_non[col].astype(float)
        if x.std() == 0:
            continue
        # ถ้าค่าบวกเยอะ -> log1p, ถ้ากระจายสองฝั่ง -> quadratic smooth
        if x.min() >= 0:
            g = np.log1p(np.maximum(x, 0))
        else:
            g = x**2 / (1 + np.abs(x))
        X_non[col] = (1 - alpha)*x + alpha*g
    return X_non

def flip_labels(y, eta=0.10):
    y2 = y.copy()
    mask = np.random.rand(len(y2)) < eta
    y2[mask] = 1 - y2[mask]
    return y2

def build_worlds():
    worlds = {}

    # A_real: clean
    worlds["WA_real"] = (X_test.copy(), y_test.copy())

    # B_real: noise + outliers
    Xb = add_gaussian_noise(X_test, std_fraction=0.20)
    Xb = add_outliers(Xb, frac=0.02)
    worlds["WB_real"] = (Xb, y_test.copy())

    # C_real: missingness (MCAR + MAR) + median impute
    Xc_raw = induce_missingness(X_test, base_p=0.10, extra_p=0.15)
    imp = SimpleImputer(strategy="median")
    Xc = Xc_raw.copy()
    Xc[NUM_COLS] = imp.fit_transform(Xc_raw[NUM_COLS])
    worlds["WC_real"] = (Xc, y_test.copy())

    # D_real: distribution shift
    Xd = distribution_shift(X_test, shift_scale=0.30)
    worlds["WD_real"] = (Xd, y_test.copy())

    # E_real: surrogate corruption
    Xe = corrupt_surrogate_features(X_test, gamma=0.5)
    worlds["WE_real"] = (Xe, y_test.copy())

    # F_real: nonlinear distortion
    Xf = nonlinear_distortion_real(X_test, alpha=0.6)
    worlds["WF_real"] = (Xf, y_test.copy())

    # G_real: label noise
    yg = flip_labels(y_test, eta=0.10)
    worlds["WG_real"] = (X_test.copy(), yg)

    return worlds

worlds = build_worlds()
print("Worlds:", list(worlds.keys()))

pert_set = ["WB_real","WC_real","WD_real","WE_real","WF_real","WG_real"]
ref_world = "WA_real"

# ------------------------
# 6) ECE
# ------------------------
def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_prob = np.clip(y_prob, 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1

    ece = 0.0
    N = len(y_prob)
    for i in range(n_bins):
        mask = bin_ids == i
        if not np.any(mask):
            continue
        bin_y = y_true[mask]
        bin_p = y_prob[mask]
        w_k = len(bin_y) / N
        acc_k = bin_y.mean()
        conf_k = bin_p.mean()
        ece += w_k * abs(acc_k - conf_k)
    return ece

def eval_on_world(pipe, Xw, yw):
    proba = pipe.predict_proba(Xw)[:,1]
    return dict(
        AUROC = roc_auc_score(yw, proba),
        Brier = brier_score_loss(yw, proba),
        LogLoss = log_loss(yw, proba),
        ECE = expected_calibration_error(yw, proba, n_bins=10)
    )

# ------------------------
# 7) TRAIN + EVAL ON ALL WORLDS
# ------------------------
records = []
for name, pipe in PIPELINES.items():
    print(f"Training: {name}")
    pipe.fit(X_train, y_train)

    for wname, (Xw, yw) in worlds.items():
        m = eval_on_world(pipe, Xw, yw)
        m["model"] = name
        m["world"] = wname
        records.append(m)

metrics_df = pd.DataFrame(records)
metrics_df.to_csv("mortality_world_metrics.csv", index=False)
print(metrics_df.head())

# ------------------------
# 8) R, S, C (A–G) แบบเดียวกับ NHANES code
# ------------------------
pivot_auc = metrics_df.pivot(index="model", columns="world", values="AUROC")
pivot_ece = metrics_df.pivot(index="model", columns="world", values="ECE")

rows = []
for model in pivot_auc.index:
    # R = AUROC บน WA_real
    R_raw = pivot_auc.loc[model, ref_world]

    # S = 1 - mean(AUROC drop)
    drops = [R_raw - pivot_auc.loc[model, w] for w in pert_set]
    S_raw = 1.0 - np.mean(drops)

    # C = 1 - mean(ECE drift)
    base_ece = pivot_ece.loc[model, ref_world]
    drifts = [abs(pivot_ece.loc[model, w] - base_ece) for w in pert_set]
    C_raw = 1.0 - np.mean(drifts)

    rows.append({
        "model": model,
        "R_raw": R_raw,
        "S_raw": S_raw,
        "C_raw": C_raw
    })

rsc_df = pd.DataFrame(rows)

# Normalize R,S,C → [0,1]
for col in ["R_raw","S_raw","C_raw"]:
    mn = rsc_df[col].min()
    mx = rsc_df[col].max()
    rsc_df[col[:-4]] = (rsc_df[col] - mn) / (mx - mn + 1e-8)

# ------------------------
# 9) E (Explainability Stability) – แบบเดียวกับ NHANES
# ------------------------
tree_models = ["RandomForest","ExtraTrees","GradientBoosting"]
E_records = []

for name in tree_models:
    print(f"Computing E for: {name}")
    pipe = PIPELINES[name]

    Xt = pipe.named_steps["preprocess"].transform(X_test)
    clf = pipe.named_steps["clf"]

    explainer = shap.TreeExplainer(clf)

    n_sub = min(200, Xt.shape[0])  # ลด sample หน่อยเพราะ N เล็ก
    idx = np.random.choice(Xt.shape[0], n_sub, replace=False)

    shapA_all = explainer.shap_values(Xt[idx])
    if isinstance(shapA_all, list):
        shapA = np.array(shapA_all[1], dtype=float)
    else:
        shapA = np.array(shapA_all, dtype=float)
    if shapA.ndim > 2:
        shapA = shapA.reshape(shapA.shape[0], -1)

    for wname in pert_set:
        Xw, _ = worlds[wname]
        Xw_p = pipe.named_steps["preprocess"].transform(Xw.iloc[idx])

        shapW_all = explainer.shap_values(Xw_p)
        if isinstance(shapW_all, list):
            shapW = np.array(shapW_all[1], dtype=float)
        else:
            shapW = np.array(shapW_all, dtype=float)
        if shapW.ndim > 2:
            shapW = shapW.reshape(shapW.shape[0], -1)

        # Global importance
        gA = np.mean(np.abs(shapA), axis=0)
        gW = np.mean(np.abs(shapW), axis=0)
        rho, _ = spearmanr(gA, gW)
        rho = np.nan_to_num(rho)
        if isinstance(rho, np.ndarray):
            rho_val = float(rho.ravel()[0])
        else:
            rho_val = float(rho)

        # Local cosine
        eps = 1e-8
        num = np.sum(shapA * shapW, axis=1)
        den = (np.linalg.norm(shapA, axis=1)+eps) * (np.linalg.norm(shapW, axis=1)+eps)
        cos_val = float(np.mean(num / den))

        E_records.append({
            "model": name,
            "world": wname,
            "global_spearman": rho_val,
            "local_cosine": cos_val
        })

E_df = pd.DataFrame(E_records)
E_df.to_csv("mortality_E_components.csv", index=False)
print(E_df.head())

E_summary = (
    E_df.groupby("model")[["global_spearman","local_cosine"]]
    .mean()
    .reset_index()
)

# Normalize global_spearman, local_cosine -> [0,1]
for col in ["global_spearman","local_cosine"]:
    mn = E_summary[col].min()
    mx = E_summary[col].max()
    E_summary[col + "_norm"] = (E_summary[col] - mn) / (mx - mn + 1e-8)

E_summary["E_raw"] = 0.5 * E_summary["global_spearman_norm"] + 0.5 * E_summary["local_cosine_norm"]

mn_E = E_summary["E_raw"].min()
mx_E = E_summary["E_raw"].max()
E_summary["E"] = (E_summary["E_raw"] - mn_E) / (mx_E - mn_E + 1e-8)

E_summary.to_csv("mortality_E_scores.csv", index=False)
print(E_summary)

# ------------------------
# 10) RSCE = 0.4R + 0.3S + 0.2C + 0.1E
# ------------------------
final = pd.merge(rsc_df, E_summary[["model","E"]], on="model", how="left")

final["RSCE_full"] = (
    0.4 * final["R"] +
    0.3 * final["S"] +
    0.2 * final["C"] +
    0.1 * final["E"].fillna(0)   # models ที่ไม่มี E (ไม่ใช่ tree_models) -> 0
)

final.to_csv("mortality_RSCE_full_scores.csv", index=False)
print(final.sort_values("RSCE_full", ascending=False))

# ตัวอย่างกราฟ RSCE_full
plt.figure(figsize=(8,5))
final_sorted = final.sort_values("RSCE_full", ascending=False)
plt.bar(final_sorted["model"], final_sorted["RSCE_full"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("RSCE_full")
plt.title("RSCE_full by model (hospital mortality)")
plt.tight_layout()
plt.show()