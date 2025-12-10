import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 0. เลือกโมเดลที่จะ plot
# ----------------------------------------------------
FOCUS_MODELS = ["RandomForest", "ExtraTrees", "GradientBoosting"]

# ----------------------------------------------------
# 1) Sensitivity ต่อ weight บน Reliability R (เหมือนกราฟล่าสุด)
# ----------------------------------------------------
rsce_df = pd.read_csv("mortality_RSCE_full_scores.csv")
rsce_df = rsce_df.dropna(subset=["R", "S", "C", "E"])
rsce_df = rsce_df[rsce_df["model"].isin(FOCUS_MODELS)]

comp = rsce_df.groupby("model")[["R", "S", "C", "E"]].mean()

w_base = np.array([0.4, 0.3, 0.2, 0.1])   # [wR, wS, wC, wE]
ratio_rest = w_base[1:] / w_base[1:].sum()

wR_values = np.linspace(0.3, 0.5, 7)   # ปรับน้ำหนัก R ตั้งแต่ 0.3–0.5
rsce_sweep = {m: [] for m in FOCUS_MODELS}

for wR in wR_values:
    remaining = 1.0 - wR
    wS, wC, wE = remaining * ratio_rest

    for m in FOCUS_MODELS:
        Rm, Sm, Cm, Em = comp.loc[m, ["R", "S", "C", "E"]].values
        rsce = wR*Rm + wS*Sm + wC*Cm + wE*Em
        rsce_sweep[m].append(rsce)

# ----------------------------------------------------
# 2) Sensitivity ต่อ "severity" ของ perturbation (ใช้ world จริง ๆ เป็น proxy)
# ----------------------------------------------------
world_df = pd.read_csv("mortality_world_metrics.csv")

# ใช้แต่ worlds ที่เป็น real (แล้วแต่ naming ของคุณ)
world_df = world_df[world_df["world"].str.endswith("_real")]
world_df = world_df[world_df["model"].isin(FOCUS_MODELS)]

# จัด order โลกให้เป็นเหมือน "ระดับความแรง" คร่าว ๆ
world_order = [
    "WA_real",  # reference
    "WB_real",  # hospital skew
    "WC_real",  # noise + missing
    "WD_real",  # shift
    "WE_real",  # surrogate corruption
    # ถ้ามี WF_real, WG_real เพิ่มเข้าไปได้
]

world_order = [w for w in world_order if w in world_df["world"].unique()]

severity_map = {w: i for i, w in enumerate(world_order)}  # 0,1,2,...

world_df = world_df[world_df["world"].isin(world_order)].copy()
world_df["severity"] = world_df["world"].map(severity_map)

# สร้าง composite world score จาก AUROC, Brier, LogLoss, ECE
metrics = ["AUROC", "Brier", "LogLoss", "ECE"]
for m in metrics:
    x = world_df[m].values
    if m == "AUROC":
        # สูงดีกว่า
        world_df[m + "_norm"] = (x - x.min()) / (x.max() - x.min() + 1e-12)
    else:
        # ต่ำดีกว่า -> invert
        z = (x - x.min()) / (x.max() - x.min() + 1e-12)
        world_df[m + "_norm"] = 1.0 - z

world_df["world_score"] = world_df[[m + "_norm" for m in metrics]].mean(axis=1)

# ----------------------------------------------------
# 3) วาดรูป 2 subplot ในรูปเดียว
# ----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ---- ซ้าย: weight sensitivity ----
ax = axes[0]
for m in FOCUS_MODELS:
    ax.plot(wR_values, rsce_sweep[m], marker="o", label=m)

ax.set_xlabel("Weight on Reliability R")
ax.set_ylabel("RSCE Score")
ax.set_title("Sensitivity of RSCE to Weight on R")
ax.grid(True)

# ใส่ legend แยกด้านนอกภาพรวมทีเดียว
# ---- ขวา: severity / world sensitivity ----
ax2 = axes[1]
for m in FOCUS_MODELS:
    sub = world_df[world_df["model"] == m].sort_values("severity")
    ax2.plot(
        sub["severity"],
        sub["world_score"],
        marker="o",
        label=m
    )

ax2.set_xlabel("Perturbation Severity (world index)")
ax2.set_ylabel("Composite World Score")
ax2.set_title("Sensitivity to Perturbation Severity (Real Worlds)")
ax2.set_xticks(list(severity_map.values()))
ax2.set_xticklabels(world_order, rotation=45)
ax2.grid(True)

# legend รวมด้านขวา
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(FOCUS_MODELS))

plt.tight_layout(rect=[0, 0, 1, 0.92])  # เผื่อพื้นที่ legend ด้านบน
plt.savefig("sensitivity_combined.png", dpi=300)
plt.show()