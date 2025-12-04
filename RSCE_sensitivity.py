import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1) Load RSCE component scores
# -----------------------------
df = pd.read_csv("RSCE_errorbars.csv")

# จัดเรียงตาม RSCE baseline (weights 0.4/0.3/0.2/0.1)
df = df.sort_values("RSCE", ascending=False).reset_index(drop=True)

# ดึง component ที่ normalize แล้ว
R = df["R_norm"].values
S = df["S_norm"].values
C = df["C_norm"].values
E = df["E_norm"].values
models = df["model"].values

# -----------------------------
# 2) สร้างชุด weights หลายแบบ
# -----------------------------
# base weight (ใช้ใน manuscript หลัก)
base_w = np.array([0.4, 0.3, 0.2, 0.1])

# ลองปรับน้ำหนักของ E ตั้งแต่ 0 → 0.2 แล้ว renormalize
E_weights = np.linspace(0.0, 0.2, 9)  # 9 ค่า: 0.00, 0.025, ..., 0.20

all_w = []
for wE in E_weights:
    # ให้ R,S,C คงสัดส่วนเดิม (0.4:0.3:0.2) แล้วเหลือที่ไม่ใช่ wE
    rest = 1.0 - wE
    wR = rest * 0.4 / (0.4 + 0.3 + 0.2)
    wS = rest * 0.3 / (0.4 + 0.3 + 0.2)
    wC = rest * 0.2 / (0.4 + 0.3 + 0.2)
    w = np.array([wR, wS, wC, wE])
    all_w.append(w)

all_w = np.stack(all_w, axis=0)  # shape: (n_weights, 4)

# -----------------------------
# 3) คำนวณ RSCE ใต้ทุกชุด weights
# -----------------------------
# RSCE_sens[weight_index, model_index]
RSCE_sens = np.zeros((all_w.shape[0], len(models)))

for i, w in enumerate(all_w):
    RSCE_sens[i, :] = (
        w[0] * R +
        w[1] * S +
        w[2] * C +
        w[3] * E
    )

# -----------------------------
# 4) วิเคราะห์ rank stability
# -----------------------------
# นับว่าแต่ละ model อยู่ top-3 บ่อยแค่ไหน
top3_counts = np.zeros(len(models), dtype=int)

for i in range(all_w.shape[0]):
    order = np.argsort(RSCE_sens[i, :])[::-1]  # high → low
    top3 = order[:3]
    top3_counts[top3] += 1

top3_fraction = top3_counts / float(all_w.shape[0])

# -----------------------------
# 5) Plot: RSCE sensitivity figure
# -----------------------------
plt.rcParams["figure.dpi"] = 300

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# ---------- Panel A: RSCE curves for top-5 baseline models ----------
ax = axes[0]

# เลือกโมเดล top-5 จาก baseline RSCE
top_k = 5
top_idx = np.arange(top_k)
top_models = models[top_idx]

for j, m in enumerate(top_models):
    ax.plot(
        E_weights,
        RSCE_sens[:, j],
        marker="o",
        label=m
    )

ax.set_xlabel("Weight on explainability stability $w_E$")
ax.set_ylabel("Composite RSCE score")
ax.set_title("RSCE sensitivity for top-5 models")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(fontsize=8, loc="best")

# ---------- Panel B: fraction of configurations with top-3 rank ----------
ax2 = axes[1]

x = np.arange(len(models))
ax2.bar(x, top3_fraction)
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=45, ha="right")
ax2.set_ylim(0, 1.05)
ax2.set_ylabel("Fraction of weight settings\nwith top-3 RSCE rank")
ax2.set_title("Rank stability under RSCE re-weighting")
ax2.grid(axis="y", linestyle="--", alpha=0.4)

fig.tight_layout()

# Save figure
fig.savefig("RSCE_sensitivity.pdf", bbox_inches="tight")
fig.savefig("RSCE_sensitivity.png", dpi=300, bbox_inches="tight")
print("Saved RSCE_sensitivity.pdf and RSCE_sensitivity.png")
