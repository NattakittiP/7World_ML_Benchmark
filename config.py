"""
config.py (v2)

Configuration for preprocessing the synthetic WBV–TCR cohort
for the IEEE Access robustness benchmark.

Key change from v1:
- We DO NOT use TG4h or TCR as features.
- Target is still defined from TG4h (>= 75th percentile),
  but TG4h is NOT part of the feature set.
"""

from __future__ import annotations

from pathlib import Path

# -----------------------------
# Basic paths
# -----------------------------

# ชื่อไฟล์ CSV ดิบ (ปรับ path ตามไฟล์ของคุณ)
DATA_PATH: Path = Path("WBV_TCR_ICBCB2026_Synthetic_1500_precise_v2 (2).csv")

# โฟลเดอร์สำหรับเซฟ output (จะสร้างอัตโนมัติถ้ายังไม่มี)
OUTPUT_DIR: Path = Path("preprocessed_outputs")

# -----------------------------
# Modeling configuration
# -----------------------------

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2  # 20% test set

# Target column name ที่จะสร้างขึ้นใหม่
TARGET_COL: str = "HighTGResponder"

# ชื่อคอลัมน์ใน dataset ดิบ
ID_COL: str = "ID"

# Categorical features
CATEGORICAL_COLS = [
    "Sex",
]

# Numerical features
# IMPORTANT:
#   - ไม่ใส่ TG4h
#   - ไม่ใส่ TCR (เพราะสัมพันธ์กับ TG0h/TG4h และใช้ทำ target)
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

#   ถ้าจะทดลองเพิ่ม TCR กลับมาในภายหลัง สามารถแก้ไขตรงนี้ทีหลังได้
#   แต่เวอร์ชัน baseline สำหรับ paper ควรใช้เฉพาะ baseline/fasting markers แบบนี้ก่อน

# -----------------------------
# File names for saved artifacts
# -----------------------------

PREPROCESSOR_PATH: Path = OUTPUT_DIR / "preprocessor.joblib"

TRAIN_ARRAYS_PATH: Path = OUTPUT_DIR / "train_arrays.npz"
TEST_ARRAYS_PATH: Path = OUTPUT_DIR / "test_arrays.npz"

SUMMARY_JSON_PATH: Path = OUTPUT_DIR / "data_summary.json"