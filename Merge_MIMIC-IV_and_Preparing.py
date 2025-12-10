import pandas as pd

# ----------------------------
# 1) โหลดไฟล์จาก Google Colab
# ----------------------------
# ถ้าคุณยังไม่ได้ upload
# from google.colab import files
# uploaded = files.upload()  # จากนั้นเลือกไฟล์ 3 อัน: patients.csv, admissions.csv, labevents.csv

# ----------------------------
# 2) อ่านไฟล์ CSV
# ----------------------------

patients = pd.read_csv("patients.csv")
admissions = pd.read_csv("admissions.csv")

# labevents ใหญ่ จึงควรกำหนด dtype บางคอลัมน์เพื่อลด memory error
labevents = pd.read_csv("labevents.csv", 
                        low_memory=False,
                        dtype={
                            "itemid": "Int64",
                            "valuenum": "float64",
                            "value": "string",
                            "valueuom": "string",
                            "flag": "string"
                        })

# ----------------------------
# 3) Merge ขั้นที่ 1:
# patients + admissions (key = subject_id)
# ----------------------------

df_pa = admissions.merge(patients, on="subject_id", how="left")

# ----------------------------
# 4) Merge ขั้นที่ 2:
# (patients + admissions) + labevents
# โดยใช้ key = subject_id และ hadm_id
# ----------------------------

df_final = labevents.merge(df_pa, on=["subject_id", "hadm_id"], how="left")

# ----------------------------
# 5) ดูผลลัพธ์และขนาด
# ----------------------------
print("Shape of final merged dataset:", df_final.shape)
df_final.head()
df_final.to_csv("merged_lab_admission_patient.csv", index=False)
print("Saved merged file!")

# โหลดไฟล์ merged ที่คุณมีอยู่แล้ว
df = pd.read_csv("merged_lab_admission_patient.csv")

# --------------------------------------------
# กำหนด "คอลัมน์สำคัญ" ที่ต้องไม่ให้ Missing
# --------------------------------------------
important_cols = [
    "hadm_id",      # ต้องมี hospital admission ID
    "valuenum",     # ค่าตัวเลขของ lab
    "valueuom",     # หน่วยของ lab
    "charttime" if "charttime" in df.columns else None  # ถ้ามี column นี้ให้ใช้
]

# ลบ None ออกจาก list เผื่อ charttime ไม่มี
important_cols = [c for c in important_cols if c is not None]

print("คอลัมน์สำคัญที่จะใช้ในการลบ:", important_cols)

# --------------------------------------------
# ลบแถวที่ missing ในคอลัมน์สำคัญ
# --------------------------------------------
df_clean = df.dropna(subset=important_cols)

print("ข้อมูลก่อนลบ:", df.shape)
print("ข้อมูลหลังลบ :", df_clean.shape)

# ดูตัวอย่าง
df_clean.head()
df_final.to_csv("clear_merged_lab_admission_patient.csv", index=False)
print("Saved merged file!")

from pathlib import Path

# ==============================
# 0) ตั้งค่าเบื้องต้น
# ==============================
DATA_PATH = Path("clear_merged_lab_admission_patient.csv")
OUTPUT_PATH = Path("analytic_dataset_mortality_all_admissions.csv")

TOP_N_LABS = 30  # จำนวน itemid ที่จะใช้เป็น feature

# ==============================
# 1) โหลดข้อมูล
# ==============================
df = pd.read_csv(DATA_PATH)
print("Shape after load:", df.shape)
print("Columns:", list(df.columns))

# แปลง id เป็นตัวเลข (ถ้าจำเป็น)
for col in ["subject_id", "hadm_id", "labevent_id", "specimen_id", "itemid"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

# แปลง age
if "anchor_age" in df.columns:
    df["anchor_age"] = pd.to_numeric(df["anchor_age"], errors="coerce")
else:
    print("คำเตือน: ไม่พบคอลัมน์ anchor_age — จะไม่ filter ตามอายุ")
    df["anchor_age"] = pd.NA  # ให้มีคอลัมน์ไว้เฉย ๆ

# ==============================
# 2) สร้าง cohort: ผู้ใหญ่ (ทุก admission)
# ==============================
# 2.1 filter อายุ >= 18 (ถ้ามี)
if df["anchor_age"].notna().any():
    df_adult = df[df["anchor_age"] >= 18].copy()
    print("Shape after age >= 18:", df_adult.shape)
else:
    df_adult = df.copy()
    print("ไม่สามารถ filter ตาม anchor_age ได้ → ใช้ทุกแถว")

# 2.2 ใช้ทุก admission (ไม่ตัด first admission ต่อ subject แล้ว)
df_cohort = df_adult.copy()
print("Shape of cohort (all adult admissions):", df_cohort.shape)
print("Unique subject_id:", df_cohort["subject_id"].nunique())
print("Unique hadm_id:", df_cohort["hadm_id"].nunique())

# ==============================
# 3) สร้าง label: in-hospital mortality
# ==============================
if "hospital_expire_flag" in df_cohort.columns:
    df_cohort["label_mortality"] = df_cohort["hospital_expire_flag"].fillna(0).astype(int)
else:
    print("คำเตือน: ไม่พบ hospital_expire_flag — จะสร้าง label_mortality = 0 ทั้งหมด")
    df_cohort["label_mortality"] = 0

print("Label value counts:")
print(df_cohort["label_mortality"].value_counts(dropna=False))

# ==============================
# 4) เตรียม lab data (ไม่ filter ตามเวลา)
# ==============================
if not {"hadm_id", "itemid", "valuenum"}.issubset(df_cohort.columns):
    print("คำเตือน: ไม่มี hadm_id หรือ itemid หรือ valuenum ครบ → จะไม่มี lab features")
    df_labs = pd.DataFrame(columns=["hadm_id", "itemid", "valuenum"])
else:
    df_labs = df_cohort[["hadm_id", "itemid", "valuenum"]].copy()

print("Shape lab data (no time filter):", df_labs.shape)

# ==============================
# 5) เลือก top-N lab itemid ที่เจอบ่อยสุด
# ==============================
if not df_labs.empty:
    lab_counts = df_labs["itemid"].value_counts()
    print("จำนวน itemid ไม่ซ้ำ:", lab_counts.shape[0])

    top_itemids = lab_counts.head(TOP_N_LABS).index.tolist()
    print(f"Using TOP {TOP_N_LABS} lab itemids as features:")
    print(top_itemids)

    df_labs_top = df_labs[df_labs["itemid"].isin(top_itemids)].copy()
else:
    top_itemids = []
    df_labs_top = df_labs.copy()
    print("ไม่มีข้อมูล lab ให้เลือก top itemid")

print("Shape labs with top itemids:", df_labs_top.shape)

# ==============================
# 6) aggregate lab ต่อ (hadm_id, itemid) → median
# ==============================
if not df_labs_top.empty:
    lab_agg = (
        df_labs_top
        .dropna(subset=["valuenum"])
        .groupby(["hadm_id", "itemid"])["valuenum"]
        .median()
        .reset_index()
    )
else:
    lab_agg = pd.DataFrame(columns=["hadm_id", "itemid", "valuenum"])

print("Shape after lab aggregation (hadm_id, itemid):", lab_agg.shape)
print("Unique hadm_id in lab_agg:", lab_agg["hadm_id"].nunique() if not lab_agg.empty else 0)

# ==============================
# 7) pivot wide: 1 แถว = 1 hadm_id
# ==============================
if not lab_agg.empty:
    lab_wide = lab_agg.pivot(index="hadm_id", columns="itemid", values="valuenum")
    lab_wide.columns = [f"lab_{int(col)}" for col in lab_wide.columns]
    lab_wide = lab_wide.reset_index()
else:
    lab_wide = pd.DataFrame(columns=["hadm_id"])
    print("คำเตือน: ไม่มี lab aggregation → lab_wide ว่าง (จะมีแต่ hadm_id เท่านั้นถ้า merge)")

print("Shape of lab_wide:", lab_wide.shape)

# ==============================
# 8) เตรียม demographic + label table (1 แถว = 1 hadm_id)
# ==============================
candidate_demo_cols = [
    "subject_id",
    "gender",
    "anchor_age",
    "race",
    "marital_status",
    "insurance",
    "admission_type",
    "admission_location",
    "discharge_location",
    "anchor_year",
    "anchor_year_group",
]

demo_cols = [c for c in candidate_demo_cols if c in df_cohort.columns]
print("Demographic columns used:", demo_cols)

# ให้เหลือ 1 แถวต่อ hadm_id
df_cohort_unique = (
    df_cohort
    .sort_values(by=["hadm_id"])
    .drop_duplicates(subset=["hadm_id"])
    .copy()
)

demo_cols_final = ["hadm_id"] + demo_cols + ["label_mortality"]
df_demo = df_cohort_unique[demo_cols_final].copy()

print("Shape of demographic + label table:", df_demo.shape)
print("Unique hadm_id in demo:", df_demo["hadm_id"].nunique())

# ==============================
# 9) merge lab_wide + demo → analytic dataset
# ==============================
if "hadm_id" in lab_wide.columns and "hadm_id" in df_demo.columns:
    df_analytic = df_demo.merge(lab_wide, on="hadm_id", how="left")
else:
    print("คำเตือน: merge ด้วย hadm_id ไม่ได้ → analytic dataset จะไม่มี lab features")
    df_analytic = df_demo.copy()

print("Final analytic dataset shape:", df_analytic.shape)
print("Unique hadm_id in final:", df_analytic["hadm_id"].nunique())

# ==============================
# 10) ตรวจ missing + save
# ==============================
if not df_analytic.empty:
    missing_report = df_analytic.isna().mean().sort_values(ascending=False)
    print("Missing rate (top 20 columns):")
    print(missing_report.head(20))

df_analytic.to_csv(OUTPUT_PATH, index=False)
print(f"Saved analytic dataset to: {OUTPUT_PATH.resolve()}")
