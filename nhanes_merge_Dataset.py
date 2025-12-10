import pandas as pd
from pathlib import Path

DATA_DIR = Path(".")

files = {
    "DEMO_L":   "DEMO_L.xpt",
    "CBC_L":    "CBC_L.xpt",
    "BIOPRO_L": "BIOPRO_L.xpt",
    "HDL_L":    "HDL_L.XPT",   
    "TRIGLY_L": "TRIGLY_L.xpt",
    "TCHOL_L":  "TCHOL_L.xpt",
    "GLU_L":    "GLU_L.xpt",
    "FASTQX_L": "FASTQX_L.xpt",
    "BMX_L":    "BMX_L.xpt",
    "AUQ_L":    "AUQ_L.xpt",
}

def read_xpt(name, filename):
    path = DATA_DIR / filename
    print(f"\n=== Reading {name}: {path} ===")
    df = pd.read_sas(path, format="xport")
    print(f"{name}: shape = {df.shape}")
    print(f"{name}: columns = {list(df.columns)[:10]} ...") 
    return df

dfs = {}
for key, fname in files.items():
    dfs[key] = read_xpt(key, fname)

merged = dfs["DEMO_L"]

for key, df in dfs.items():
    if key == "DEMO_L":
        continue
    print(f"Merging DEMO_L with {key} on SEQN ...")
    merged = merged.merge(df, on="SEQN", how="left")

print("\nFinal merged shape:", merged.shape)

output_name = "nhanes_merged_10files.csv"
merged.to_csv(output_name, index=False)
print(f"\nSaved merged CSV to: {output_name}")

df = pd.read_csv("nhanes_rsce_dataset.csv")
print("Before dropna:", df.shape)
df_clean = df.dropna()
print("After dropna:", df_clean.shape)
df_clean.to_csv("nhanes_rsce_dataset_clean.csv", index=False)
print("Saved: nhanes_rsce_dataset_clean.csv")