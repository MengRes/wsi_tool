import os
import pandas as pd


# === Setting ===
manifest_path = "check_results/tcga-luad.txt"                   # TCGA manifest file path
svs_folder = "/home/mxz3935/dataset_folder/tcga-luad/"          # Local downloaded .svs path
output_csv = "check_results/tcga-luad_missing_files.csv"        # Missing file information output path
missing_manifest = "check_results/tcga-luad_missing_manifest.txt"             # Missing file manifest（for gdc-client）

# === Read manifest file ===
manifest = pd.read_csv(manifest_path, sep="\t")
if "filename" not in manifest.columns:
    raise ValueError("Manifest file lacks 'filename' column")

manifest_filenames = set(manifest["filename"])

# === Recursively scan local folders and collect .svs files ===
existing_files = set()
for root, _, files in os.walk(svs_folder):
    for file in files:
        if file.lower().endswith(".svs"):
            existing_files.add(file)

# === Find failed downloaded file ===
missing_filenames = manifest_filenames - existing_files
missing_df = manifest[manifest["filename"].isin(missing_filenames)]

# === Save .csv file ===
missing_df.to_csv(output_csv, index=False)
print(f"Total {len(missing_filenames)} fails download, details are saved in: {output_csv}")

# === Generate GDC download manifest（keep necessary column） ===
required_columns = ["id", "filename", "md5", "size", "state"]
available_columns = [col for col in required_columns if col in missing_df.columns]
manifest_for_download = missing_df[available_columns]
manifest_for_download.to_csv(missing_manifest, sep="\t", index=False)
print(f"Generate GDC download manifest: {missing_manifest}")