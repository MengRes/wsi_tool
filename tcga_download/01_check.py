import os
import csv


def find_svs_files(root_dir, output_csv):
    svs_files = []

    # Traverse the folder and its subfolders
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith(".svs"):
                full_path = os.path.join(dirpath, file)
                svs_files.append((file, full_path))

    # Write CSV file
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["File Name", "Full Path"])
        writer.writerows(svs_files)

    print(f"Found {len(svs_files)} .svs files.")
    print(f"Results saved to: {output_csv}")

# Get folder size
def get_folder_size(path):
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size

def format_size(bytes_size):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024

if __name__ == "__main__":
    # TCGA-LUAD as example
    root_folder = "/home/mxz3935/dataset_folder/tcga-luad/"
    output_csv = "check_results/tcga-luad_svs_files_summary.csv"
    find_svs_files(root_folder, output_csv)

    size_bytes = get_folder_size(root_folder)
    print(f"Folder size: {format_size(size_bytes)}")