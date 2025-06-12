import os
import shutil

import os
import shutil

def move_svs_files_to_root(folder_A):
    """
    Move all .svs files from subdirectories to the root folder.
    If a file with the same name exists, it will be overwritten.
    """
    for root, dirs, files in os.walk(folder_A):
        if root == folder_A:
            continue  # Skip the root folder itself

        for file in files:
            if file.endswith(".svs"):
                source_path = os.path.join(root, file)
                target_path = os.path.join(folder_A, file)

                # Overwrite if file already exists in root
                if os.path.exists(target_path):
                    print(f"Overwriting existing file: {target_path}")
                    os.remove(target_path)

                print(f"Moving file: {source_path} -> {target_path}")
                shutil.move(source_path, target_path)


def remove_empty_subfolders(folder_A):
    """
    Remove all subdirectories under the given folder that do not contain any .svs files.
    Traverses from bottom to top (post-order).
    """
    for dirpath, dirnames, filenames in os.walk(folder_A, topdown=False):
        if dirpath == folder_A:
            continue  # Skip the root folder

        has_svs = any(f.endswith(".svs") for f in filenames)
        if not has_svs:
            print(f"Removing empty folder (no .svs files): {dirpath}")
            shutil.rmtree(dirpath)


if __name__ == "__main__":
    
    # Set the path to folder A (change this to your actual path)
    folder_A = "/home/mxz3935/dataset_folder/tcga-luad/"

    move_svs_files_to_root(folder_A)
    remove_empty_subfolders(folder_A)