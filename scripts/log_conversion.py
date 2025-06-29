import numpy as np
import os


# ONLY RUN ONCE TO GENERATE THE DATASETS
def apply_log1p_to_npz_datasets(folder_path="data/processed"):
    """
    Loads train/val/test .npz datasets, applies log1p to X and y,
    and saves new files with '_log1p' in the filename.
    """
    file_map = {
        "train_data.npz": "train_data_log1p.npz",
        "val_data.npz": "val_data_log1p.npz",
        "test_data.npz": "test_data_log1p.npz"
    }

    for original, transformed in file_map.items():
        input_path = os.path.join(folder_path, original)
        output_path = os.path.join(folder_path, transformed)

        if not os.path.exists(input_path):
            print(f"File not found: {input_path}")
            continue

        data = np.load(input_path)
        X, y = data["X"], data["y"]

        X_log = np.log1p(X)
        y_log = np.log1p(y)

        np.savez_compressed(output_path, X=X_log, y=y_log)
        print(f"Saved log1p-transformed dataset to: {output_path}")


apply_log1p_to_npz_datasets()