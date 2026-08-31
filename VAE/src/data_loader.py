"""
Data Loader Module for VAE + RF + AdaBoost Pipeline.
Supports local execution and Google Colab environments.
"""

import os
import pandas as pd
from typing import Optional


def find_dataset_path(custom_path: Optional[str] = None) -> str:
    """
    Auto-detect dataset path across local workspace and Google Colab environments.
    """
    if custom_path and os.path.exists(custom_path):
        return custom_path

    candidates = [
        "dataset/gastric_cancer_dataset_raw.csv",
        "../dataset/gastric_cancer_dataset_raw.csv",
        "../../dataset/gastric_cancer_dataset_raw.csv",
        "/content/dataset/gastric_cancer_dataset_raw.csv",
        "/content/major_project/dataset/gastric_cancer_dataset_raw.csv",
        "/content/drive/MyDrive/major_project/dataset/gastric_cancer_dataset_raw.csv",
    ]

    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)

    raise FileNotFoundError(
        f"Could not locate 'gastric_cancer_dataset_raw.csv'. Looked in: {candidates}."
    )


def load_raw_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads raw gastric cancer CSV dataset.
    """
    resolved_path = find_dataset_path(data_path)
    print(f"[DataLoader] Loading raw dataset from: {resolved_path}")
    df = pd.read_csv(resolved_path)
    print(f"[DataLoader] Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return df
