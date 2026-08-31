"""
Data Loader Module for Gastric Cancer Classification Pipeline.
Supports local execution as well as Google Colab environments.
"""

import os
import pandas as pd
from typing import Optional, Tuple

from sklearn.model_selection import train_test_split


def find_dataset_path(custom_path: Optional[str] = None) -> str:
    """
    Auto-detect the dataset path across local workspace and Google Colab environments.
    """
    if custom_path and os.path.exists(custom_path):
        return custom_path

    # Candidate paths
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
        f"Could not locate 'gastric_cancer_dataset_raw.csv'. Looked in: {candidates}. "
        "Please provide an explicit --data_path argument."
    )


def load_raw_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads raw gastric cancer CSV dataset.
    """
    resolved_path = find_dataset_path(data_path)
    print(f"[DataLoader] Loading raw dataset from: {resolved_path}")
    df = pd.read_csv(resolved_path)
    print(f"[DataLoader] Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


def split_and_save_data(
    df: Optional[pd.DataFrame] = None,
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    test_size: float = 0.20,
    random_state: int = 42,
    target_col: Optional[str] = None,
) -> Tuple[str, str]:
    """ 
    Splits the raw dataset into train (80%) and test (20%) sets, saving both to CSV files.

    Parameters:
        df: Input DataFrame. If None, it will be loaded via load_raw_data(data_path).
        data_path: Optional path to raw dataset CSV if df is not passed.
        output_dir: Directory where train.csv and test.csv will be saved.
                    Defaults to the directory containing the source CSV.
        test_size: Proportion of data to include in test split (default: 0.20).
        random_state: Seed for reproducible split.
        target_col: Optional target column name to apply Stratified splitting.

    Returns:
        Tuple containing (train_csv_path, test_csv_path).
    """
    if df is None:
        df = load_raw_data(data_path)

    # Determine stratification target if provided
    stratify = df[target_col] if target_col and target_col in df.columns else None

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Determine default output directory relative to source data path
    if output_dir is None:
        resolved_source_path = find_dataset_path(data_path)
        output_dir = os.path.dirname(resolved_source_path)

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "gastric_cancer_train.csv")
    test_path = os.path.join(output_dir, "gastric_cancer_test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[DataLoader] Data split complete:")
    print(f"             - Train Set: {train_df.shape[0]} rows ({(1 - test_size) * 100:.0f}%)")
    print(f"             - Test Set:  {test_df.shape[0]} rows ({test_size * 100:.0f}%)")
    print(f"[DataLoader] Saved Train CSV -> {train_path}")
    print(f"[DataLoader] Saved Test CSV  -> {test_path}")

    return train_path, test_path


if __name__ == "__main__":
    # Script entry point to run split directly from command line
    split_and_save_data()