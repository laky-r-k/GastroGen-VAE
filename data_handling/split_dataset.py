"""Create a stratified raw train/test split and save it under dataset/.

Example:
    python3 data_handling/split_dataset.py --train-percent 80 --test-percent 20
    python3 data_handling/split_dataset.py --train-percent 70 --test-percent 30

The script only splits rows. It deliberately does not clean, transform, scale,
or select features; those steps belong in the reusable preprocessing pipeline.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "dataset" / "orginal_data" / "gastric_cancer_dataset_raw.csv"


def _class_counts(series: pd.Series) -> Dict[str, int]:
    counts = series.value_counts(dropna=False).sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def split_dataset(
    source: Path,
    output_dir: Path,
    train_percent: int,
    test_percent: int,
    label_column: str = "label",
    id_column: str = "patient_id",
    random_seed: int = 42,
    overwrite: bool = False,
) -> Dict[str, object]:
    """Split source rows and write raw train/test CSVs plus split metadata."""
    if train_percent <= 0 or test_percent <= 0:
        raise ValueError("Train and test percentages must both be greater than zero.")
    if train_percent + test_percent != 100:
        raise ValueError("Train and test percentages must add up to 100.")
    if not source.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {source}")

    train_path = output_dir / "gastric_cancer_train.csv"
    test_path = output_dir / "gastric_cancer_test.csv"
    metadata_path = output_dir / "split_metadata.json"
    output_files = (train_path, test_path, metadata_path)
    existing = [path for path in output_files if path.exists()]
    if existing and not overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Split output already exists ({paths}). Choose another output directory "
            "or pass --overwrite to replace these files."
        )

    data = pd.read_csv(source)
    if label_column not in data.columns:
        raise ValueError(f"Label column {label_column!r} is missing from {source}.")
    if data[label_column].isna().any():
        raise ValueError(
            f"The label column {label_column!r} contains missing values; "
            "resolve them before stratified splitting."
        )
    if id_column in data.columns and data[id_column].duplicated().any():
        raise ValueError(f"ID column {id_column!r} contains duplicate patient IDs.")

    train_df, test_df = train_test_split(
        data,
        test_size=test_percent / 100.0,
        random_state=random_seed,
        shuffle=True,
        stratify=data[label_column],
    )

    if id_column in data.columns:
        overlap = set(train_df[id_column]) & set(test_df[id_column])
        all_ids = set(train_df[id_column]) | set(test_df[id_column])
        if overlap:
            raise RuntimeError(f"Patient IDs appeared in both splits: {sorted(overlap)[:5]}")
        if all_ids != set(data[id_column]):
            raise RuntimeError("The train/test split does not cover every source patient ID.")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(source.resolve()),
        "train_csv": train_path.name,
        "test_csv": test_path.name,
        "total_rows": int(len(data)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "requested_train_percent": int(train_percent),
        "requested_test_percent": int(test_percent),
        "actual_train_percent": round(len(train_df) / len(data) * 100, 4),
        "actual_test_percent": round(len(test_df) / len(data) * 100, 4),
        "label_column": label_column,
        "train_label_counts": _class_counts(train_df[label_column]),
        "test_label_counts": _class_counts(test_df[label_column]),
        "id_column": id_column if id_column in data.columns else None,
        "random_seed": int(random_seed),
        "stratified": True,
        "preprocessing_applied": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split the original raw dataset into stratified train/test CSVs."
    )
    parser.add_argument(
        "--train-percent",
        type=int,
        default=80,
        help="Training percentage (default: 80).",
    )
    parser.add_argument(
        "--test-percent",
        type=int,
        default=None,
        help="Test percentage (default: 100 minus train percentage).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Source raw CSV (default: {DEFAULT_SOURCE}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: dataset/split_<train>_<test>).",
    )
    parser.add_argument("--label-column", default="label", help="Label column name.")
    parser.add_argument("--id-column", default="patient_id", help="Patient ID column name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing train/test/metadata files in the output directory.",
    )
    args = parser.parse_args()

    test_percent = (
        args.test_percent
        if args.test_percent is not None
        else 100 - args.train_percent
    )
    output_dir = args.output_dir or (
        PROJECT_ROOT / "dataset" / f"split_{args.train_percent}_{test_percent}"
    )
    source = args.source if args.source.is_absolute() else PROJECT_ROOT / args.source
    output_dir = output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir

    metadata = split_dataset(
        source=source,
        output_dir=output_dir,
        train_percent=args.train_percent,
        test_percent=test_percent,
        label_column=args.label_column,
        id_column=args.id_column,
        random_seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"Saved raw train split: {output_dir / metadata['train_csv']} ({metadata['train_rows']} rows)")
    print(f"Saved raw test split:  {output_dir / metadata['test_csv']} ({metadata['test_rows']} rows)")
    print(f"Saved split metadata:  {output_dir / 'split_metadata.json'}")
    print(f"Train label counts: {metadata['train_label_counts']}")
    print(f"Test label counts:  {metadata['test_label_counts']}")


if __name__ == "__main__":
    main()
