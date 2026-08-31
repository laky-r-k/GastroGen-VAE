"""
Main Entry Point for Training Random Forest, AdaBoost, and Hybrid Ensembles.

Usage:
    python RF_addaboost/src/main.py
    python RF_addaboost/src/main.py --feature_mode optimized_subset --n_splits 5
"""

import os
import sys
import argparse

# Ensure src directory is on sys.path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_loader import load_raw_data 
from preprocessing import PreprocessingPipeline
from train import train_and_export_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Random Forest + AdaBoost Ensemble for Gastric Cancer Diagnosis"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to raw CSV dataset (default: auto-detect)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.path.join(os.path.dirname(SRC_DIR), "model"),
        help="Directory to save trained .joblib models",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(SRC_DIR)), "results", "RF_addaboost"),
        help="Directory to save metric reports and plots",
    )
    parser.add_argument(
        "--feature_mode",
        type=str,
        default="optimized_subset",
        choices=["optimized_subset", "all_engineered", "raw_cleaned"],
        help="Feature selection strategy to use",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of Stratified K-Fold splits",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("GASTRIC CANCER DETECTION: RANDOM FOREST + ADABOOST TRAINING PIPELINE")
    print("=" * 80)
    print(f"Feature Mode   : {args.feature_mode}")
    print(f"Model Output   : {args.model_dir}")
    print(f"Results Output : {args.results_dir}")
    print(f"CV Folds       : {args.n_splits}")
    print("=" * 80)

    # 1. Load Data
    raw_df = load_raw_data(args.data_path)


    # 2. Build Preprocessing Pipeline
    pipeline = PreprocessingPipeline(
        feature_mode=args.feature_mode,
        scaler_type="robust",
        apply_scaling=False,
    )

    # 3. Train, Evaluate, and Export
    results = train_and_export_pipeline(
        raw_df=raw_df,
        pipeline=pipeline,
        model_output_dir=args.model_dir,
        results_output_dir=args.results_dir,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )

    print("\n[Complete] Pipeline training, model serialization, and evaluation finished successfully!")


if __name__ == "__main__":
    main()
