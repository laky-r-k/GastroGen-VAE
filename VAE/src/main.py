"""
Main CLI Entry Point for VAE Compression + Random Forest + AdaBoost Pipeline.

Usage:
    python VAE/src/main.py
    python VAE/src/main.py --latent_dim 8 --epochs 150 --beta 0.005
"""

import os
import sys
import argparse

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_loader import load_raw_data
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from train_pipeline import train_vae_ensemble_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="VAE Feature Compression & RF + AdaBoost Ensemble Training Pipeline"
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
        help="Directory to save trained VAE and ensemble models",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(SRC_DIR)), "results", "VAE"),
        help="Directory to save metric reports and diagnostic plots",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=8,
        help="Dimensionality of compressed latent space z (default: 8)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of VAE training epochs (default: 150)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Mini-batch size for VAE training (default: 64)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.005,
        help="KL divergence weighting coefficient beta in VAE loss (default: 0.005)",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of Stratified K-Fold CV splits (default: 5)",
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
    print("GASTRIC CANCER DETECTION: VAE FEATURE COMPRESSION + RF & ADABOOST")
    print("=" * 80)
    print(f"Latent Dimension (z) : {args.latent_dim}")
    print(f"VAE Epochs           : {args.epochs}")
    print(f"Beta (KL weight)     : {args.beta}")
    print(f"Model Output Dir     : {args.model_dir}")
    print(f"Results Output Dir   : {args.results_dir}")
    print(f"CV Folds             : {args.n_splits}")
    print("=" * 80)

    # 1. Load Data
    raw_df = load_raw_data(args.data_path)

    # 2. Run Pipeline
    results = train_vae_ensemble_pipeline(
        raw_df=raw_df,
        model_output_dir=args.model_dir,
        results_output_dir=args.results_dir,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_splits=args.n_splits,
        beta=args.beta,
        random_state=args.random_state,
    )

    print("\n[Complete] VAE Feature Compression & Ensemble Training finished successfully!")


if __name__ == "__main__":
    main()
