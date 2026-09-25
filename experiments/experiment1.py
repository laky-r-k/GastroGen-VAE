import os
import sys
import torch
import numpy as np
import pandas as pd

# Ensure we can import from the root module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_handling.utils import load_raw_data
from model_src import TabularVAE, VAETrainer, VAEClassifierPipeline, build_gradient_boosting
from evaluation import run_classification_evaluation, run_vae_evaluation


def main():
    base_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "experiment1")

    # 1. Load processed dataset
    print("Loading datasets...")
    train_df = pd.read_csv("dataset/processed_dataset/processed_train.csv")
    test_df = pd.read_csv("dataset/processed_dataset/processed_test.csv")

    # Separate features and labels
    feature_cols = [c for c in train_df.columns if c != "label"]
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    # 2. Train VAE
    print("Initializing and training VAE...")
    in_features = X_train.shape[1]
    vae = TabularVAE(in_features=in_features, latent_dim=8)

    trainer = VAETrainer(vae, lr=1e-3, beta=0.005)
    trainer.fit(X_train, epochs=100, batch_size=64, verbose=True)

    # VAE Diagnostics → results/experiment1/vae_diagnostics/
    run_vae_evaluation(
        vae_model=vae,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_cols,
        history=trainer.history,
        results_dir=os.path.join(base_results_dir, "vae_diagnostics")
    )

    # 3. Extract encoding from VAE and train VAE + Gradient Boosting
    print("Extracting latent representations...")
    vae.eval()
    with torch.no_grad():
        Z_train = vae.get_latent(torch.tensor(X_train, dtype=torch.float32), deterministic=True).cpu().numpy()

    print("Training VAE + Gradient Boosting model...")
    gb_vae_core = build_gradient_boosting(random_state=42)
    gb_vae_core.fit(Z_train, y_train)
    vae_gb_pipeline = VAEClassifierPipeline(vae, gb_vae_core)

    # VAE + GB Results → results/experiment1/vae_gradient_boosting/
    run_classification_evaluation(
        models={"VAE + Gradient Boosting": vae_gb_pipeline},
        X_test=X_test,
        y_test=y_test,
        results_dir=os.path.join(base_results_dir, "vae_gradient_boosting"),
        feature_names=feature_cols
    )

    # 4. Train Standalone Gradient Boosting Model (Baseline)
    print("Training Baseline Gradient Boosting model (Original Features)...")
    gb_baseline = build_gradient_boosting(random_state=42)
    gb_baseline.fit(X_train, y_train)

    # Baseline GB Results → results/experiment1/gradient_boosting/
    run_classification_evaluation(
        models={"Gradient Boosting": gb_baseline},
        X_test=X_test,
        y_test=y_test,
        results_dir=os.path.join(base_results_dir, "gradient_boosting"),
        feature_names=feature_cols
    )

    print("\nExperiment 1 complete! Results saved in:", base_results_dir)


if __name__ == "__main__":
    main()
