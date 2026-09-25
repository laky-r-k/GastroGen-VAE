import os
import sys
import torch
import numpy as np
import pandas as pd

# Ensure we can import from the root module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_handling.utils import load_raw_data # Adjust if you have a different loader
from model_src import TabularVAE, build_gradient_boosting
from model_src import VAETrainer
from evaluation import run_classification_evaluation, run_vae_evaluation

# ---------------------------------------------------------
# Wrapper class to make the VAE + GB pipeline behave like a normal Sklearn model
# ---------------------------------------------------------
class VAEGradientBoostingWrapper:
    def __init__(self, vae_model, gb_model):
        self.vae = vae_model
        self.gb = gb_model

    def predict(self, X):
        # Extract compressed representation first
        self.vae.eval()
        with torch.no_grad():
            Z = self.vae.get_latent(torch.tensor(X, dtype=torch.float32), deterministic=True).cpu().numpy()
        # Predict using Gradient Boosting
        return self.gb.predict(Z)

    def predict_proba(self, X):
        self.vae.eval()
        with torch.no_grad():
            Z = self.vae.get_latent(torch.tensor(X, dtype=torch.float32), deterministic=True).cpu().numpy()
        return self.gb.predict_proba(Z)

def main():
    results_dir = os.path.join(os.path.dirname(__file__), "experiment1_results")
    
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
    
    # We use the VAETrainer to handle the PyTorch training loop
    trainer = VAETrainer(vae, lr=1e-3, beta=0.005)
    # Note: we use fit() - assuming it handles the epochs and batching natively
    trainer.fit(X_train, epochs=100, batch_size=64, verbose=True)
    # Optional: run standard VAE evaluations (loss curves, R2, t-SNE)
    run_vae_evaluation(vae, X_test, y_test, feature_cols, trainer.history, results_dir)
    
    # 3. Extract encoding from VAE and train VAE + Gradient Boosting
    print("Extracting latent representations...")
    vae.eval()
    with torch.no_grad():
        Z_train = vae.get_latent(torch.tensor(X_train, dtype=torch.float32), deterministic=True).cpu().numpy()
        
    print("Training VAE + Gradient Boosting model...")
    gb_vae_core = build_gradient_boosting(random_state=42)
    gb_vae_core.fit(Z_train, y_train)
    
    # Wrap it so it can be evaluated easily
    vae_gb_pipeline = VAEGradientBoostingWrapper(vae, gb_vae_core)

    # 4. Train Standalone Gradient Boosting Model
    print("Training Baseline Gradient Boosting model (Original Features)...")
    gb_baseline = build_gradient_boosting(random_state=42)
    gb_baseline.fit(X_train, y_train)

    # 5. Evaluate the two models and compare the results
    models_to_evaluate = {
        "VAE + Gradient Boosting": vae_gb_pipeline,
        "Baseline Gradient Boosting": gb_baseline
    }
    
    # 6. Save the results in experiment1_results folder
    # This orchestrator will calculate all metrics, generate ROC/PR curves, and save to JSON
    metrics = run_classification_evaluation(
        models=models_to_evaluate,
        X_test=X_test,
        y_test=y_test,
        results_dir=results_dir,
        feature_names=feature_cols
    )
    
    print("Experiment 1 complete! Results saved in:", results_dir)

if __name__ == "__main__":
    main()
