import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any

from .metrics import compute_reconstruction_r2
from .visualization import (
    plot_vae_loss_curves,
    plot_latent_space_tsne,
    plot_reconstruction_r2
)
from .reporting import save_vae_results_summary

def run_vae_evaluation(
    vae_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    history: Dict[str, List[float]],
    results_dir: str,
    device: str = "cpu"
) -> None:
    """
    High-level orchestrator for evaluating a trained VAE.
    Automatically generates latent embeddings, reconstructs features, computes R2, and generates plots.
    """
    print(f"\n[VAE Evaluation] Starting VAE diagnostic evaluation. Output directory: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Forward Pass to get Latent Z and Reconstructions
    print("[VAE Evaluation] Extracting latent space and reconstructing test features...")
    vae_model.eval()
    vae_model.to(device)
    with torch.no_grad():
        x_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        x_recon_t, _, _ = vae_model(x_t)
        Z_latent_t = vae_model.get_latent(x_t, deterministic=True)
        
        X_recon = x_recon_t.cpu().numpy()
        Z_latent = Z_latent_t.cpu().numpy()

    # 2. Compute Reconstruction R2
    r2_df = compute_reconstruction_r2(X_test, X_recon, feature_names)

    # 3. Generate Plots
    print("[VAE Evaluation] Generating diagnostic plots...")
    
    if history and len(history.get("epoch", [])) > 0:
        plot_vae_loss_curves(
            history,
            os.path.join(results_dir, "vae_training_loss.png")
        )
        
    if Z_latent is not None and y_test is not None:
        plot_latent_space_tsne(
            Z_latent,
            y_test,
            os.path.join(results_dir, "vae_latent_space.png")
        )
        
    if not r2_df.empty:
        plot_reconstruction_r2(
            r2_df,
            os.path.join(results_dir, "vae_reconstruction_r2.png")
        )
        
    # 4. Save results summary
    print("[VAE Evaluation] Saving metrics and summaries...")
    
    # Dummy placeholder metrics if needed, as VAE doesn't produce standard classification metrics 
    # unless tied to downstream tasks (which happens in classification_evaluation).
    overall_metrics = {}
    cv_fold_df = pd.DataFrame()
    
    save_vae_results_summary(
        all_metrics=overall_metrics,
        cv_fold_df=cv_fold_df,
        r2_df=r2_df,
        results_dir=results_dir
    )
    
    print("[VAE Evaluation] VAE evaluation complete.\n")
