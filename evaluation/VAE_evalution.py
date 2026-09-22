import os
import pandas as pd
from typing import Dict, List, Any

from .visualization import (
    plot_vae_loss_curves,
    plot_latent_space_tsne,
    plot_reconstruction_r2
)
from .reporting import save_vae_results_summary

def run_vae_evaluation(
    history: Dict[str, List[float]],
    Z_latent,
    y_test,
    r2_df: pd.DataFrame,
    overall_metrics: Dict[str, Dict[str, float]],
    cv_fold_df: pd.DataFrame,
    results_dir: str
) -> None:
    """
    High-level orchestrator for evaluating a trained VAE.
    Handles generating loss curves, latent space projections, R2 charts, and saving results.
    """
    print(f"\n[VAE Evaluation] Starting VAE diagnostic evaluation. Output directory: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)
    
    print("[VAE Evaluation] Generating diagnostic plots...")
    
    # 1. Plot training loss history
    if history and len(history.get("epoch", [])) > 0:
        plot_vae_loss_curves(
            history,
            os.path.join(results_dir, "vae_training_loss.png")
        )
        
    # 2. Plot latent space clusters using t-SNE and PCA
    if Z_latent is not None and y_test is not None:
        plot_latent_space_tsne(
            Z_latent,
            y_test,
            os.path.join(results_dir, "vae_latent_space.png")
        )
        
    # 3. Plot Reconstruction R2 fidelity
    if not r2_df.empty:
        plot_reconstruction_r2(
            r2_df,
            os.path.join(results_dir, "vae_reconstruction_r2.png")
        )
        
    # 4. Save results summary
    print("[VAE Evaluation] Saving metrics and summaries...")
    save_vae_results_summary(
        all_metrics=overall_metrics,
        cv_fold_df=cv_fold_df,
        r2_df=r2_df,
        results_dir=results_dir
    )
    
    print("[VAE Evaluation] VAE evaluation complete.\n")
