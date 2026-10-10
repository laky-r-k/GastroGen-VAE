"""
Experiment 1: VAE Latent Space + Gradient Boosting vs Baseline Gradient Boosting.
Subclasses BaseExperiment to enable config-driven execution and versioned outputs.
"""

import os
import sys
import torch
import numpy as np

# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.base_experiment import BaseExperiment
from model_src import (
    TabularVAE,
    VAETrainer,
    VAEClassifierPipeline,
    build_gradient_boosting,
)
from evaluation import run_classification_evaluation, run_vae_evaluation


class Experiment1(BaseExperiment):
    """
    Evaluates whether non-linear dimensionality reduction via VAE improves
    or impacts Gradient Boosting classification performance compared to
    training directly on full clinical features.
    """

    def __init__(
        self,
        config_path: str = None,
        config: dict = None,
        output_dir: str = None,
        timestamped: bool = True,
    ):
        super().__init__(
            config_path=config_path,
            config=config,
            output_dir=output_dir,
            timestamped=timestamped,
        )

    def run(self):
        # 1. Load Dataset using BaseExperiment helper
        print("[1/5] Loading datasets...")
        train_df, test_df, feature_cols, y_train_series, y_test_series = self.load_data()

        X_train = train_df[feature_cols].values
        y_train = y_train_series.values
        X_test = test_df[feature_cols].values
        y_test = y_test_series.values

        in_features = X_train.shape[1]

        # 2. Extract Config Hyperparameters
        vae_cfg = self.config.get("vae", {})
        gb_cfg = self.config.get("gradient_boosting", {})

        # 3. Build & Train VAE
        print("[2/5] Initializing and training Tabular VAE...")
        vae = TabularVAE(
            in_features=in_features,
            latent_dim=vae_cfg.get("latent_dim", 8),
            hidden_dim1=vae_cfg.get("hidden_dim1", 128),
            hidden_dim2=vae_cfg.get("hidden_dim2", 64),
            dropout_rate=vae_cfg.get("dropout_rate", 0.1),
        )

        trainer = VAETrainer(
            model=vae,
            lr=vae_cfg.get("learning_rate", 1e-3),
            weight_decay=vae_cfg.get("weight_decay", 1e-5),
            beta=vae_cfg.get("beta", 0.005),
        )

        epochs = vae_cfg.get("epochs", 100)
        batch_size = vae_cfg.get("batch_size", 64)
        trainer.fit(X_train, epochs=epochs, batch_size=batch_size, verbose=True)

        # 4. Evaluate VAE Diagnostics
        print("[3/5] Running VAE diagnostic evaluation...")
        vae_diag_dir = self.get_results_subdir("vae_diagnostics")
        run_vae_evaluation(
            vae_model=vae,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_cols,
            history=trainer.history,
            results_dir=vae_diag_dir,
        )

        # 5. Extract Latent Representations & Train VAE + Gradient Boosting
        print("[4/5] Training VAE + Gradient Boosting pipeline...")
        vae.eval()
        with torch.no_grad():
            x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            Z_train = vae.get_latent(x_train_tensor, deterministic=True).cpu().numpy()

        gb_vae_core = build_gradient_boosting(
            n_estimators=gb_cfg.get("n_estimators", 200),
            learning_rate=gb_cfg.get("learning_rate", 0.1),
            max_depth=gb_cfg.get("max_depth", 4),
            min_samples_split=gb_cfg.get("min_samples_split", 4),
            min_samples_leaf=gb_cfg.get("min_samples_leaf", 2),
            max_features=gb_cfg.get("max_features", "sqrt"),
            random_state=gb_cfg.get("random_state", 42),
        )
        gb_vae_core.fit(Z_train, y_train)
        vae_gb_pipeline = VAEClassifierPipeline(vae, gb_vae_core)

        vae_gb_dir = self.get_results_subdir("vae_gradient_boosting")
        run_classification_evaluation(
            models={"VAE + Gradient Boosting": vae_gb_pipeline},
            X_test=X_test,
            y_test=y_test,
            results_dir=vae_gb_dir,
            feature_names=feature_cols,
        )

        # 6. Train Baseline Gradient Boosting (Raw Features)
        print("[5/5] Training Baseline Gradient Boosting model (Raw Features)...")
        gb_baseline = build_gradient_boosting(
            n_estimators=gb_cfg.get("n_estimators", 200),
            learning_rate=gb_cfg.get("learning_rate", 0.1),
            max_depth=gb_cfg.get("max_depth", 4),
            min_samples_split=gb_cfg.get("min_samples_split", 4),
            min_samples_leaf=gb_cfg.get("min_samples_leaf", 2),
            max_features=gb_cfg.get("max_features", "sqrt"),
            random_state=gb_cfg.get("random_state", 42),
        )
        gb_baseline.fit(X_train, y_train)

        baseline_gb_dir = self.get_results_subdir("gradient_boosting")
        run_classification_evaluation(
            models={"Gradient Boosting": gb_baseline},
            X_test=X_test,
            y_test=y_test,
            results_dir=baseline_gb_dir,
            feature_names=feature_cols,
        )

        return {
            "vae_diagnostics_dir": vae_diag_dir,
            "vae_gradient_boosting_dir": vae_gb_dir,
            "gradient_boosting_dir": baseline_gb_dir,
        }


def main():
    exp = Experiment1()
    exp()


if __name__ == "__main__":
    main()
