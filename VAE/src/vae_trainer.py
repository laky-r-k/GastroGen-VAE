"""
VAE Model Trainer Module.
Handles PyTorch DataLoader creation, mini-batch training loop, learning rate scheduling,
and feature reconstruction fidelity evaluation.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score

from vae_model import TabularVAE


class VAETrainer:
    """
    Manages VAE training lifecycle and performance metrics tracking.
    """

    def __init__(
        self,
        model: TabularVAE,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        beta: float = 0.005,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.beta = beta
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-5
        )
        self.history = {
            "epoch": [],
            "train_loss": [],
            "train_recon": [],
            "train_kl": [],
            "val_loss": [],
            "val_recon": [],
            "val_kl": [],
        }

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Executes one training epoch across mini-batches.
        """
        self.model.train()
        total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
        n_batches = len(dataloader)

        for (x_batch,) in dataloader:
            x_batch = x_batch.to(self.device)
            self.optimizer.zero_grad()

            x_recon, mu, logvar = self.model(x_batch)
            loss_dict = self.model.loss_function(x_recon, x_batch, mu, logvar, beta=self.beta)

            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss_dict["loss"].item()
            total_recon += loss_dict["recon_loss"].item()
            total_kl += loss_dict["kl_loss"].item()

        return {
            "loss": total_loss / n_batches,
            "recon_loss": total_recon / n_batches,
            "kl_loss": total_kl / n_batches,
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluates VAE on validation/test DataLoader.
        """
        self.model.eval()
        total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
        n_batches = len(dataloader)

        with torch.no_grad():
            for (x_batch,) in dataloader:
                x_batch = x_batch.to(self.device)
                x_recon, mu, logvar = self.model(x_batch)
                loss_dict = self.model.loss_function(x_recon, x_batch, mu, logvar, beta=self.beta)

                total_loss += loss_dict["loss"].item()
                total_recon += loss_dict["recon_loss"].item()
                total_kl += loss_dict["kl_loss"].item()

        return {
            "loss": total_loss / max(1, n_batches),
            "recon_loss": total_recon / max(1, n_batches),
            "kl_loss": total_kl / max(1, n_batches),
        }

    def fit(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        epochs: int = 150,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Fits VAE with early stopping and loss logging.
        """
        train_tensor = torch.tensor(X_train, dtype=torch.float32)
        train_loader = DataLoader(
            TensorDataset(train_tensor), batch_size=batch_size, shuffle=True
        )

        val_loader = None
        if X_val is not None:
            val_tensor = torch.tensor(X_val, dtype=torch.float32)
            val_loader = DataLoader(
                TensorDataset(val_tensor), batch_size=batch_size, shuffle=False
            )

        best_val_loss = float("inf")
        best_weights = None
        patience = 30
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                self.scheduler.step(val_metrics["loss"])

                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                val_loss_val = val_metrics["loss"]
                val_recon_val = val_metrics["recon_loss"]
                val_kl_val = val_metrics["kl_loss"]
            else:
                self.scheduler.step(train_metrics["loss"])
                val_loss_val = train_metrics["loss"]
                val_recon_val = train_metrics["recon_loss"]
                val_kl_val = train_metrics["kl_loss"]

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_recon"].append(train_metrics["recon_loss"])
            self.history["train_kl"].append(train_metrics["kl_loss"])
            self.history["val_loss"].append(val_loss_val)
            self.history["val_recon"].append(val_recon_val)
            self.history["val_kl"].append(val_kl_val)

            if verbose and (epoch % 25 == 0 or epoch == epochs):
                print(
                    f"  Epoch [{epoch:3d}/{epochs}] | Train Loss: {train_metrics['loss']:.4f} (Recon: {train_metrics['recon_loss']:.4f}, KL: {train_metrics['kl_loss']:.4f}) | Val Loss: {val_loss_val:.4f}"
                )

            if val_loader is not None and patience_counter >= patience:
                if verbose:
                    print(f"  [EarlyStopping] Triggered at epoch {epoch} (Best Val Loss: {best_val_loss:.4f})")
                break

        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        return self.history

    def compute_reconstruction_r2(
        self, X_input: np.ndarray, feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Computes per-feature R2 reconstruction scores to measure how well each biomarker is preserved.
        """
        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(X_input, dtype=torch.float32).to(self.device)
            x_recon, _, _ = self.model(x_t)
            X_rec = x_recon.cpu().numpy()

        r2_scores = []
        for i, feat in enumerate(feature_names):
            score = r2_score(X_input[:, i], X_rec[:, i])
            mse = np.mean((X_input[:, i] - X_rec[:, i]) ** 2)
            r2_scores.append({"feature": feat, "r2_score": max(0.0, score), "mse": mse})

        return pd.DataFrame(r2_scores).sort_values(by="r2_score", ascending=False)
