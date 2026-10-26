"""
Semi-Supervised VAE Model Trainer Module.
Handles mini-batch training loop, joint multi-task loss optimization,
learning rate scheduling, and performance tracking without embedded evaluation logic.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from torch.utils.data import TensorDataset, DataLoader

from .ss_vae_model import SemiSupervisedTabularVAE


class SemiSupervisedVAETrainer:
    """
    Manages SS-TabularVAE training lifecycle, mini-batch updates, and history tracking.
    Jointly optimizes reconstruction loss (generative) and classification loss (supervised).
    """

    def __init__(
        self,
        model: SemiSupervisedTabularVAE,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        beta: float = 0.005,
        alpha: float = 10.0,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.beta = beta
        self.alpha = alpha

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-5
        )

        self.history = {
            "epoch": [],
            "total_loss": [],
            "recon_loss": [],
            "kl_loss": [],
            "class_loss": [],
            "train_accuracy": [],
        }

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss_accum = 0.0
        recon_loss_accum = 0.0
        kl_loss_accum = 0.0
        class_loss_accum = 0.0
        correct_predictions = 0
        total_labeled = 0

        n_batches = len(dataloader)

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()

            x_recon, mu, logvar, y_logits = self.model(x_batch)

            loss_dict = self.model.loss_function(
                x_recon=x_recon,
                x=x_batch,
                mu=mu,
                logvar=logvar,
                y_logits=y_logits,
                y_true=y_batch,
                beta=self.beta,
                alpha=self.alpha,
            )

            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss_accum += loss_dict["loss"].item()
            recon_loss_accum += loss_dict["recon_loss"].item()
            kl_loss_accum += loss_dict["kl_loss"].item()
            class_loss_accum += loss_dict["class_loss"].item()

            # Track training accuracy on labeled samples
            labeled_mask = (y_batch >= 0)
            if labeled_mask.sum() > 0:
                probs = torch.sigmoid(y_logits[labeled_mask])
                preds = (probs >= 0.5).long()
                correct_predictions += (preds == y_batch[labeled_mask].long()).sum().item()
                total_labeled += labeled_mask.sum().item()

        accuracy = correct_predictions / max(1, total_labeled)

        return {
            "total_loss": total_loss_accum / n_batches,
            "recon_loss": recon_loss_accum / n_batches,
            "kl_loss": kl_loss_accum / n_batches,
            "class_loss": class_loss_accum / n_batches,
            "accuracy": accuracy,
        }

    def fit(
        self,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Fits SemiSupervisedTabularVAE on training features and optional labels.
        Unlabeled rows can be indicated by label=-1 or y_train=None.
        """
        x_tensor = torch.tensor(X_train, dtype=torch.float32)

        if y_train is not None:
            y_tensor = torch.tensor(y_train, dtype=torch.float32)
        else:
            y_tensor = torch.full((len(X_train),), -1.0, dtype=torch.float32)

        dataset = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(1, epochs + 1):
            epoch_metrics = self.train_epoch(dataloader)
            self.scheduler.step(epoch_metrics["total_loss"])

            self.history["epoch"].append(epoch)
            self.history["total_loss"].append(epoch_metrics["total_loss"])
            self.history["recon_loss"].append(epoch_metrics["recon_loss"])
            self.history["kl_loss"].append(epoch_metrics["kl_loss"])
            self.history["class_loss"].append(epoch_metrics["class_loss"])
            self.history["train_accuracy"].append(epoch_metrics["accuracy"])

            if verbose and (epoch % 20 == 0 or epoch == epochs):
                print(
                    f"  Epoch [{epoch:3d}/{epochs}] | Loss: {epoch_metrics['total_loss']:.4f} "
                    f"(Recon: {epoch_metrics['recon_loss']:.4f}, KL: {epoch_metrics['kl_loss']:.4f}, "
                    f"Class: {epoch_metrics['class_loss']:.4f}) | Train Acc: {epoch_metrics['accuracy']:.4f}"
                )

        return self.history
