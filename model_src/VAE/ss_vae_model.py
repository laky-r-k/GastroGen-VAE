"""
Semi-Supervised Tabular Variational Autoencoder (SS-TabularVAE) Architecture in PyTorch.
Integrates a shared encoder with dual output heads:
  1. Generative Decoder Head (feature reconstruction via MSE + KL divergence)
  2. Classification MLP Head (cancer diagnosis via Binary Cross-Entropy)
Enables end-to-end joint gradient backpropagation into the latent manifold.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional


class SemiSupervisedTabularVAE(nn.Module):
    """
    Semi-Supervised Variational Autoencoder for Tabular Clinical Biomarkers.
    Jointly optimizes reconstruction loss (generative) and classification loss (supervised).
    """

    def __init__(
        self,
        in_features: int,
        latent_dim: int = 8,
        hidden_dim1: int = 128,
        hidden_dim2: int = 64,
        classifier_hidden: int = 32,
        dropout_rate: float = 0.1,
    ):
        super(SemiSupervisedTabularVAE, self).__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim

        # --- 1. SHARED ENCODER BACKBONE ---
        self.encoder_backbone = nn.Sequential(
            nn.Linear(in_features, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
        )

        # Latent Gaussian parameter projections
        self.fc_mu = nn.Linear(hidden_dim2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim2, latent_dim)

        # --- 2. GENERATIVE DECODER BACKBONE ---
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, in_features),
        )

        # --- 3. CLASSIFICATION HEAD (MLP) ---
        self.classifier_head = nn.Sequential(
            nn.Linear(latent_dim, classifier_hidden),
            nn.BatchNorm1d(classifier_hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_hidden, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            # Output raw logit (for numerical stability with BCEWithLogitsLoss)
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps input clinical features to latent Gaussian parameters (mu, logvar)."""
        h = self.encoder_backbone(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Applies reparameterization trick: z = mu + sigma * eps, eps ~ N(0, I)"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstructs clinical feature space from latent representation z."""
        return self.decoder(z)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        """Predicts classification logit from latent representation z."""
        return self.classifier_head(z).squeeze(-1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass through shared encoder, decoder, and classifier head.
        Returns:
            x_recon: Reconstructed features
            mu: Latent mean
            logvar: Latent log variance
            y_logits: Predicted logits for classification
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        y_logits = self.classify(mu if not self.training else z)
        return x_recon, mu, logvar, y_logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Outputs calibrated predicted probabilities P(Cancer | x) in [0, 1]."""
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
            logits = self.classify(mu)
            return torch.sigmoid(logits)

    def get_latent(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Extracts cancer-guided latent vectors z for downstream tree models."""
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(x)
            if deterministic:
                return mu
            return self.reparameterize(mu, logvar)

    def loss_function(
        self,
        x_recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        y_logits: torch.Tensor,
        y_true: Optional[torch.Tensor] = None,
        beta: float = 0.005,
        alpha: float = 10.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes joint semi-supervised multi-task loss:
            L_total = L_recon + beta * L_kl + alpha * L_bce
        """
        # 1. Generative Reconstruction Loss (MSE across batch and features)
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")

        # 2. Analytical KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_div = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        # 3. Supervised Classification Loss (BCEWithLogits)
        if y_true is not None:
            # Mask out any unlabeled rows if label == -1
            labeled_mask = (y_true >= 0).float()
            if labeled_mask.sum() > 0:
                bce_raw = F.binary_cross_entropy_with_logits(
                    y_logits, y_true.float(), reduction="none"
                )
                class_loss = (bce_raw * labeled_mask).sum() / (labeled_mask.sum() + 1e-8)
            else:
                class_loss = torch.tensor(0.0, device=x.device)
        else:
            class_loss = torch.tensor(0.0, device=x.device)

        total_loss = recon_loss + (beta * kl_div) + (alpha * class_loss)

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_div,
            "class_loss": class_loss,
        }
