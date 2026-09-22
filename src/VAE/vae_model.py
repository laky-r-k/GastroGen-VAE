"""
Tabular Variational Autoencoder (VAE) Architecture in PyTorch.
Compresses multi-dimensional clinical features into a low-dimensional regularized latent manifold.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional


class TabularVAE(nn.Module):
    """
    Variational Autoencoder tailored for Tabular Clinical Biomarkers.
    """

    def __init__(
        self,
        in_features: int,
        latent_dim: int = 8,
        hidden_dim1: int = 128,
        hidden_dim2: int = 64,
        dropout_rate: float = 0.1,
    ):
        super(TabularVAE, self).__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim

        # --- ENCODER NETWORK ---
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

        # Latent Gaussian projections
        self.fc_mu = nn.Linear(hidden_dim2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim2, latent_dim)

        # --- DECODER NETWORK ---
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

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Maps input features to latent Gaussian distribution parameters (mu, logvar).
        """
        h = self.encoder_backbone(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Applies reparameterization trick: z = mu + sigma * eps, eps ~ N(0, I)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs original feature space from latent representation z.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full VAE forward pass. Returns (reconstruction, mu, logvar).
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def get_latent(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Extracts compressed latent space features.
        If deterministic=True, returns mean vector mu. Otherwise samples z.
        """
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
        beta: float = 0.005,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes beta-VAE loss = Reconstruction Loss (MSE) + beta * KL Divergence.
        """
        # Reconstruction Loss (Mean Squared Error across batch and features)
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")

        # Analytical KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_div = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        total_loss = recon_loss + beta * kl_div

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_div,
        }

    def save_model(self, filepath: str) -> None:
        """
        Saves model weights and architecture configuration.
        """
        torch.save(
            {
                "state_dict": self.state_dict(),
                "in_features": self.in_features,
                "latent_dim": self.latent_dim,
            },
            filepath,
        )

    @classmethod
    def load_model(cls, filepath: str, device: str = "cpu") -> "TabularVAE":
        """
        Loads saved model from checkpoint file.
        """
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(
            in_features=checkpoint["in_features"],
            latent_dim=checkpoint["latent_dim"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model
