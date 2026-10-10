"""
Conditional Wasserstein GAN with Gradient Penalty (CWGAN-GP) Model Definitions.
Tailored for tabular clinical biomarker synthesis under small-sample regimes.
"""

import torch
import torch.nn as nn
from typing import Optional


class Generator(nn.Module):
    """
    Conditional Generator MLP.
    Takes latent noise z concatenated with one-hot class label c,
    and maps to continuous tabular features.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        num_classes: int = 2,
        output_dim: int = 21,
        hidden_dim: int = 128,
        use_layer_norm: bool = True,
        use_batch_norm: bool = False,
    ):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.output_dim = output_dim

        in_dim = latent_dim + num_classes

        layers = []
        # Layer 1
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        elif use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Layer 2
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        elif use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Output Layer (linear activation for standardized/scaled continuous features)
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, labels_onehot: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent noise tensor of shape (batch_size, latent_dim)
            labels_onehot: One-hot encoded class labels of shape (batch_size, num_classes)
        Returns:
            Generated tabular features of shape (batch_size, output_dim)
        """
        inp = torch.cat([z, labels_onehot], dim=1)
        return self.net(inp)


class Critic(nn.Module):
    """
    Conditional Critic MLP for WGAN-GP.
    Takes tabular features concatenated with one-hot class label c,
    and outputs a scalar Wasserstein score.
    NEVER uses BatchNorm to satisfy the 1-Lipschitz gradient penalty constraint.
    """

    def __init__(
        self,
        input_dim: int = 21,
        num_classes: int = 2,
        hidden_dim: int = 128,
        use_layer_norm: bool = True,
    ):
        super(Critic, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        in_dim = input_dim + num_classes

        layers = []
        # Layer 1
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Layer 2
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Scalar output (no Sigmoid, unconstrained Wasserstein score)
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, labels_onehot: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tabular features tensor of shape (batch_size, input_dim)
            labels_onehot: One-hot encoded class labels of shape (batch_size, num_classes)
        Returns:
            Critic score of shape (batch_size, 1)
        """
        inp = torch.cat([x, labels_onehot], dim=1)
        return self.net(inp)
