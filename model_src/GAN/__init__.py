"""
GAN Module for Tabular Clinical Data Synthesis.
"""

from .wgan_gp_model import Generator, Critic
from .wgan_trainer import WGANGPTrainer

__all__ = ["Generator", "Critic", "WGANGPTrainer"]
