"""
Experiments module for ML models and comparative studies.
"""

from .base_experiment import BaseExperiment
from .experiment1 import Experiment1
from .experiment2 import Experiment2
from .experiment_wgan import ExperimentWGAN

__all__ = ["BaseExperiment", "Experiment1", "Experiment2", "ExperimentWGAN"]
