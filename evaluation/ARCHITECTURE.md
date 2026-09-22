# Evaluation Module Architecture

This document explains the software architecture and structural design of the `evaluation` module. This module was specifically refactored to adhere to the principle of **Separation of Concerns**, ensuring that mathematical calculations, data visualizations, and file I/O operations are kept strictly isolated from one another.

## Directory Structure
```text
evaluation/
├── __init__.py                  # Exposes all module functions for clean importing
├── metrics.py                   # Pure mathematical/statistical computations
├── visualization.py             # Plotting utilities (matplotlib code)
├── reporting.py                 # File I/O utilities (saving to JSON/CSV)
├── classification_evaluation.py # High-level orchestrator for classical ML models
└── VAE_evalution.py             # High-level orchestrator for VAE diagnostics
```

---

## Component Breakdown

### 1. `metrics.py`
**Responsibility:** Pure computational logic. 
* Contains `compute_metrics()`, which calculates core predictive statistics (AUC, F1, Accuracy, Brier Score, etc.) using `scikit-learn`.
* **Design Rule:** This file does not perform any plotting or file saving. It strictly takes in arrays and returns dictionaries of floats.

### 2. `visualization.py`
**Responsibility:** Graphical plotting and visual diagnostics.
* Contains all functions utilizing `matplotlib` and `seaborn`.
* Responsible for generating `plot_roc_curves`, `plot_precision_recall_curves`, `plot_confusion_matrices`, `plot_latent_space_tsne`, etc.
* **Design Rule:** Isolating the plotting code allows for easy modification of styling, colors, and formatting without risking the integrity of the underlying mathematical logic.

### 3. `reporting.py`
**Responsibility:** File System I/O operations.
* Contains `save_results_summary` and `save_vae_results_summary`.
* Responsible for safely dumping Python dictionaries to `.json` files and Pandas DataFrames to `.csv` files.

### 4. High-Level Orchestrators
To prevent downstream scripts (like the main training pipeline) from becoming bloated, the module exposes orchestrator wrappers:
* **`classification_evaluation.py` (`run_classification_evaluation`)**: 
  Takes a dictionary of trained classical ML models (e.g., Random Forest, AdaBoost), executes predictions, calls `metrics.py` to get stats, loops through `visualization.py` to generate the necessary curves and matrices, and finally triggers `reporting.py` to save the results.
* **`VAE_evalution.py` (`run_vae_evaluation`)**: 
  Coordinates the evaluation of the Variational Autoencoder. Handles the plotting of training loss histories, latent space projections (t-SNE/PCA), and feature reconstruction fidelity.

---

## Best Practices for Usage
Because of the `__init__.py` file, you do not need to worry about the internal file structure when using the module in external scripts. You can import any function directly from the root module namespace:

```python
# Import high-level orchestrators
from evaluation import run_classification_evaluation, run_vae_evaluation

# Or import specific utilities if needed
from evaluation import compute_metrics, plot_roc_curves
```
