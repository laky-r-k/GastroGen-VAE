# VAE Feature Compression & Ensemble Classification Pipeline

**Directory:** `VAE/src/`  
**Purpose:** Implementation of a **Tabular Variational Autoencoder (VAE)** for non-linear clinical feature compression into a regularized latent space ($z \in \mathbb{R}^8$), followed by downstream classification using **Random Forest**, **AdaBoost**, and **Hybrid Ensembles (Soft Voting & Stacking)**.

---

## 1. Modular Architecture

| Source File | Role / Responsibility |
| :--- | :--- |
| [`data_loader.py`](file:///home/laky/Desktop/proj/major_project/VAE/src/data_loader.py) | Auto-detects and loads dataset across local paths and Google Colab environments. |
| [`vae_preprocessing.py`](file:///home/laky/Desktop/proj/major_project/VAE/src/vae_preprocessing.py) | Cleans data entry anomalies, applies $\log(1+x)$ transformations, derives clinical ratios, and fits standard/robust scaling. |
| [`vae_model.py`](file:///home/laky/Desktop/proj/major_project/VAE/src/vae_model.py) | PyTorch `TabularVAE` neural network (Encoder, Reparameterization Trick, Decoder, and $\beta$-VAE loss function). |
| [`vae_trainer.py`](file:///home/laky/Desktop/proj/major_project/VAE/src/vae_trainer.py) | Manages mini-batch training loop, `ReduceLROnPlateau` scheduler, early stopping, and per-feature $R^2$ reconstruction computation. |
| [`downstream_classifiers.py`](file:///home/laky/Desktop/proj/major_project/VAE/src/downstream_classifiers.py) | Builds Random Forest, AdaBoost, Soft Voting, and Stacking classifiers on latent and hybrid representations. |
| [`vae_evaluate.py`](file:///home/laky/Desktop/proj/major_project/VAE/src/vae_evaluate.py) | Generates VAE training loss curves, latent space 2D t-SNE/PCA projections, feature reconstruction fidelity bar charts, ROC curves, PR curves, and confusion matrices. |
| [`train_pipeline.py`](file:///home/laky/Desktop/proj/major_project/VAE/src/train_pipeline.py) | Stratified 5-Fold Cross-Validation, out-of-fold metric evaluation, production retraining on 100% data, and model serialization. |
| [`main.py`](file:///home/laky/Desktop/proj/major_project/VAE/src/main.py) | CLI entry point for executing the entire pipeline with customizable hyperparameters. |

---

## 2. VAE Architectural Design

### A. Encoder Network ($q_\phi(z|x)$)
- **Input Dimension:** $D = 24$ normalized clinical features.
- **Layers:**
  - $\text{Linear}(24 \rightarrow 128) \rightarrow \text{BatchNorm1d} \rightarrow \text{LeakyReLU}(0.2) \rightarrow \text{Dropout}(0.1)$
  - $\text{Linear}(128 \rightarrow 64) \rightarrow \text{BatchNorm1d} \rightarrow \text{LeakyReLU}(0.2) \rightarrow \text{Dropout}(0.1)$
- **Latent Parameters:**
  - $\mu = \text{Linear}(64 \rightarrow z_{\text{dim}}=8)$
  - $\log \sigma^2 = \text{Linear}(64 \rightarrow z_{\text{dim}}=8)$

### B. Reparameterization Trick
$$z = \mu + \sigma \odot \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, I)$$

### C. Decoder Network ($p_\theta(x|z)$)
- **Layers:**
  - $\text{Linear}(8 \rightarrow 64) \rightarrow \text{BatchNorm1d} \rightarrow \text{LeakyReLU}(0.2) \rightarrow \text{Dropout}(0.1)$
  - $\text{Linear}(64 \rightarrow 128) \rightarrow \text{BatchNorm1d} \rightarrow \text{LeakyReLU}(0.2) \rightarrow \text{Dropout}(0.1)$
  - $\text{Linear}(128 \rightarrow 24) \rightarrow \hat{X}$ (Reconstructed feature matrix)

### D. Loss Formulation ($\beta$-VAE)
$$\mathcal{L}(\theta, \phi; x) = \frac{1}{D} \sum_{j=1}^D (x_j - \hat{x}_j)^2 + \beta \cdot \left( -\frac{1}{2} \sum_{k=1}^K (1 + \log \sigma_k^2 - \mu_k^2 - \sigma_k^2) \right)$$
*(We set $\beta = 0.005$ to balance sharp biomarker reconstruction with smooth, well-regularized latent geometry).*

---

## 3. Downstream Ensemble Integration

The VAE Encoder compresses the 24-dimensional feature vector into a compact 8-dimensional latent vector $z = \mu(x)$. We evaluate three representations:
1. **Pure Latent Representation ($z \in \mathbb{R}^8$):** Evaluates whether the compressed manifold alone provides sufficient discriminative signal for tree splitting.
2. **Hybrid Representation ($[X_{\text{features}}, z] \in \mathbb{R}^{32}$):** Combines raw clinical biomarker thresholds with global non-linear latent manifolds.
3. **Downstream Models Evaluated:**
   - Random Forest (250 trees, max depth 6, balanced class weights)
   - AdaBoost (150 estimators, depth 2 decision stumps, learning rate 0.05)
   - Soft Voting Ensemble (blended predicted probabilities)
   - Stacking Classifier (RF + AdaBoost Level-0 learners, Logistic Regression meta-learner)

---

## 4. Execution Instructions

### Running Locally:
```bash
# From workspace root
python3 VAE/src/main.py

# Customizing parameters:
python3 VAE/src/main.py --latent_dim 8 --epochs 150 --beta 0.005 --n_splits 5
```

### Running in Google Colab:
```python
# In a Google Colab notebook cell:
!python VAE/src/main.py --data_path /content/major_project/dataset/gastric_cancer_dataset_raw.csv
```

---

## 5. Output Artifacts

### Models Saved in [`VAE/model/`](file:///home/laky/Desktop/proj/major_project/VAE/model)
- `vae_model.pt`: Serialized PyTorch VAE checkpoint (weights & architecture).
- `vae_preprocessing_pipeline.joblib`: Serialized preprocessing & scaler pipeline.
- `latent_rf_model.joblib`: Random Forest trained on compressed latent space $z$.
- `latent_adaboost_model.joblib`: AdaBoost trained on compressed latent space $z$.
- `latent_soft_voting_model.joblib`: Soft Voting Ensemble trained on latent space $z$.
- `latent_stacking_model.joblib`: Stacking Classifier trained on latent space $z$.
- `hybrid_soft_voting_model.joblib`: Soft Voting Ensemble trained on hybrid $[X, z]$.
- `hybrid_stacking_model.joblib`: Stacking Classifier trained on hybrid $[X, z]$.

### Results Saved in [`results/VAE/`](file:///home/laky/Desktop/proj/major_project/results/VAE)
- `vae_training_loss.png`: Training & validation loss dynamics (Recon & KL).
- `latent_space_tsne_pca.png`: 2D t-SNE & PCA visualization of the compressed latent space.
- `feature_reconstruction_fidelity.png`: Feature-by-feature $R^2$ reconstruction fidelity ranking.
- `roc_curves_comparison.png`: ROC curves comparing Latent & Hybrid models.
- `precision_recall_curves.png`: Precision-Recall curves.
- `confusion_matrices.png`: Confusion matrices.
- `vae_ensemble_metrics_summary.json`: Comprehensive out-of-fold performance summary.
- `vae_cv_fold_scores.csv`: Fold-by-fold cross-validation metrics.
- `vae_reconstruction_r2.csv`: Feature reconstruction $R^2$ scores.
