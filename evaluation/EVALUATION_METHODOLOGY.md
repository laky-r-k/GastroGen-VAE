# Evaluation Methodology

This document outlines the comprehensive evaluation framework used to assess model performance across the classification pipelines (Random Forest, AdaBoost, Ensembles) and the Variational Autoencoder (VAE) in this project. 

## 1. Classification Metrics (Testing & Inference Phase)

To rigorously evaluate the predictive performance of the classifiers, the pipeline calculates a diverse set of statistical metrics. These metrics are computed using the test holdout set (or across cross-validation folds).

### Core Predictive Metrics
* **ROC AUC (Receiver Operating Characteristic - Area Under Curve)**: Measures the model's ability to discriminate between the Gastric Cancer class and the Control class across all possible classification thresholds.
* **PR AUC (Precision-Recall Area Under Curve)**: Assessed via the Average Precision score. This is particularly valuable for understanding performance on imbalanced datasets, focusing heavily on the positive (Cancer) class.
* **Accuracy**: The raw percentage of correctly classified instances.
* **F1 Score**: The harmonic mean of Precision and Recall, providing a balanced metric that penalizes extreme disparities between the two.
* **Precision (Positive Predictive Value)**: The proportion of positive predictions that were actually correct (minimizing False Positives).
* **Recall / Sensitivity (True Positive Rate)**: The proportion of actual positive cases that were successfully identified (minimizing False Negatives).
* **Specificity (True Negative Rate)**: The proportion of actual negative cases (Control) that were correctly identified. Calculated as $TN / (TN + FP)$.
* **Brier Score Loss**: Evaluates the calibration of the model's predicted probabilities. A lower Brier score indicates that the predicted probabilities align closely with the actual outcomes.

### Confusion Matrix Analytics
The raw prediction outcomes are extracted into four components to provide granular insight into misclassifications:
* **True Positives (TP)**
* **True Negatives (TN)**
* **False Positives (FP)**
* **False Negatives (FN)**

---

## 2. VAE-Specific Diagnostics (Training & Compression)

The Variational Autoencoder requires specialized evaluation to ensure that it is learning a meaningful, regularized, and reconstructable latent representation.

### Training Stability & Convergence
* **Total Loss Tracking**: The aggregate $\beta$-VAE loss is tracked across epochs for both training and validation sets to monitor convergence and detect overfitting.
* **Reconstruction Loss**: Measures the Mean Squared Error (MSE) between the original input features and the VAE's reconstructed outputs.
* **KL Divergence**: Tracks the regularization penalty, ensuring the latent space conforms to a standard normal distribution ($\mathcal{N}(0, I)$) without collapsing.

### Latent Space Topography
* **t-SNE & PCA Projections**: The high-dimensional latent embeddings ($z$) are projected down to 2D space. This visualization verifies whether the VAE has naturally learned to cluster patients into distinct physiological profiles (e.g., separating Cancer vs. Control) purely through unsupervised reconstruction.

### Feature Reconstruction Fidelity
* **Reconstruction $R^2$ Score**: Measures how accurately the decoder can reconstruct each individual clinical biomarker from the compressed latent space. A high $R^2$ indicates that the critical variance of that feature is preserved in the latent embedding.

---

## 3. Visual Diagnostics Suite

The evaluation pipelines automatically generate high-quality visualizations to aid in model interpretability:

1. **ROC & PR Curves**: Overlayed line plots comparing all trained models, demonstrating the trade-offs between sensitivity, specificity, and precision.
2. **Confusion Matrices**: Normalized, color-coded heatmaps showing the percentage distribution of True vs. Predicted diagnoses.
3. **Feature Importance Rankings**: Horizontal bar charts displaying the Mean Impurity Reduction (for tree-based models), highlighting the top most discriminative clinical biomarkers.
4. **VAE Loss Curves**: Multi-panel line plots tracking total loss, reconstruction loss, and KL divergence over the training lifecycle.
