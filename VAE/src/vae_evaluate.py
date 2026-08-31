"""
Evaluation and Diagnostic Visualization Module for VAE and Downstream Classifiers.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    brier_score_loss,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Computes a complete suite of classification metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall_sensitivity": float(recall_score(y_true, y_pred)),
        "specificity": float(specificity),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def plot_vae_loss_curves(history: Dict[str, List[float]], output_path: str) -> None:
    """
    Plots Total Loss, Reconstruction Loss, and KL Divergence over epochs.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)

    # 1. Total Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#2b5c8f", lw=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#d95f02", lw=2, linestyle="--")
    axes[0].set_title("Total VAE Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Loss", fontsize=10)
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # 2. Reconstruction Loss
    axes[1].plot(epochs, history["train_recon"], label="Train Recon (MSE)", color="#2b5c8f", lw=2)
    axes[1].plot(epochs, history["val_recon"], label="Val Recon (MSE)", color="#d95f02", lw=2, linestyle="--")
    axes[1].set_title("Reconstruction Loss (MSE)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # 3. KL Divergence
    axes[2].plot(epochs, history["train_kl"], label="Train KL Divergence", color="#7570b3", lw=2)
    axes[2].plot(epochs, history["val_kl"], label="Val KL Divergence", color="#1b9e77", lw=2, linestyle="--")
    axes[2].set_title("KL Divergence (Latent Regularization)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch", fontsize=10)
    axes[2].legend()
    axes[2].grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Variational Autoencoder (VAE) Training Dynamics", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[Evaluation] Saved VAE loss curves to: {output_path}")


def plot_latent_space_tsne(Z_latent: np.ndarray, y: np.ndarray, output_path: str) -> None:
    """
    Visualizes compressed VAE latent manifold using 2D t-SNE and PCA.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Compute 2D t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z_tsne = tsne.fit_transform(Z_latent)

    # Compute 2D PCA
    pca = PCA(n_components=2, random_state=42)
    z_pca = pca.fit_transform(Z_latent)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # t-SNE Plot
    scatter0 = axes[0].scatter(
        z_tsne[y == 0, 0], z_tsne[y == 0, 1], c="#2b5c8f", label="Control (0)", alpha=0.75, s=35, edgecolors="none"
    )
    scatter1 = axes[0].scatter(
        z_tsne[y == 1, 0], z_tsne[y == 1, 1], c="#d95f02", label="Gastric Cancer (1)", alpha=0.75, s=35, edgecolors="none"
    )
    axes[0].set_title("t-SNE Projection of VAE Latent Space (z)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("t-SNE Dimension 1", fontsize=10)
    axes[0].set_ylabel("t-SNE Dimension 2", fontsize=10)
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].grid(True, linestyle=":", alpha=0.5)

    # PCA Plot
    axes[1].scatter(
        z_pca[y == 0, 0], z_pca[y == 0, 1], c="#2b5c8f", label="Control (0)", alpha=0.75, s=35, edgecolors="none"
    )
    axes[1].scatter(
        z_pca[y == 1, 0], z_pca[y == 1, 1], c="#d95f02", label="Gastric Cancer (1)", alpha=0.75, s=35, edgecolors="none"
    )
    axes[1].set_title(
        f"PCA Projection of VAE Latent Space (Expl. Var: {(pca.explained_variance_ratio_.sum()*100):.1f}%)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_xlabel("PC 1", fontsize=10)
    axes[1].set_ylabel("PC 2", fontsize=10)
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("VAE Compressed Latent Representation Structure", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[Evaluation] Saved Latent Space t-SNE/PCA to: {output_path}")


def plot_reconstruction_r2(r2_df: pd.DataFrame, output_path: str, top_n: int = 20) -> None:
    """
    Plots horizontal bar chart of feature reconstruction R2 scores.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_top = r2_df.head(top_n).sort_values(by="r2_score", ascending=True)

    plt.figure(figsize=(9, max(6, top_n * 0.3)), dpi=300)
    bars = plt.barh(df_top["feature"], df_top["r2_score"], color="#1b9e77", edgecolor="black", alpha=0.85)

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.xlabel("Reconstruction Fidelity (R² Score)", fontsize=11, fontweight="bold")
    plt.title(f"VAE Feature Reconstruction Fidelity (Top {top_n} Features)", fontsize=13, fontweight="bold", pad=12)
    plt.xlim(0, 1.15)
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[Evaluation] Saved reconstruction fidelity plot to: {output_path}")


def plot_comparison_roc(
    model_predictions: Dict[str, Dict[str, np.ndarray]],
    y_true: np.ndarray,
    output_path: str,
) -> None:
    """
    Plots multi-model ROC Curves.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8.5, 6.5), dpi=300)

    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"]
    for idx, (name, preds) in enumerate(model_predictions.items()):
        y_prob = preds["prob"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, lw=2.2, color=colors[idx % len(colors)], label=f"{name} (AUC = {auc:.4f})")

    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Chance (AUC = 0.50)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    plt.title("ROC Curves: VAE Latent & Hybrid Models vs. Baseline", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=9, frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[Evaluation] Saved comparison ROC plot to: {output_path}")


def plot_comparison_pr(
    model_predictions: Dict[str, Dict[str, np.ndarray]],
    y_true: np.ndarray,
    output_path: str,
) -> None:
    """
    Plots Precision-Recall Curves.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8.5, 6.5), dpi=300)

    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"]
    for idx, (name, preds) in enumerate(model_predictions.items()):
        y_prob = preds["prob"]
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.plot(rec, prec, lw=2.2, color=colors[idx % len(colors)], label=f"{name} (AP = {ap:.4f})")

    baseline = y_true.sum() / len(y_true)
    plt.axhline(y=baseline, color="gray", linestyle="--", lw=1.5, label=f"Baseline ({baseline:.2f})")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("Recall (Sensitivity)", fontsize=11, fontweight="bold")
    plt.ylabel("Precision", fontsize=11, fontweight="bold")
    plt.title("Precision-Recall Curves: VAE Latent & Hybrid Models", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower left", fontsize=9, frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[Evaluation] Saved PR comparison plot to: {output_path}")


def plot_confusion_matrices(
    model_predictions: Dict[str, Dict[str, np.ndarray]],
    y_true: np.ndarray,
    output_path: str,
) -> None:
    """
    Plots normalized confusion matrices for top models.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n_models = len(model_predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(4.2 * n_models, 3.8), dpi=300)

    if n_models == 1:
        axes = [axes]

    for ax, (name, preds) in zip(axes, model_predictions.items()):
        y_pred = preds["pred"]
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Purples", vmin=0, vmax=1)
        ax.set_title(name, fontsize=10, fontweight="bold")

        classes = ["Control", "Cancer"]
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=9)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=9)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text_color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight="bold",
                    fontsize=9,
                )

        ax.set_ylabel("True Diagnosis", fontsize=9, fontweight="bold")
        ax.set_xlabel("Predicted Diagnosis", fontsize=9, fontweight="bold")

    plt.suptitle("Confusion Matrices (Counts and Proportions)", fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[Evaluation] Saved confusion matrices to: {output_path}")


def save_vae_results_summary(
    all_metrics: Dict[str, Dict[str, float]],
    cv_fold_df: pd.DataFrame,
    r2_df: pd.DataFrame,
    results_dir: str,
) -> None:
    """
    Exports metrics, CV records, and reconstruction tables to disk.
    """
    os.makedirs(results_dir, exist_ok=True)

    json_path = os.path.join(results_dir, "vae_ensemble_metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"[Evaluation] Saved metrics JSON to: {json_path}")

    cv_csv_path = os.path.join(results_dir, "vae_cv_fold_scores.csv")
    cv_fold_df.to_csv(cv_csv_path, index=False)
    print(f"[Evaluation] Saved CV fold scores CSV to: {cv_csv_path}")

    r2_csv_path = os.path.join(results_dir, "vae_reconstruction_r2.csv")
    r2_df.to_csv(r2_csv_path, index=False)
    print(f"[Evaluation] Saved reconstruction R2 CSV to: {r2_csv_path}")
