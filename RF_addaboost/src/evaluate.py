"""
Evaluation and Diagnostic Visualization Module for Gastric Cancer Models.
Computes clinical metrics and generates publication-quality plots.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List
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
    Computes a comprehensive suite of binary classification metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
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
    return metrics


def plot_roc_curves(
    model_predictions: Dict[str, Dict[str, np.ndarray]],
    y_true: np.ndarray,
    output_path: str,
) -> None:
    """
    Plots multi-model ROC Curves with AUC confidence annotations.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 6), dpi=300)

    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77", "#e7298a"]
    for idx, (name, preds) in enumerate(model_predictions.items()):
        y_prob = preds["prob"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, lw=2.2, color=colors[idx % len(colors)], label=f"{name} (AUC = {auc:.4f})")

    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, fontweight="bold")
    plt.title("ROC Curves - Gastric Cancer Detection", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[Evaluation] Saved ROC curves plot to: {output_path}")


def plot_precision_recall_curves(
    model_predictions: Dict[str, Dict[str, np.ndarray]],
    y_true: np.ndarray,
    output_path: str,
) -> None:
    """
    Plots Multi-Model Precision-Recall curves.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 6), dpi=300)

    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77", "#e7298a"]
    for idx, (name, preds) in enumerate(model_predictions.items()):
        y_prob = preds["prob"]
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.plot(rec, prec, lw=2.2, color=colors[idx % len(colors)], label=f"{name} (AP = {ap:.4f})")

    baseline = y_true.sum() / len(y_true)
    plt.axhline(y=baseline, color="gray", linestyle="--", lw=1.5, label=f"No Skill Baseline ({baseline:.2f})")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("Recall (Sensitivity)", fontsize=12, fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=12, fontweight="bold")
    plt.title("Precision-Recall Curves - Gastric Cancer Detection", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="lower left", fontsize=10, frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[Evaluation] Saved PR curves plot to: {output_path}")


def plot_confusion_matrices(
    model_predictions: Dict[str, Dict[str, np.ndarray]],
    y_true: np.ndarray,
    output_path: str,
) -> None:
    """
    Plots side-by-side normalized Confusion Matrix heatmaps.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n_models = len(model_predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4), dpi=300)

    if n_models == 1:
        axes = [axes]

    for ax, (name, preds) in zip(axes, model_predictions.items()):
        y_pred = preds["pred"]
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(name, fontsize=11, fontweight="bold")

        classes = ["Control", "Cancer"]
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=10)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=10)

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
                    fontsize=10,
                )

        ax.set_ylabel("True Diagnosis", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted Diagnosis", fontsize=10, fontweight="bold")

    plt.suptitle("Confusion Matrices (Counts and Class Proportions)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[Evaluation] Saved confusion matrices to: {output_path}")


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    output_path: str,
    top_n: int = 20,
) -> None:
    """
    Plots horizontal bar chart of top feature importances.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(
        by="Importance", ascending=True
    )

    feat_df_top = feat_df.tail(top_n)

    plt.figure(figsize=(9, max(6, top_n * 0.35)), dpi=300)
    bars = plt.barh(feat_df_top["Feature"], feat_df_top["Importance"], color="#2b5c8f", edgecolor="black", alpha=0.85)

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.xlabel("Mean Impurity Reduction (Feature Importance)", fontsize=11, fontweight="bold")
    plt.title(f"Top {top_n} Most Discriminative Clinical Biomarkers (Random Forest)", fontsize=13, fontweight="bold", pad=12)
    plt.xlim(0, max(feat_df_top["Importance"]) * 1.15)
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[Evaluation] Saved feature importance plot to: {output_path}")


def save_results_summary(
    all_metrics: Dict[str, Dict[str, float]],
    cv_fold_df: pd.DataFrame,
    feature_names: List[str],
    feature_importances: np.ndarray,
    results_dir: str,
) -> None:
    """
    Saves JSON and CSV metric reports into results folder.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Save JSON summary
    json_path = os.path.join(results_dir, "metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"[Evaluation] Saved metrics summary JSON to: {json_path}")

    # Save CV fold results CSV
    cv_csv_path = os.path.join(results_dir, "cv_fold_scores.csv")
    cv_fold_df.to_csv(cv_csv_path, index=False)
    print(f"[Evaluation] Saved CV fold scores CSV to: {cv_csv_path}")

    # Save feature importances CSV
    feat_df = pd.DataFrame({"feature": feature_names, "importance": feature_importances}).sort_values(
        by="importance", ascending=False
    )
    feat_csv_path = os.path.join(results_dir, "feature_importance_ranking.csv")
    feat_df.to_csv(feat_csv_path, index=False)
    print(f"[Evaluation] Saved feature ranking CSV to: {feat_csv_path}")
