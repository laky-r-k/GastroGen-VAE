import numpy as np
from typing import Dict
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, precision_score, recall_score, confusion_matrix, brier_score_loss
)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Computes standard classification metrics."""
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

import pandas as pd
from sklearn.metrics import r2_score
from typing import List

def compute_reconstruction_r2(X_input: np.ndarray, X_recon: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """
    Computes per-feature R2 reconstruction scores to measure how well each biomarker is preserved.
    """
    r2_scores = []
    for i, feat in enumerate(feature_names):
        score = r2_score(X_input[:, i], X_recon[:, i])
        mse = np.mean((X_input[:, i] - X_recon[:, i]) ** 2)
        r2_scores.append({"feature": feat, "r2_score": max(0.0, score), "mse": mse})

    return pd.DataFrame(r2_scores).sort_values(by="r2_score", ascending=False)
