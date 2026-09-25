import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List

def save_results_summary(all_metrics: Dict[str, Dict[str, float]], cv_fold_df: pd.DataFrame, feature_names: List[str], feature_importances: np.ndarray, results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics_summary.json"), "w") as f:
        json.dump(all_metrics, f, indent=4)
    cv_fold_df.to_csv(os.path.join(results_dir, "cv_fold_scores.csv"), index=False)
    feat_df = pd.DataFrame({"feature": feature_names, "importance": feature_importances}).sort_values(by="importance", ascending=False)
    feat_df.to_csv(os.path.join(results_dir, "feature_importance_ranking.csv"), index=False)

def save_vae_results_summary(all_metrics: Dict[str, Dict[str, float]], cv_fold_df: pd.DataFrame, r2_df: pd.DataFrame, results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "vae_ensemble_metrics_summary.json"), "w") as f:
        json.dump(all_metrics, f, indent=4)
    cv_fold_df.to_csv(os.path.join(results_dir, "vae_cv_fold_scores.csv"), index=False)
    r2_df.to_csv(os.path.join(results_dir, "vae_reconstruction_r2.csv"), index=False)
