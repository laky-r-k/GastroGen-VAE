"""
Training Pipeline for Random Forest, AdaBoost, and Hybrid Ensembles.
Performs Stratified K-Fold Cross-Validation, Out-Of-Fold Evaluation,
Model Serialization, and Diagnostic Reporting.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import StratifiedKFold

from preprocessing import PreprocessingPipeline
from models import (
    build_random_forest,
    build_adaboost,
    build_soft_voting_ensemble,
    build_stacking_ensemble,
    save_model,
)
from evaluate import (
    compute_metrics,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_confusion_matrices,
    plot_feature_importance,
    save_results_summary,
)


def run_cross_validation_training(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, np.ndarray]], pd.DataFrame, np.ndarray]:
    """
    Executes Stratified K-Fold Cross-Validation for all candidate models.
    Returns aggregated metrics, out-of-fold predictions, fold records, and RF feature importances.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    model_factories = {
        "Random Forest": build_random_forest,
        "AdaBoost": build_adaboost,
        "Soft Voting (RF + AdaBoost)": lambda: build_soft_voting_ensemble(
            build_random_forest(random_state=random_state),
            build_adaboost(random_state=random_state),
        ),
        "Stacking (RF + AdaBoost)": lambda: build_stacking_ensemble(
            build_random_forest(random_state=random_state),
            build_adaboost(random_state=random_state),
        ),
    }

    oof_predictions = {
        name: {
            "pred": np.zeros(len(y), dtype=int),
            "prob": np.zeros(len(y), dtype=float),
        }
        for name in model_factories
    }

    fold_records = []
    rf_feature_importances = np.zeros(len(feature_names))

    print(f"\n[Training] Starting {n_splits}-Fold Stratified Cross-Validation on {len(X)} samples...")
    start_time = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"\n--- Fold {fold}/{n_splits} (Train: {len(X_train)} samples, Val: {len(X_val)} samples) ---")

        for name, factory in model_factories.items():
            model = factory()
            model.fit(X_train, y_train)

            # Validation predictions
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]

            oof_predictions[name]["pred"][val_idx] = y_pred
            oof_predictions[name]["prob"][val_idx] = y_prob

            # Compute fold metrics
            metrics = compute_metrics(y_val, y_pred, y_prob)
            fold_record = {"fold": fold, "model": name, **metrics}
            fold_records.append(fold_record)

            print(
                f"  [{name:28s}] AUC: {metrics['roc_auc']:.4f} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | Rec: {metrics['recall_sensitivity']:.4f}"
            )

            # Accumulate RF feature importance
            if name == "Random Forest":
                rf_feature_importances += model.feature_importances_ / n_splits

    elapsed = time.time() - start_time
    print(f"\n[Training] Cross-Validation finished in {elapsed:.2f}s.")

    cv_fold_df = pd.DataFrame(fold_records)

    # Compute overall Out-Of-Fold metrics
    overall_metrics = {}
    print("\n" + "=" * 88)
    print("FINAL 5-FOLD OUT-OF-FOLD (OOF) PERFORMANCE SUMMARY")
    print("=" * 88)
    for name in model_factories:
        y_pred = oof_predictions[name]["pred"]
        y_prob = oof_predictions[name]["prob"]
        m = compute_metrics(y, y_pred, y_prob)

        sub_df = cv_fold_df[cv_fold_df["model"] == name]
        m["cv_auc_mean"] = float(sub_df["roc_auc"].mean())
        m["cv_auc_std"] = float(sub_df["roc_auc"].std())
        m["cv_acc_mean"] = float(sub_df["accuracy"].mean())
        m["cv_acc_std"] = float(sub_df["accuracy"].std())
        m["cv_f1_mean"] = float(sub_df["f1_score"].mean())
        m["cv_f1_std"] = float(sub_df["f1_score"].std())

        overall_metrics[name] = m

        print(
            f"{name:28s} | OOF AUC: {m['roc_auc']:.4f} (CV: {m['cv_auc_mean']:.4f}±{m['cv_auc_std']:.4f}) | "
            f"Acc: {m['accuracy']:.4f} | F1: {m['f1_score']:.4f} | Recall: {m['recall_sensitivity']:.4f} | Spec: {m['specificity']:.4f}"
        )
    print("=" * 88)

    return overall_metrics, oof_predictions, cv_fold_df, rf_feature_importances


def train_and_export_pipeline(
    raw_df: pd.DataFrame,
    pipeline: PreprocessingPipeline,
    model_output_dir: str,
    results_output_dir: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Full pipeline execution:
      1. Preprocessing, anomaly sanitization, feature engineering & selection
      2. 5-Fold Stratified Cross Validation with OOF metric calculation
      3. Generation of diagnostic plots and saving of summary tables
      4. Retraining final models on 100% of data and serializing to model/
    """
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(results_output_dir, exist_ok=True)

    # 1. Preprocessing & Feature Engineering
    X, y, feature_names = pipeline.fit_transform(raw_df)
    print(f"[Pipeline] Features ready: {X.shape[0]} samples × {X.shape[1]} features.")

    # 2. Run Cross-Validation evaluation
    overall_metrics, oof_predictions, cv_fold_df, rf_importances = run_cross_validation_training(
        X, y, feature_names, n_splits=n_splits, random_state=random_state
    )

    # 3. Generate and save diagnostic visualizations
    print("\n[Pipeline] Generating diagnostic plots...")
    plot_roc_curves(
        oof_predictions,
        y,
        os.path.join(results_output_dir, "roc_curves.png"),
    )
    plot_precision_recall_curves(
        oof_predictions,
        y,
        os.path.join(results_output_dir, "precision_recall_curves.png"),
    )
    plot_confusion_matrices(
        oof_predictions,
        y,
        os.path.join(results_output_dir, "confusion_matrices.png"),
    )
    plot_feature_importance(
        feature_names,
        rf_importances,
        os.path.join(results_output_dir, "feature_importance.png"),
        top_n=min(20, len(feature_names)),
    )

    # 4. Save metrics summaries and rankings
    save_results_summary(
        overall_metrics,
        cv_fold_df,
        feature_names,
        rf_importances,
        results_output_dir,
    )

    # 5. Retrain final models on entire dataset and save to model/
    print("\n[Pipeline] Retraining final production models on 100% data...")
    final_rf = build_random_forest(random_state=random_state).fit(X, y)
    final_ada = build_adaboost(random_state=random_state).fit(X, y)
    final_voting = build_soft_voting_ensemble(
        build_random_forest(random_state=random_state),
        build_adaboost(random_state=random_state),
    ).fit(X, y)
    final_stacking = build_stacking_ensemble(
        build_random_forest(random_state=random_state),
        build_adaboost(random_state=random_state),
    ).fit(X, y)

    # Export serialized artifacts
    save_model(final_rf, os.path.join(model_output_dir, "random_forest_model.joblib"))
    save_model(final_ada, os.path.join(model_output_dir, "adaboost_model.joblib"))
    save_model(final_voting, os.path.join(model_output_dir, "hybrid_soft_voting_model.joblib"))
    save_model(final_stacking, os.path.join(model_output_dir, "hybrid_stacking_model.joblib"))
    save_model(pipeline, os.path.join(model_output_dir, "preprocessing_pipeline.joblib"))

    print(f"\n[Pipeline] All models and pipeline artifacts successfully saved to: {model_output_dir}")
    print(f"[Pipeline] All diagnostic plots and metrics saved to: {results_output_dir}")

    return {
        "metrics": overall_metrics,
        "feature_names": feature_names,
        "model_dir": model_output_dir,
        "results_dir": results_output_dir,
    }
