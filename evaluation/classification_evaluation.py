import os
import pandas as pd
from typing import Dict, Any, List

from .metrics import compute_metrics
from .visualization import (
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_confusion_matrices,
    plot_feature_importance
)
from .reporting import save_results_summary

def run_classification_evaluation(
    models: Dict[str, Any],
    X_test,
    y_test,
    results_dir: str,
    feature_names: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    High-level orchestrator for evaluating multiple classification models.
    Automatically generates predictions, computes metrics, creates plots, and saves results.
    """
    print(f"\n[Evaluation] Starting classification evaluation. Output directory: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)
    
    test_predictions = {}
    overall_metrics = {}

    for name, model in models.items():
        print(f"  Evaluating {name}...")
        y_pred = model.predict(X_test)
        
        # Check if the model supports predict_proba (some ensembles might differ)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            # Fallback if no proba is available
            y_prob = y_pred

        test_predictions[name] = {
            "pred": y_pred,
            "prob": y_prob
        }

        # Compute core metrics
        metrics = compute_metrics(y_test, y_pred, y_prob)
        overall_metrics[name] = metrics

        print(
            f"    [{name:28s}] AUC: {metrics['roc_auc']:.4f} | Acc: {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1_score']:.4f} | Rec: {metrics['recall_sensitivity']:.4f}"
        )
        
        # Optionally extract feature importances for Random Forest or AdaBoost
        if feature_names and hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            plot_path = os.path.join(results_dir, f"feature_importance_{name.replace(' ', '_')}.png")
            plot_feature_importance(feature_names, importances, plot_path)
            
            # Save the specific feature importance csv
            feat_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(by="importance", ascending=False)
            feat_df.to_csv(os.path.join(results_dir, f"feature_importance_{name.replace(' ', '_')}.csv"), index=False)

    print("\n[Evaluation] Generating diagnostic plots...")
    plot_roc_curves(
        test_predictions,
        y_test,
        os.path.join(results_dir, "roc_curves.png")
    )
    plot_precision_recall_curves(
        test_predictions,
        y_test,
        os.path.join(results_dir, "precision_recall_curves.png")
    )
    plot_confusion_matrices(
        test_predictions,
        y_test,
        os.path.join(results_dir, "confusion_matrices.png")
    )
    
    # Save the main metrics json
    # Note: cv_fold_df and overall feature_importances are set to dummy/None here 
    # since this pipeline orchestrates the test-set evaluation rather than CV.
    save_results_summary(
        all_metrics=overall_metrics,
        cv_fold_df=pd.DataFrame(), # Empty placeholder if no CV was run
        feature_names=[],          # Handled individually above
        feature_importances=[],
        results_dir=results_dir
    )
    
    print("[Evaluation] Classification evaluation complete.\n")
    return overall_metrics
