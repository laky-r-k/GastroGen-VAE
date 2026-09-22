import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any

# Ensure we can import from the root module
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from preprocessing import PreprocessingPipeline
from RF_addaboost.src.models import (
    build_random_forest,
    build_adaboost,
    build_soft_voting_ensemble,
    build_stacking_ensemble,
)
from evaluation import (
    compute_metrics,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_confusion_matrices,
)
import json

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(root_dir, "dataset", "gastric_cancer_train.csv")
    test_path = os.path.join(root_dir, "dataset", "gastric_cancer_test.csv")
    results_dir = os.path.join(root_dir, "results", "splittrainresult")
    
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print("GASTRIC CANCER DETECTION: TRAIN-TEST SPLIT EVALUATION PIPELINE")
    print("=" * 80)
    
    # 1. Load Data
    print(f"Loading train data from {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)

    # 2. Preprocessing
    print("Preprocessing data...")
    pipeline = PreprocessingPipeline(feature_mode="optimized_subset", scaler_type="robust", apply_scaling=False)
    X_train, y_train, feature_names = pipeline.fit_transform(train_df)
    X_test, y_test = pipeline.transform(test_df)
    print(f"Train features shape: {X_train.shape}, Test features shape: {X_test.shape}")

    # 3. Train Models
    random_state = 42
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

    test_predictions = {}
    overall_metrics = {}

    print("Training models and evaluating on test set...")
    for name, factory in model_factories.items():
        print(f"  Training {name}...")
        model = factory()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        test_predictions[name] = {
            "pred": y_pred,
            "prob": y_prob
        }

        metrics = compute_metrics(y_test, y_pred, y_prob)
        overall_metrics[name] = metrics

        print(
            f"    [{name:28s}] AUC: {metrics['roc_auc']:.4f} | Acc: {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1_score']:.4f} | Rec: {metrics['recall_sensitivity']:.4f}"
        )

    # 4. Generate and save diagnostic visualizations
    print("\nGenerating diagnostic plots...")
    plot_roc_curves(
        test_predictions,
        y_test,
        os.path.join(results_dir, "roc_curves_test.png"),
        title="ROC Curves on Test Set"
    )
    plot_precision_recall_curves(
        test_predictions,
        y_test,
        os.path.join(results_dir, "precision_recall_curves_test.png"),
        title="PR Curves on Test Set"
    )
    plot_confusion_matrices(
        test_predictions,
        y_test,
        os.path.join(results_dir, "confusion_matrices_test.png"),
    )
    
    # 5. Save metrics
    metrics_path = os.path.join(results_dir, "test_metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(overall_metrics, f, indent=4)
    print(f"Saved metrics summary to {metrics_path}")
    
    print("\nPipeline finished successfully!")

if __name__ == "__main__":
    main()
