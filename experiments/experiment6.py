"""Experiment 6: train and store a paper-settings GB model on eight features."""

import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg", force=True)
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_handling.paper_preprocessing import (
    PAPER_INPUT_FEATURES,
    PAPER_SELECTED_FEATURES,
    PaperFeaturePreprocessor,
)
from evaluation import run_classification_evaluation
from evaluation.visualization import plot_feature_importance
from experiments.base_experiment import BaseExperiment


class Experiment6(BaseExperiment):
    """Fit, evaluate, and serialize a raw-input-to-GB inference pipeline."""

    experiment_name = "experiment6"

    def run(self):
        dataset_cfg = self.config.get("dataset", {})
        preprocess_cfg = self.config.get("preprocessing", {})
        gb_cfg = self.config.get("gradient_boosting", {})
        label_column = dataset_cfg.get("label_column", "label")
        id_column = dataset_cfg.get("id_column", "patient_id")

        train_path = self._resolve_path(
            dataset_cfg.get("train_path", "dataset/split70_30/gastric_cancer_train.csv")
        )
        test_path = self._resolve_path(
            dataset_cfg.get("test_path", "dataset/split70_30/gastric_cancer_test.csv")
        )
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        missing = [column for column in PAPER_INPUT_FEATURES if column not in train_df.columns]
        if missing:
            raise ValueError(f"Training file lacks required paper predictors: {missing}")
        missing_test = [column for column in PAPER_INPUT_FEATURES if column not in test_df.columns]
        if missing_test:
            raise ValueError(f"Test file lacks required paper predictors: {missing_test}")

        X_train = train_df[PAPER_INPUT_FEATURES]
        y_train = train_df[label_column].to_numpy()
        X_test = test_df[PAPER_INPUT_FEATURES]
        y_test = test_df[label_column].to_numpy()

        model_pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    PaperFeaturePreprocessor(
                        n_clusters=int(preprocess_cfg.get("kmeans_clusters", 5)),
                        random_state=int(self.config.get("seed", 42)),
                    ),
                ),
                (
                    "classifier",
                    GradientBoostingClassifier(
                        n_estimators=int(gb_cfg.get("n_estimators", 70)),
                        learning_rate=float(gb_cfg.get("learning_rate", 0.12)),
                        max_depth=int(gb_cfg.get("max_depth", 5)),
                        min_samples_leaf=int(gb_cfg.get("min_samples_leaf", 10)),
                        min_samples_split=int(gb_cfg.get("min_samples_split", 2)),
                        max_features=gb_cfg.get("max_features", None),
                        random_state=int(self.config.get("seed", 42)),
                    ),
                ),
            ]
        )

        print(f"Training rows: {len(X_train)} | Holdout rows: {len(X_test)}")
        print(f"Raw predictors provided: {len(PAPER_INPUT_FEATURES)}")
        print(f"Selected predictors used by GB: {PAPER_SELECTED_FEATURES}")
        print(f"Split seed: {self.config.get('seed', 42)}")
        model_pipeline.fit(X_train, y_train)

        metrics_dir = self.get_results_subdir("metrics")
        metrics = run_classification_evaluation(
            models={"Stored Paper-Settings GB": model_pipeline},
            X_test=X_test,
            y_test=y_test,
            results_dir=metrics_dir,
            feature_names=PAPER_SELECTED_FEATURES,
        )

        model_dir = self.get_results_subdir("models")
        joblib.dump(model_pipeline, os.path.join(model_dir, "gb_raw_to_selected_features_pipeline.joblib"))
        joblib.dump(
            model_pipeline.named_steps["classifier"],
            os.path.join(model_dir, "gradient_boosting_estimator.joblib"),
        )
        joblib.dump(
            model_pipeline.named_steps["preprocessor"],
            os.path.join(model_dir, "fitted_paper_preprocessor.joblib"),
        )

        estimator = model_pipeline.named_steps["classifier"]
        importance_df = pd.DataFrame(
            {"feature": PAPER_SELECTED_FEATURES, "importance": estimator.feature_importances_}
        ).sort_values("importance", ascending=False)
        importance_df.to_csv(os.path.join(metrics_dir, "selected_feature_importance.csv"), index=False)
        plot_feature_importance(
            PAPER_SELECTED_FEATURES,
            estimator.feature_importances_,
            os.path.join(metrics_dir, "selected_feature_importance.png"),
        )

        model_card = {
            "model_type": "sklearn Pipeline: PaperFeaturePreprocessor + GradientBoostingClassifier",
            "seed": int(self.config.get("seed", 42)),
            "training_file": train_path,
            "test_file": test_path,
            "training_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "raw_input_features": PAPER_INPUT_FEATURES,
            "selected_model_features": PAPER_SELECTED_FEATURES,
            "label_column": label_column,
            "id_column_excluded_from_model": id_column,
            "preprocessing": {
                "outlier_rules": {"hb": ">300", "alb": ">100", "rdw": ">40"},
                "imputation": "nearest training-fitted K-means centroid",
                "kmeans_clusters": int(preprocess_cfg.get("kmeans_clusters", 5)),
            },
            "gradient_boosting_parameters": estimator.get_params(deep=False),
            "holdout_metrics": metrics,
            "pipeline_artifact": "models/gb_raw_to_selected_features_pipeline.joblib",
            "note": "The paper does not publish the split seed, outlier cutoffs, or K-means cluster count; this model uses the documented project settings.",
        }
        with open(os.path.join(model_dir, "model_card.json"), "w", encoding="utf-8") as f:
            json.dump(model_card, f, indent=2)

        print(f"Stored inference pipeline: {os.path.join(model_dir, 'gb_raw_to_selected_features_pipeline.joblib')}")
        return {"model_dir": model_dir, "metrics_dir": metrics_dir, "metrics": metrics}


def main():
    Experiment6()()


if __name__ == "__main__":
    main()
