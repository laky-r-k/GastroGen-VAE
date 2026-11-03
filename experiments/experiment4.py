"""Experiment 4: reproduce the paper's reported Gradient Boosting settings.

The classifier is evaluated on this project's existing real train/test split.
This isolates the effect of matching the paper's reported hyperparameters;
it does not reproduce the paper's cohort or its evaluation protocol.
"""

import os
import sys

# This experiment only writes figures to files; use a non-interactive backend
# so plotting does not initialize Qt/GNOME settings on desktop Linux.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.base_experiment import BaseExperiment
from model_src import build_gradient_boosting
from evaluation import run_classification_evaluation


class Experiment4(BaseExperiment):
    """Train and evaluate Gradient Boosting with the paper's settings."""

    experiment_name = "experiment4"

    def run(self):
        print("[1/3] Loading the project's real train/test split...")
        train_df, test_df, feature_cols, y_train_series, y_test_series = self.load_data()

        X_train = train_df[feature_cols].values
        y_train = y_train_series.values
        X_test = test_df[feature_cols].values
        y_test = y_test_series.values

        gb_cfg = self.config["gradient_boosting"]
        print("[2/3] Building Gradient Boosting with paper-reported parameters...")
        model = build_gradient_boosting(
            n_estimators=gb_cfg["n_estimators"],
            learning_rate=gb_cfg["learning_rate"],
            max_depth=gb_cfg["max_depth"],
            min_samples_split=gb_cfg["min_samples_split"],
            min_samples_leaf=gb_cfg["min_samples_leaf"],
            max_features=gb_cfg["max_features"],
            random_state=gb_cfg["random_state"],
        )
        model.fit(X_train, y_train)

        print("[3/3] Evaluating on the held-out real test set...")
        results_dir = self.get_results_subdir("gradient_boosting")
        metrics = run_classification_evaluation(
            models={"Paper-parameter Gradient Boosting": model},
            X_test=X_test,
            y_test=y_test,
            results_dir=results_dir,
            feature_names=feature_cols,
        )
        return {"gradient_boosting_dir": results_dir, "metrics": metrics}


def main():
    Experiment4()()


if __name__ == "__main__":
    main()
