"""Create paper-style 70/30 data files and evaluate the reported GBDT setup.

Run from the project root:
    python3 experiments/paper_reproduction.py

The exact paper split and some preprocessing details were not published. This
script records its seed and explicit assumptions so the run is reproducible.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The experiment writes figures to files and should not initialize a desktop GUI.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)

from data_handling.paper_preprocessing import (
    PAPER_INPUT_FEATURES,
    PAPER_SELECTED_FEATURES,
    PaperFeaturePreprocessor,
)
from data_handling.split_dataset import split_dataset
from evaluation import run_classification_evaluation


SOURCE_CSV = PROJECT_ROOT / "dataset" / "orginal_data" / "gastric_cancer_dataset_raw.csv"
N_CLUSTERS = 5


def _make_results_dir(seed: int) -> Path:
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = PROJECT_ROOT / "results" / "paper_reproduction" / f"seed_{seed}_run_{now}"
    results_dir.mkdir(parents=True, exist_ok=False)
    return results_dir


def run(seed: int = 42, split_dir: Path = None, overwrite_split: bool = False) -> Path:
    if split_dir is None:
        split_name = "split70_30" if seed == 42 else f"split70_30_seed{seed}"
        split_dir = PROJECT_ROOT / "dataset" / split_name
    elif not split_dir.is_absolute():
        split_dir = PROJECT_ROOT / split_dir
    processed_dir = split_dir / "processed_and_selected_feature"
    gbdt_parameters = {
        "n_estimators": 70,
        "learning_rate": 0.12,
        "max_depth": 5,
        "min_samples_leaf": 10,
        # The paper does not report these two sklearn parameters; retain defaults.
        "min_samples_split": 2,
        "max_features": None,
        "random_state": seed,
    }

    # 1. Split raw records before any fitted preprocessing.
    split_dataset(
        source=SOURCE_CSV,
        output_dir=split_dir,
        train_percent=70,
        test_percent=30,
        random_seed=seed,
        overwrite=overwrite_split,
    )
    raw_train = pd.read_csv(split_dir / "gastric_cancer_train.csv")
    raw_test = pd.read_csv(split_dir / "gastric_cancer_test.csv")
    label_column = "label"

    # Save patient partition membership without making the identifier a model feature.
    pd.concat(
        [
            raw_train[["patient_id"]].assign(partition="train"),
            raw_test[["patient_id"]].assign(partition="test"),
        ],
        ignore_index=True,
    ).to_csv(split_dir / "patient_split_manifest.csv", index=False)

    X_train_raw = raw_train[PAPER_INPUT_FEATURES]
    X_test_raw = raw_test[PAPER_INPUT_FEATURES]
    y_train = raw_train[label_column].to_numpy()
    y_test = raw_test[label_column].to_numpy()

    # 2. Paper feature selection is fixed from the published result. Fit the
    # imputation state on training data only, then reuse it for the holdout.
    processor = PaperFeaturePreprocessor(n_clusters=N_CLUSTERS, random_state=seed)
    X_train = processor.fit_transform(X_train_raw)
    X_test = processor.transform(X_test_raw)

    processed_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([X_train.reset_index(drop=True), pd.Series(y_train, name=label_column)], axis=1).to_csv(
        processed_dir / "gastric_cancer_train.csv", index=False
    )
    pd.concat([X_test.reset_index(drop=True), pd.Series(y_test, name=label_column)], axis=1).to_csv(
        processed_dir / "gastric_cancer_test.csv", index=False
    )
    joblib.dump(processor, processed_dir / "fitted_preprocessor.joblib")

    preprocessing_metadata = {
        "source_csv": str(SOURCE_CSV),
        "split_directory": str(split_dir),
        "train_rows": int(len(raw_train)),
        "test_rows": int(len(raw_test)),
        "random_seed": seed,
        "stratified_by": label_column,
        "paper_input_features": PAPER_INPUT_FEATURES,
        "selected_model_features": PAPER_SELECTED_FEATURES,
        "outlier_rules_used": {"hb": ">300", "alb": ">100", "rdw": ">40"},
        "outlier_rule_source": "Project dataset audit; paper does not specify cutoffs.",
        "imputation": {
            "method": "nearest fitted K-means centroid, feature-wise",
            "clusters": processor.n_clusters_,
            "random_seed": seed,
            "distance_scaling": "StandardScaler fitted on complete training records",
            "fit_scope": "training split only",
        },
        "feature_selection": "Eight predictors reported by the paper; no test-set fitting.",
        "note": "Paper does not publish exact split IDs, random seed, outlier cutoffs, or K-means cluster count.",
    }
    (processed_dir / "preprocessing_metadata.json").write_text(
        json.dumps(preprocessing_metadata, indent=2) + "\n", encoding="utf-8"
    )

    # 3. Five-fold stratified CV on training data only. The preprocessor is
    # inside the sklearn pipeline, so each CV fold fits its own imputer.
    cv_pipeline = Pipeline(
        [
            ("preprocessor", PaperFeaturePreprocessor(n_clusters=N_CLUSTERS, random_state=seed)),
            ("classifier", GradientBoostingClassifier(**gbdt_parameters)),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_auc = cross_val_score(
        cv_pipeline,
        X_train_raw,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,
    )

    # 4. Fit the paper-reported final parameters on all 70% training patients.
    model = GradientBoostingClassifier(**gbdt_parameters)
    model.fit(X_train, y_train)
    results_dir = _make_results_dir(seed)
    metrics = run_classification_evaluation(
        models={"Paper-style Gradient Boosting": model},
        X_test=X_test,
        y_test=y_test,
        results_dir=str(results_dir / "metrics"),
        feature_names=PAPER_SELECTED_FEATURES,
    )
    cv_summary = {
        "fold_auc": [float(value) for value in cv_auc],
        "mean_auc": float(np.mean(cv_auc)),
        "std_auc": float(np.std(cv_auc, ddof=1)),
        "n_splits": 5,
        "scoring": "roc_auc",
    }
    (results_dir / "cross_validation_auc.json").write_text(
        json.dumps(cv_summary, indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "split_directory": str(split_dir),
                "processed_data_directory": str(processed_dir),
                "gbdt_parameters": gbdt_parameters,
                "holdout_metrics": metrics,
                "paper_test_size": 213,
                "actual_test_size": int(len(y_test)),
                "note": "The paper's split IDs and some preprocessing details are unavailable; this is a documented reproduction attempt, not an exact rerun.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    joblib.dump(model, results_dir / "gradient_boosting_model.joblib")

    print(f"Raw 70/30 split: {split_dir}")
    print(f"Processed selected-feature files: {processed_dir}")
    print(f"Cross-validation AUC: {cv_summary['mean_auc']:.4f} ± {cv_summary['std_auc']:.4f}")
    print(f"Holdout results: {results_dir}")
    return results_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite-split",
        action="store_true",
        help="Replace existing raw split CSVs and metadata under dataset/split70_30.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split, CV, imputation, and model seed (default: 42).")
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=None,
        help="Optional output directory. Other seeds default to dataset/split70_30_seed<seed>.",
    )
    args = parser.parse_args()
    run(seed=args.seed, split_dir=args.split_dir, overwrite_split=args.overwrite_split)
