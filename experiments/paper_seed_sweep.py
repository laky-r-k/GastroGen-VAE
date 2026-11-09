"""Run the paper-style 70/30 evaluation for a range of random seeds.

Example:
    python3 experiments/paper_seed_sweep.py --start-seed 42 --end-seed 100

The script saves per-seed metrics and patient partition membership. It does not
write dozens of duplicate train/test CSVs; seed 42's full prepared CSVs remain
in dataset/split70_30/processed_and_selected_feature/.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_handling.paper_preprocessing import (
    PAPER_INPUT_FEATURES,
    PAPER_SELECTED_FEATURES,
    PaperFeaturePreprocessor,
)
from evaluation.metrics import compute_metrics


SOURCE_CSV = PROJECT_ROOT / "dataset" / "orginal_data" / "gastric_cancer_dataset_raw.csv"
LABEL_COLUMN = "label"
N_CLUSTERS = 5


def run_sweep(start_seed: int, end_seed: int, output_dir: Path = None) -> Path:
    if start_seed > end_seed:
        raise ValueError("start-seed must be less than or equal to end-seed")
    if not SOURCE_CSV.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {SOURCE_CSV}")

    if output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "paper_reproduction" / f"seed_sweep_{start_seed}_{end_seed}"
    elif not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=False)

    raw_data = pd.read_csv(SOURCE_CSV)
    X_all = raw_data[PAPER_INPUT_FEATURES]
    y_all = raw_data[LABEL_COLUMN].to_numpy()
    parameter_template = {
        "n_estimators": 70,
        "learning_rate": 0.12,
        "max_depth": 5,
        "min_samples_leaf": 10,
        "min_samples_split": 2,
        "max_features": None,
    }

    score_path = output_dir / "seed_metrics.csv"
    manifest_path = output_dir / "patient_split_manifest.csv"
    manifest_chunks = []
    score_rows = []

    total = end_seed - start_seed + 1
    for index, seed in enumerate(range(start_seed, end_seed + 1), start=1):
        train_indices, test_indices = train_test_split(
            np.arange(len(raw_data)),
            test_size=0.30,
            random_state=seed,
            shuffle=True,
            stratify=y_all,
        )
        X_train_raw = X_all.iloc[train_indices]
        X_test_raw = X_all.iloc[test_indices]
        y_train = y_all[train_indices]
        y_test = y_all[test_indices]

        params = {**parameter_template, "random_state": seed}
        cv_pipeline = Pipeline(
            [
                ("preprocessor", PaperFeaturePreprocessor(n_clusters=N_CLUSTERS, random_state=seed)),
                ("classifier", GradientBoostingClassifier(**params)),
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

        processor = PaperFeaturePreprocessor(n_clusters=N_CLUSTERS, random_state=seed)
        X_train = processor.fit_transform(X_train_raw)
        X_test = processor.transform(X_test_raw)
        model = GradientBoostingClassifier(**params).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        holdout_metrics = compute_metrics(y_test, y_pred, y_prob)

        score_rows.append(
            {
                "seed": seed,
                "train_n": len(train_indices),
                "test_n": len(test_indices),
                "cv_auc_mean": float(np.mean(cv_auc)),
                "cv_auc_std": float(np.std(cv_auc, ddof=1)),
                "holdout_auc": holdout_metrics["roc_auc"],
                "holdout_pr_auc": holdout_metrics["pr_auc"],
                "holdout_accuracy": holdout_metrics["accuracy"],
                "holdout_f1": holdout_metrics["f1_score"],
                "holdout_sensitivity": holdout_metrics["recall_sensitivity"],
                "holdout_specificity": holdout_metrics["specificity"],
            }
        )
        manifest_chunks.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "patient_id": raw_data.iloc[np.concatenate([train_indices, test_indices])]["patient_id"].to_numpy(),
                    "partition": ["train"] * len(train_indices) + ["test"] * len(test_indices),
                }
            )
        )
        # Keep progress and partial results available if the full range is interrupted.
        pd.DataFrame(score_rows).to_csv(score_path, index=False)
        pd.concat(manifest_chunks, ignore_index=True).to_csv(manifest_path, index=False)
        current = score_rows[-1]
        print(
            f"[{index}/{total}] seed={seed}: CV AUC={current['cv_auc_mean']:.4f} "
            f"± {current['cv_auc_std']:.4f}; holdout AUC={current['holdout_auc']:.4f}",
            flush=True,
        )

    scores = pd.DataFrame(score_rows)
    summary = {
        "seed_range": [start_seed, end_seed],
        "number_of_splits": len(scores),
        "split": "stratified 70/30, independently generated per seed",
        "features": PAPER_SELECTED_FEATURES,
        "preprocessing": {
            "anomaly_rules": {"hb": ">300", "alb": ">100", "rdw": ">40"},
            "imputer": "training-fitted K-means centroid imputation; k=5",
        },
        "gradient_boosting_parameters": parameter_template,
        "cv_auc_mean_across_seeds": float(scores["cv_auc_mean"].mean()),
        "cv_auc_sd_across_seeds": float(scores["cv_auc_mean"].std(ddof=1)),
        "holdout_auc_mean": float(scores["holdout_auc"].mean()),
        "holdout_auc_sd": float(scores["holdout_auc"].std(ddof=1)),
        "holdout_auc_min": float(scores["holdout_auc"].min()),
        "holdout_auc_min_seed": int(scores.loc[scores["holdout_auc"].idxmin(), "seed"]),
        "holdout_auc_max": float(scores["holdout_auc"].max()),
        "holdout_auc_max_seed": int(scores.loc[scores["holdout_auc"].idxmax(), "seed"]),
        "interpretation": "Each row is a different random 70/30 split; this describes split sensitivity, not independent external validation.",
    }
    (output_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nSeed sweep metrics: {score_path}")
    print(f"Patient split manifests: {manifest_path}")
    print(f"Summary: {output_dir / 'sweep_summary.json'}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--end-seed", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=None)
    arguments = parser.parse_args()
    run_sweep(arguments.start_seed, arguments.end_seed, arguments.output_dir)
