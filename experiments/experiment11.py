"""Experiment 11: default local TabPFN with real-only vs real + WGAN data."""

import json
import os
import random
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import tabpfn
    from tabpfn import TabPFNClassifier
except ImportError as exc:
    raise ImportError(
        "Experiment 11 requires the local 'tabpfn' package. Install and pin the "
        "package version before running; this experiment does not use AutoTabPFN "
        "or a hosted client."
    ) from exc

from data_handling.paper_preprocessing import (
    PAPER_INPUT_FEATURES,
    PAPER_SELECTED_FEATURES,
    PaperFeaturePreprocessor,
)
from evaluation import compute_metrics
from experiments.base_experiment import BaseExperiment
from experiments.experiment10 import _class_counts, seed_everything
from experiments.experiment9 import (
    FEATURES,
    nearest_neighbor_audit,
)


class Experiment11(BaseExperiment):
    """Compare default local TabPFN on real-only and augmented training splits."""

    experiment_name = "experiment11"

    def run(self):
        cfg = self.config
        torch.set_num_threads(int(cfg.get("runtime", {}).get("torch_num_threads", 1)))
        dataset_cfg = cfg.get("dataset", {})
        split_cfg = cfg.get("split", {})
        prep_cfg = cfg.get("preprocessing", {})

        raw_path = self._resolve_path(dataset_cfg.get(
            "raw_path", "dataset/orginal_data/gastric_cancer_dataset_raw.csv"
        ))
        label_column = dataset_cfg.get("label_column", "label")
        id_column = dataset_cfg.get("id_column", "patient_id")
        cohort = pd.read_csv(raw_path)
        missing = [c for c in PAPER_INPUT_FEATURES + [label_column] if c not in cohort]
        if missing:
            raise ValueError(f"Raw cohort lacks required columns: {missing}")
        if cohort[label_column].isna().any():
            raise ValueError("The cohort contains missing labels.")

        seed_start = int(split_cfg.get("seed_start", 42))
        seed_end = int(split_cfg.get("seed_end", 100))
        test_size = float(split_cfg.get("test_size", 0.30))
        if seed_start > seed_end:
            raise ValueError("split.seed_start must be <= split.seed_end")
        seeds = list(range(seed_start, seed_end + 1))
        y_all = cohort[label_column].to_numpy(dtype=int)
        x_all = cohort[PAPER_INPUT_FEATURES]
        metrics_dir = self.get_results_subdir("metrics")
        synthetic_root = self._resolve_path(cfg.get(
            "synthetic_input_dir", "dataset/synthetic_8_features/experiment10"
        ))
        split_manifest_path = self._resolve_path(cfg.get(
            "synthetic_split_manifest",
            "results/experiment10/run_2026-11-10_02-14-37/splits/patient_split_manifest.csv",
        ))
        if not os.path.isfile(split_manifest_path):
            raise FileNotFoundError(f"Experiment 10 split manifest not found: {split_manifest_path}")
        source_splits = pd.read_csv(split_manifest_path)

        # Instantiate once to record the exact package defaults; create a fresh,
        # identically configured classifier for each condition and seed.
        default_probe = TabPFNClassifier()
        tabpfn_params = default_probe.get_params(deep=True)
        del default_probe
        package_version = getattr(tabpfn, "__version__", "unknown")
        expected_version = cfg.get("tabpfn", {}).get("expected_version")
        if expected_version and package_version != str(expected_version):
            raise RuntimeError(
                f"Experiment 11 expects tabpfn=={expected_version}, found {package_version}. "
                "Install the pinned package version before running so the paper settings are reproducible."
            )
        classifier_record = {
            "class": "tabpfn.TabPFNClassifier",
            "package_version": package_version,
            "parameters": tabpfn_params,
            "constructor_overrides": {},
            "hosted_client": False,
            "auto_tabpfn": False,
        }
        with open(os.path.join(self.run_dir, "tabpfn_configuration.json"), "w", encoding="utf-8") as f:
            json.dump(classifier_record, f, indent=2, default=str)

        print(f"TabPFN package version: {package_version}")
        print("Estimator: default local TabPFNClassifier() with no parameter overrides")
        print(f"Seeds: {seed_start}–{seed_end} ({len(seeds)} stratified splits); holdout: {test_size:.0%}")
        print(f"Reusing saved WGAN-GP data from: {synthetic_root}")

        metrics: List[Dict[str, float]] = []
        manifests = []
        synthetic_quality = []
        for seed_index, seed in enumerate(seeds, 1):
            seed_everything(seed)
            train_idx, test_idx = train_test_split(
                np.arange(len(cohort)), test_size=test_size, random_state=seed,
                shuffle=True, stratify=y_all,
            )
            x_train_raw, x_test_raw = x_all.iloc[train_idx], x_all.iloc[test_idx]
            y_train, y_test = y_all[train_idx], y_all[test_idx]

            preprocessor = PaperFeaturePreprocessor(
                n_clusters=int(prep_cfg.get("kmeans_clusters", 5)), random_state=seed
            )
            X_train_df = preprocessor.fit_transform(x_train_raw)
            X_test_df = preprocessor.transform(x_test_raw)
            X_train = X_train_df[FEATURES].to_numpy(dtype=float)
            X_test = X_test_df[FEATURES].to_numpy(dtype=float)

            real_counts = _class_counts(y_train)
            if set(real_counts) != {0, 1}:
                raise ValueError(f"Seed {seed} lacks both binary classes: {real_counts}")
            # Verify the saved WGAN split matches this exact 70/30 split before use.
            manifest_seed = source_splits[source_splits.seed == seed]
            def normalized_ids(values):
                # CSV inference may represent the same integer patient ID as 281 or 281.0.
                return {str(int(float(value))) for value in values}

            expected_train = normalized_ids(cohort.iloc[train_idx][id_column])
            expected_test = normalized_ids(cohort.iloc[test_idx][id_column])
            saved_train = normalized_ids(manifest_seed.loc[manifest_seed.partition == "train", "patient_id"])
            saved_test = normalized_ids(manifest_seed.loc[manifest_seed.partition == "test", "patient_id"])
            if expected_train != saved_train or expected_test != saved_test:
                raise RuntimeError(f"Seed {seed}: saved Experiment 10 split does not match the current cohort split.")

            seed_dir = os.path.join(synthetic_root, f"seed_{seed}")
            synthetic_path = os.path.join(seed_dir, "wgan_gp_balancing.csv")
            metadata_path = os.path.join(seed_dir, "generation_metadata.json")
            if not os.path.isfile(synthetic_path) or not os.path.isfile(metadata_path):
                raise FileNotFoundError(f"Missing saved WGAN data or metadata for seed {seed}: {seed_dir}")
            with open(metadata_path, encoding="utf-8") as f:
                source_metadata = json.load(f)
            if source_metadata.get("seed") != seed or source_metadata.get("generator_fit_scope") != "this seed's real training partition only":
                raise RuntimeError(f"Seed {seed}: saved synthetic-data provenance is unexpected.")
            synthetic_raw = pd.read_csv(synthetic_path)
            if not set(FEATURES + ["label"]).issubset(synthetic_raw.columns):
                raise ValueError(f"Seed {seed}: synthetic data lacks expected eight features or label.")
            generated_y = synthetic_raw["label"].to_numpy(dtype=int)
            synth_counts = _class_counts(generated_y)
            combined_counts = {c: real_counts[c] + synth_counts.get(c, 0) for c in (0, 1)}
            if len(synthetic_raw) != len(y_train) or combined_counts[0] != combined_counts[1]:
                raise RuntimeError(f"Seed {seed}: saved synthetic data does not balance real training labels: {combined_counts}")
            # Saved Experiment 10 rows already contain the eight paper-selected
            # features in clinical units after that split's training-fitted preprocessing.
            X_synthetic = synthetic_raw[FEATURES].to_numpy(dtype=float)
            synthetic_df = synthetic_raw[FEATURES + ["label"]].copy()

            real_train_frame = pd.DataFrame(X_train, columns=FEATURES)
            real_train_frame["label"] = y_train
            synthetic_quality.append({
                "seed": seed,
                "mean_feature_ks": float(np.mean([
                    ks_2samp(real_train_frame.loc[real_train_frame.label == label, feature],
                             synthetic_df.loc[synthetic_df.label == label, feature]).statistic
                    for label in (0, 1) for feature in FEATURES
                ])),
                "mean_feature_wasserstein_train_sd": float(np.mean([
                    wasserstein_distance(
                        real_train_frame.loc[real_train_frame.label == label, feature],
                        synthetic_df.loc[synthetic_df.label == label, feature],
                    ) / max(float(real_train_frame[feature].std(ddof=1)), 1e-12)
                    for label in (0, 1) for feature in FEATURES
                ])),
                **nearest_neighbor_audit(real_train_frame, synthetic_df, FEATURES),
                "synthetic_class_0": synth_counts.get(0, 0),
                "synthetic_class_1": synth_counts.get(1, 0),
                "combined_class_0": combined_counts[0],
                "combined_class_1": combined_counts[1],
                "invalid_gender_rows": int((~synthetic_df.gender.isin([0, 1])).sum()),
                "invalid_age_rows": int((~synthetic_df.age.between(2, 9) | ~np.isclose(synthetic_df.age, np.round(synthetic_df.age))).sum()),
                "negative_biomarker_rows": int((synthetic_df[FEATURES[2:]] < 0).any(axis=1).sum()),
            })

            # Models receive the same eight raw predictors. TabPFN handles its
            # own default preprocessing. The real-only and augmented estimators
            # have identical constructor parameters and are independently fit.
            X_augmented = np.vstack([X_train, X_synthetic])
            y_augmented = np.concatenate([y_train, generated_y])
            for condition, X_fit, y_fit in (
                ("real_only", X_train, y_train),
                ("real_plus_wgan", X_augmented, y_augmented),
            ):
                seed_everything(seed)
                estimator = TabPFNClassifier()
                if estimator.get_params(deep=True) != tabpfn_params:
                    raise RuntimeError("TabPFN defaults changed during this run; refusing inconsistent comparison.")
                estimator.fit(X_fit, y_fit)
                probabilities = estimator.predict_proba(X_test)[:, 1]
                predictions = estimator.predict(X_test)
                metrics.append({
                    "seed": seed,
                    "condition": condition,
                    "tabpfn_version": package_version,
                    "real_train_rows": int(len(y_train)),
                    "synthetic_rows": 0 if condition == "real_only" else int(len(generated_y)),
                    "training_rows": int(len(y_fit)),
                    "training_class_0": int((y_fit == 0).sum()),
                    "training_class_1": int((y_fit == 1).sum()),
                    "test_rows": int(len(y_test)),
                    **compute_metrics(y_test, predictions, probabilities),
                })

            for row_idx in train_idx:
                manifests.append({
                    "seed": seed,
                    "patient_id": cohort.iloc[row_idx][id_column] if id_column in cohort else int(row_idx),
                    "partition": "train",
                })
            for row_idx in test_idx:
                manifests.append({
                    "seed": seed,
                    "patient_id": cohort.iloc[row_idx][id_column] if id_column in cohort else int(row_idx),
                    "partition": "test",
                })
            pd.DataFrame(metrics).to_csv(os.path.join(metrics_dir, "per_seed_metrics.csv"), index=False)
            pd.DataFrame(synthetic_quality).to_csv(os.path.join(metrics_dir, "synthetic_quality_by_seed.csv"), index=False)
            pd.DataFrame(manifests).to_csv(
                os.path.join(self.get_results_subdir("splits"), "patient_split_manifest.csv"), index=False
            )
            print(f"[{seed_index}/{len(seeds)}] finished seed={seed}", flush=True)

        details = pd.DataFrame(metrics)
        summary = (
            details.groupby("condition", sort=False)
            .agg(
                n_seeds=("seed", "nunique"),
                auc_mean=("roc_auc", "mean"), auc_sd=("roc_auc", "std"),
                pr_auc_mean=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"),
                accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
                f1_mean=("f1_score", "mean"), f1_sd=("f1_score", "std"),
                sensitivity_mean=("recall_sensitivity", "mean"),
                specificity_mean=("specificity", "mean"),
            ).reset_index()
        )
        summary.to_csv(os.path.join(metrics_dir, "summary.csv"), index=False)
        paired = details.pivot(index="seed", columns="condition", values="roc_auc").dropna()
        delta = paired["real_plus_wgan"] - paired["real_only"]
        pd.DataFrame({"seed": delta.index, "augmented_minus_real_auc": delta.values}).to_csv(
            os.path.join(metrics_dir, "paired_auc_difference_by_seed.csv"), index=False
        )
        paired_summary = pd.DataFrame([{
            "n_seeds": len(delta),
            "mean_auc_delta": float(delta.mean()),
            "sd_auc_delta": float(delta.std(ddof=1)),
            "seeds_augmented_higher": int((delta > 0).sum()),
            "seeds_tied": int((delta == 0).sum()),
            "seeds_real_only_higher": int((delta < 0).sum()),
        }])
        paired_summary.to_csv(os.path.join(metrics_dir, "paired_comparison_summary.csv"), index=False)
        print(f"\nExperiment 11 | TabPFN {package_version} (default params)")
        for _, row in summary.iterrows():
            print(f"  {row['condition']}: ROC-AUC {row['auc_mean']:.4f} ± {row['auc_sd']:.4f}")
        print(f"  Mean paired AUC change (real+synthetic minus real-only): {delta.mean():+.4f}; ")
        print(f"  augmented higher/tied/lower: {(delta > 0).sum()}/{(delta == 0).sum()}/{(delta < 0).sum()}")
        return {"summary": summary, "details": details, "paired_auc_delta": delta}


def main():
    Experiment11()()


if __name__ == "__main__":
    main()
