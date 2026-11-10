"""Experiment 10: class-balanced real + WGAN-GP data on paper features."""

import json
import os
import random
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_handling.paper_preprocessing import (
    PAPER_INPUT_FEATURES,
    PAPER_SELECTED_FEATURES,
    PaperFeaturePreprocessor,
)
from evaluation import compute_metrics
from experiments.base_experiment import BaseExperiment
from experiments.experiment9 import (
    FEATURES,
    LOG_FEATURES,
    export_frame,
    inverse_generator_transform,
    nearest_neighbor_audit,
    postprocess_samples,
    transform_for_generator,
)
from model_src import (
    Critic,
    Generator,
    SemiSupervisedTabularVAE,
    SemiSupervisedVAETrainer,
    TabularVAE,
    VAETrainer,
    WGANGPTrainer,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _class_counts(labels: np.ndarray) -> Dict[int, int]:
    values = pd.Series(np.asarray(labels, dtype=int)).value_counts().sort_index()
    return {int(label): int(count) for label, count in values.items()}


def _quality_metrics(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    seed: int,
) -> tuple[List[Dict[str, float]], Dict[str, float]]:
    """Compare synthetic rows to this seed's real training partition only."""
    marginal_rows: List[Dict[str, float]] = []
    class_corr_distances = []
    for label in sorted(real_train.label.unique()):
        real_class = real_train.loc[real_train.label == label, FEATURES]
        syn_class = synthetic.loc[synthetic.label == label, FEATURES]
        for feature in FEATURES:
            real_values = real_class[feature].to_numpy(dtype=float)
            syn_values = syn_class[feature].to_numpy(dtype=float)
            ks = ks_2samp(real_values, syn_values)
            wd = float(wasserstein_distance(real_values, syn_values))
            marginal_rows.append({
                "seed": seed,
                "label": int(label),
                "feature": feature,
                "n_real_train": int(len(real_values)),
                "n_synthetic": int(len(syn_values)),
                "real_train_mean": float(np.mean(real_values)),
                "synthetic_mean": float(np.mean(syn_values)),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue_descriptive": float(ks.pvalue),
                "wasserstein_distance": wd,
                "wasserstein_distance_in_train_sd": wd / max(float(real_train[feature].std(ddof=1)), 1e-12),
            })
        real_corr = real_class.corr(method="spearman").fillna(0).to_numpy()
        syn_corr = syn_class.corr(method="spearman").fillna(0).to_numpy()
        class_corr_distances.append(float(np.linalg.norm(real_corr - syn_corr, ord="fro")))

    nn = nearest_neighbor_audit(real_train, synthetic, FEATURES)
    syn_x = synthetic[FEATURES]
    validity = {
        "invalid_gender_rows": int((~syn_x.gender.isin([0, 1])).sum()),
        "invalid_age_rows": int((~syn_x.age.between(2, 9) | ~np.isclose(syn_x.age, np.round(syn_x.age))).sum()),
        "negative_biomarker_rows": int((syn_x[FEATURES[2:]] < 0).any(axis=1).sum()),
    }
    counts = _class_counts(synthetic.label.to_numpy())
    summary = {
        "seed": seed,
        "synthetic_rows": int(len(synthetic)),
        "synthetic_class_0": counts.get(0, 0),
        "synthetic_class_1": counts.get(1, 0),
        "mean_class_conditional_ks": float(np.mean([r["ks_statistic"] for r in marginal_rows])),
        "mean_class_conditional_wasserstein_train_sd": float(np.mean([
            r["wasserstein_distance_in_train_sd"] for r in marginal_rows
        ])),
        "mean_spearman_corr_frobenius_distance": float(np.mean(class_corr_distances)),
        **validity,
        **nn,
    }
    return marginal_rows, summary


class Experiment10(BaseExperiment):
    """Train GB, VAE+GB, and SS-VAE+GB on balanced real + WGAN data."""

    experiment_name = "experiment10"

    def run(self):
        cfg = self.config
        torch.set_num_threads(int(cfg.get("runtime", {}).get("torch_num_threads", 1)))
        dataset_cfg = cfg.get("dataset", {})
        split_cfg = cfg.get("split", {})
        prep_cfg = cfg.get("preprocessing", {})
        wgan_cfg = cfg.get("wgan_gp", {})
        vae_cfg = cfg.get("vae", {})
        ss_cfg = cfg.get("ss_vae", {})

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
        x_all_raw = cohort[PAPER_INPUT_FEATURES]
        xgb_params = {
            "n_estimators": int(cfg.get("gradient_boosting", {}).get("n_estimators", 200)),
            "learning_rate": float(cfg.get("gradient_boosting", {}).get("learning_rate", 0.1)),
            "max_depth": int(cfg.get("gradient_boosting", {}).get("max_depth", 4)),
            "min_samples_split": int(cfg.get("gradient_boosting", {}).get("min_samples_split", 4)),
            "min_samples_leaf": int(cfg.get("gradient_boosting", {}).get("min_samples_leaf", 2)),
            "max_features": cfg.get("gradient_boosting", {}).get("max_features", "sqrt"),
        }
        paper_gb_params = {
            "n_estimators": int(cfg.get("paper_gradient_boosting", {}).get("n_estimators", 70)),
            "learning_rate": float(cfg.get("paper_gradient_boosting", {}).get("learning_rate", 0.12)),
            "max_depth": int(cfg.get("paper_gradient_boosting", {}).get("max_depth", 5)),
            "min_samples_split": int(cfg.get("paper_gradient_boosting", {}).get("min_samples_split", 2)),
            "min_samples_leaf": int(cfg.get("paper_gradient_boosting", {}).get("min_samples_leaf", 10)),
            "max_features": cfg.get("paper_gradient_boosting", {}).get("max_features", None),
        }
        beta = float(cfg.get("beta", 0.0005))
        metrics_dir = self.get_results_subdir("metrics")
        synth_root = self._resolve_path(cfg.get(
            "synthetic_output_dir", "dataset/synthetic_8_features/experiment10"
        ))
        os.makedirs(synth_root, exist_ok=True)

        print(f"Raw cohort: {len(cohort)} rows; selected features: {PAPER_SELECTED_FEATURES}")
        print(f"Seeds: {seed_start}–{seed_end} ({len(seeds)} stratified splits); holdout: {test_size:.0%}")
        print("Training condition: real training rows + same-sized WGAN-GP synthetic rows; combined labels balanced.")
        print(f"VAE beta={beta}; WGAN generator steps={wgan_cfg.get('generator_steps', 5000)}")

        records: List[Dict[str, float]] = []
        manifest_rows = []
        quality_rows = []
        marginal_rows = []

        for seed_index, seed in enumerate(seeds, 1):
            seed_everything(seed)
            train_idx, test_idx = train_test_split(
                np.arange(len(cohort)), test_size=test_size, random_state=seed,
                shuffle=True, stratify=y_all,
            )
            y_train = y_all[train_idx]
            y_test = y_all[test_idx]
            x_train_raw = x_all_raw.iloc[train_idx]
            x_test_raw = x_all_raw.iloc[test_idx]

            preprocessor = PaperFeaturePreprocessor(
                n_clusters=int(prep_cfg.get("kmeans_clusters", 5)), random_state=seed
            )
            X_train_real_df = preprocessor.fit_transform(x_train_raw)
            X_test_df = preprocessor.transform(x_test_raw)
            X_train_real = X_train_real_df[FEATURES].to_numpy(dtype=float)
            X_test = X_test_df[FEATURES].to_numpy(dtype=float)

            real_counts = _class_counts(y_train)
            if set(real_counts) != {0, 1}:
                raise ValueError(f"Seed {seed} does not contain both binary labels: {real_counts}")
            # Generate a synthetic copy of the real training set, reversing its
            # class counts so the combined real + synthetic training set is 50/50.
            synth_counts = {0: real_counts[1], 1: real_counts[0]}
            combined_counts = {label: real_counts[label] + synth_counts[label] for label in (0, 1)}
            if combined_counts[0] != combined_counts[1]:
                raise RuntimeError(f"Seed {seed}: augmentation failed to balance labels: {combined_counts}")

            X_gen = transform_for_generator(X_train_real)
            gen_scaler = StandardScaler().fit(X_gen)
            X_gen_scaled = gen_scaler.transform(X_gen)
            seed_everything(seed)
            generator = Generator(
                latent_dim=int(wgan_cfg.get("latent_dim", 16)), num_classes=2,
                output_dim=len(FEATURES), hidden_dim=int(wgan_cfg.get("hidden_dim", 128)),
                use_layer_norm=bool(wgan_cfg.get("use_layer_norm", True)), use_batch_norm=False,
            )
            critic = Critic(
                input_dim=len(FEATURES), num_classes=2,
                hidden_dim=int(wgan_cfg.get("hidden_dim", 128)),
                use_layer_norm=bool(wgan_cfg.get("use_layer_norm", True)),
            )
            trainer = WGANGPTrainer(
                generator=generator, critic=critic,
                lr_g=float(wgan_cfg.get("learning_rate_g", 1e-4)),
                lr_d=float(wgan_cfg.get("learning_rate_d", 1e-4)),
                betas=tuple(wgan_cfg.get("betas", [0.5, 0.9])),
                lambda_gp=float(wgan_cfg.get("lambda_gp", 10.0)),
                n_critic=int(wgan_cfg.get("n_critic", 5)),
                device=str(wgan_cfg.get("device", "cpu")),
            )
            print(f"[{seed_index}/{len(seeds)}] seed={seed}: fitting WGAN-GP on {len(y_train)} real training rows; generating {sum(synth_counts.values())} rows", flush=True)
            trainer.fit(
                X_train=X_gen_scaled, y_train=y_train,
                generator_steps=int(wgan_cfg.get("generator_steps", 5000)),
                batch_size=int(wgan_cfg.get("batch_size", 64)),
                checkpoint_interval=int(wgan_cfg.get("checkpoint_interval", 500)),
                verbose=bool(wgan_cfg.get("verbose", False)),
            )
            generated_scaled = trainer.generate(synth_counts)
            generated_parts = [generated_scaled[label] for label in sorted(generated_scaled)]
            generated_labels = np.concatenate([
                np.full(synth_counts[label], label, dtype=int) for label in sorted(synth_counts)
            ])
            X_synthetic = inverse_generator_transform(np.vstack(generated_parts), gen_scaler)
            X_synthetic, corrections = postprocess_samples(X_synthetic)
            synthetic_df = export_frame(X_synthetic, generated_labels, "wgan_gp_balancing")

            real_train_df = pd.DataFrame(X_train_real, columns=FEATURES)
            real_train_df["label"] = y_train
            class_marginals, quality = _quality_metrics(real_train_df, synthetic_df, seed)
            quality.update({
                "real_train_rows": len(y_train),
                "real_train_class_0": real_counts[0],
                "real_train_class_1": real_counts[1],
                "synthetic_class_0_requested": synth_counts[0],
                "synthetic_class_1_requested": synth_counts[1],
                "combined_class_0": combined_counts[0],
                "combined_class_1": combined_counts[1],
                "correction_counts": json.dumps(corrections, sort_keys=True),
            })
            quality_rows.append(quality)
            marginal_rows.extend(class_marginals)

            seed_dir = os.path.join(synth_root, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            synthetic_path = os.path.join(seed_dir, "wgan_gp_balancing.csv")
            synthetic_df.to_csv(synthetic_path, index=False)
            metadata = {
                "experiment": "experiment10",
                "seed": seed,
                "train_rows": int(len(y_train)),
                "test_rows": int(len(y_test)),
                "features": FEATURES,
                "label_column": label_column,
                "generator_fit_scope": "this seed's real training partition only",
                "generator_transforms": {"log1p": LOG_FEATURES, "scaler": "StandardScaler fitted on this seed's transformed training partition"},
                "wgan_gp": wgan_cfg,
                "real_train_class_counts": {str(k): v for k, v in real_counts.items()},
                "synthetic_class_counts": {str(k): v for k, v in synth_counts.items()},
                "combined_class_counts": {str(k): v for k, v in combined_counts.items()},
                "synthetic_postprocessing_adjustments": corrections,
                "quality_reference": "same seed's real training partition; held-out test is not used for synthetic quality checks",
                "synthetic_csv": os.path.relpath(synthetic_path, PROJECT_ROOT),
            }
            with open(os.path.join(seed_dir, "generation_metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            pd.DataFrame(class_marginals).to_csv(os.path.join(seed_dir, "quality_by_feature.csv"), index=False)

            # All models use the identical real + synthetic training rows.
            X_aug = np.vstack([X_train_real, X_synthetic])
            y_aug = np.concatenate([y_train, generated_labels])
            if not np.array_equal(np.sort(np.unique(y_test)), np.array([0, 1])):
                raise ValueError(f"Seed {seed} held-out partition is missing a class")
            data_meta = {
                "real_train_rows": int(len(y_train)),
                "synthetic_rows": int(len(generated_labels)),
                "combined_train_rows": int(len(y_aug)),
                "real_train_class_0": real_counts[0],
                "real_train_class_1": real_counts[1],
                "synthetic_class_0": synth_counts[0],
                "synthetic_class_1": synth_counts[1],
                "combined_class_0": combined_counts[0],
                "combined_class_1": combined_counts[1],
            }

            # 1) Paper-config Gradient Boosting directly on the eight selected features.
            raw_gb = GradientBoostingClassifier(**paper_gb_params, random_state=seed)
            raw_gb.fit(X_aug, y_aug)
            probabilities = raw_gb.predict_proba(X_test)[:, 1]
            records.append({
                "seed": seed, "model": "Raw 8-feature GB + WGAN-GP",
                "beta": np.nan, "gb_parameter_set": "paper",
                **data_meta,
                **compute_metrics(y_test, raw_gb.predict(X_test), probabilities),
            })

            # Scale from real training rows only, preserving a train-fitted map
            # for the VAE inputs as in Experiments 5, 7, and 8.
            model_scaler = StandardScaler().fit(X_train_real)
            X_train_real_scaled = model_scaler.transform(X_train_real)
            X_synthetic_scaled = model_scaler.transform(X_synthetic)
            X_test_scaled = model_scaler.transform(X_test)
            X_aug_scaled = np.vstack([X_train_real_scaled, X_synthetic_scaled])
            x_aug_tensor = torch.tensor(X_aug_scaled, dtype=torch.float32)
            x_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

            # 2) Unsupervised VAE + project-config GB; labels are not used to fit VAE.
            seed_everything(seed)
            vae = TabularVAE(
                in_features=len(FEATURES), latent_dim=int(vae_cfg.get("latent_dim", 4)),
                hidden_dim1=int(vae_cfg.get("hidden_dim1", 128)),
                hidden_dim2=int(vae_cfg.get("hidden_dim2", 64)),
                dropout_rate=float(vae_cfg.get("dropout_rate", 0.1)),
            )
            vae_trainer = VAETrainer(
                model=vae, lr=float(vae_cfg.get("learning_rate", 1e-3)),
                weight_decay=float(vae_cfg.get("weight_decay", 1e-5)), beta=beta,
                device=str(vae_cfg.get("device", "cpu")),
            )
            vae_trainer.fit(
                X_aug_scaled, epochs=int(vae_cfg.get("epochs", 100)),
                batch_size=int(vae_cfg.get("batch_size", 64)), verbose=False,
            )
            vae.eval()
            with torch.no_grad():
                z_aug = vae.get_latent(x_aug_tensor, deterministic=True).cpu().numpy()
                z_test = vae.get_latent(x_test_tensor, deterministic=True).cpu().numpy()
            vae_gb = GradientBoostingClassifier(**xgb_params, random_state=seed)
            vae_gb.fit(z_aug, y_aug)
            probabilities = vae_gb.predict_proba(z_test)[:, 1]
            records.append({
                "seed": seed, "model": "VAE (z=4) + GB + WGAN-GP",
                "beta": beta, "gb_parameter_set": "project_latent",
                **data_meta,
                **compute_metrics(y_test, vae_gb.predict(z_test), probabilities),
            })

            # 3) Semi-supervised VAE + project-config GB; training labels guide SS-VAE.
            seed_everything(seed)
            ss_vae = SemiSupervisedTabularVAE(
                in_features=len(FEATURES), latent_dim=int(ss_cfg.get("latent_dim", 4)),
                hidden_dim1=int(ss_cfg.get("hidden_dim1", 128)),
                hidden_dim2=int(ss_cfg.get("hidden_dim2", 64)),
                classifier_hidden=int(ss_cfg.get("classifier_hidden", 32)),
                dropout_rate=float(ss_cfg.get("dropout_rate", 0.1)),
            )
            ss_trainer = SemiSupervisedVAETrainer(
                model=ss_vae, lr=float(ss_cfg.get("learning_rate", 1e-3)),
                weight_decay=float(ss_cfg.get("weight_decay", 1e-5)), beta=beta,
                alpha=float(ss_cfg.get("alpha", 10.0)), device=str(ss_cfg.get("device", "cpu")),
            )
            ss_trainer.fit(
                X_train=X_aug_scaled, y_train=y_aug,
                epochs=int(ss_cfg.get("epochs", 100)),
                batch_size=int(ss_cfg.get("batch_size", 64)), verbose=False,
            )
            ss_vae.eval()
            with torch.no_grad():
                z_aug_ss = ss_vae.get_latent(x_aug_tensor, deterministic=True).cpu().numpy()
                z_test_ss = ss_vae.get_latent(x_test_tensor, deterministic=True).cpu().numpy()
            ss_gb = GradientBoostingClassifier(**xgb_params, random_state=seed)
            ss_gb.fit(z_aug_ss, y_aug)
            probabilities = ss_gb.predict_proba(z_test_ss)[:, 1]
            records.append({
                "seed": seed, "model": "SS-VAE (z=4) + GB + WGAN-GP",
                "beta": beta, "gb_parameter_set": "project_latent",
                **data_meta,
                **compute_metrics(y_test, ss_gb.predict(z_test_ss), probabilities),
            })

            # Save per-seed predictions and patient split membership for auditability.
            for row_idx in train_idx:
                manifest_rows.append({
                    "seed": seed, "patient_id": cohort.iloc[row_idx][id_column] if id_column in cohort else row_idx,
                    "partition": "train",
                })
            for row_idx in test_idx:
                manifest_rows.append({
                    "seed": seed, "patient_id": cohort.iloc[row_idx][id_column] if id_column in cohort else row_idx,
                    "partition": "test",
                })
            pd.DataFrame(records).to_csv(os.path.join(metrics_dir, "per_seed_metrics.csv"), index=False)
            pd.DataFrame(quality_rows).to_csv(os.path.join(metrics_dir, "synthetic_quality_by_seed.csv"), index=False)
            pd.DataFrame(marginal_rows).to_csv(os.path.join(metrics_dir, "synthetic_quality_by_feature.csv"), index=False)
            pd.DataFrame(manifest_rows).to_csv(
                os.path.join(self.get_results_subdir("splits"), "patient_split_manifest.csv"), index=False
            )
            print(f"[{seed_index}/{len(seeds)}] finished seed={seed}", flush=True)

        details = pd.DataFrame(records)
        summary = (
            details.groupby(["model", "beta", "gb_parameter_set"], dropna=False, sort=False)
            .agg(
                n_seeds=("seed", "nunique"),
                auc_mean=("roc_auc", "mean"), auc_sd=("roc_auc", "std"),
                pr_auc_mean=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"),
                accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
                f1_mean=("f1_score", "mean"), f1_sd=("f1_score", "std"),
                sensitivity_mean=("recall_sensitivity", "mean"), sensitivity_sd=("recall_sensitivity", "std"),
                specificity_mean=("specificity", "mean"), specificity_sd=("specificity", "std"),
            ).reset_index()
        )
        max_rows = details.loc[details.groupby("model")["roc_auc"].idxmax(), ["model", "seed", "roc_auc"]]
        summary = summary.merge(max_rows.rename(columns={"seed": "best_auc_seed", "roc_auc": "best_auc"}), on="model")
        summary.to_csv(os.path.join(metrics_dir, "summary.csv"), index=False)
        pd.DataFrame(quality_rows).to_csv(os.path.join(metrics_dir, "synthetic_quality_by_seed.csv"), index=False)
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.figure(figsize=(10, 6), dpi=180)
            sns.boxplot(data=details, x="model", y="roc_auc", color="#8fb6d8")
            sns.stripplot(data=details, x="model", y="roc_auc", color="#263746", alpha=0.45, size=3)
            plt.ylabel("Holdout ROC-AUC")
            plt.xlabel("")
            plt.xticks(rotation=12, ha="right")
            plt.title("Experiment 10: models trained on balanced real + WGAN-GP data")
            plt.ylim(0.5, 1.0)
            plt.tight_layout()
            plt.savefig(os.path.join(metrics_dir, "holdout_auc_comparison.png"))
            plt.close()
        except ImportError:
            print("Plotting skipped: matplotlib or seaborn is unavailable.")
        print("\nExperiment 10 summary (holdout ROC-AUC mean ± SD):")
        for _, row in summary.iterrows():
            print(f"  {row['model']}: {row['auc_mean']:.4f} ± {row['auc_sd']:.4f}; best={row['best_auc']:.4f} seed={int(row['best_auc_seed'])}")
        return {"summary": summary, "details": details, "synthetic_quality": pd.DataFrame(quality_rows)}


def main():
    Experiment10()()


if __name__ == "__main__":
    main()
