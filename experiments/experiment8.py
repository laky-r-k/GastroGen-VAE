"""Experiment 8: beta=1 comparison of plain VAE+GB and SS-VAE+GB."""

import os
import random
import sys
from typing import Dict, List

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
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
from model_src import (
    SemiSupervisedTabularVAE,
    SemiSupervisedVAETrainer,
    TabularVAE,
    VAETrainer,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Experiment8(BaseExperiment):
    """Compare plain and semi-supervised VAE latent GB at beta=1 on matched splits."""

    experiment_name = "experiment8"

    def run(self):
        cfg = self.config
        torch.set_num_threads(int(cfg.get("runtime", {}).get("torch_num_threads", 1)))
        dataset_cfg = cfg.get("dataset", {})
        split_cfg = cfg.get("split", {})
        preprocess_cfg = cfg.get("preprocessing", {})
        vae_cfg = cfg.get("vae", {})
        ss_vae_cfg = cfg.get("ss_vae", {})
        gb_cfg = cfg.get("gradient_boosting", {})

        source_path = self._resolve_path(
            dataset_cfg.get("raw_path", "dataset/orginal_data/gastric_cancer_dataset_raw.csv")
        )
        label_column = dataset_cfg.get("label_column", "label")
        id_column = dataset_cfg.get("id_column", "patient_id")
        df = pd.read_csv(source_path)
        if label_column not in df:
            raise ValueError(f"Label column {label_column!r} is missing from {source_path}")
        missing = [column for column in PAPER_INPUT_FEATURES if column not in df.columns]
        if missing:
            raise ValueError(f"Raw dataset lacks required paper predictors: {missing}")

        seed_start = int(split_cfg.get("seed_start", 42))
        seed_end = int(split_cfg.get("seed_end", 100))
        if seed_start > seed_end:
            raise ValueError("split.seed_start must be less than or equal to split.seed_end")
        seeds = list(range(seed_start, seed_end + 1))
        test_size = float(split_cfg.get("test_size", 0.30))
        beta = float(cfg.get("beta", 1.0))
        X_all = df[PAPER_INPUT_FEATURES]
        y_all = df[label_column].to_numpy()
        records: List[Dict[str, float]] = []
        split_manifests: List[pd.DataFrame] = []
        manifest_path = os.path.join(self.get_results_subdir("splits"), "patient_split_manifest.csv")
        metrics_dir = self.get_results_subdir("metrics")

        gb_params = {
            "n_estimators": int(gb_cfg.get("n_estimators", 200)),
            "learning_rate": float(gb_cfg.get("learning_rate", 0.1)),
            "max_depth": int(gb_cfg.get("max_depth", 4)),
            "min_samples_split": int(gb_cfg.get("min_samples_split", 4)),
            "min_samples_leaf": int(gb_cfg.get("min_samples_leaf", 2)),
            "max_features": gb_cfg.get("max_features", "sqrt"),
        }

        print(f"Raw cohort: {len(df)} rows; selected model features: {PAPER_SELECTED_FEATURES}")
        print(f"Seeds: {seed_start}–{seed_end} ({len(seeds)} matched splits); holdout: {test_size:.0%}")
        print(f"Beta: {beta}; latent dimension: {vae_cfg.get('latent_dim', 4)}")
        print("Comparing VAE+GB (VAE sees no labels) with SS-VAE+GB (VAE uses training labels).")

        for seed_index, seed in enumerate(seeds, 1):
            train_idx, test_idx = train_test_split(
                np.arange(len(df)),
                test_size=test_size,
                random_state=seed,
                shuffle=True,
                stratify=y_all,
            )
            X_train_raw = X_all.iloc[train_idx]
            X_test_raw = X_all.iloc[test_idx]
            y_train = y_all[train_idx]
            y_test = y_all[test_idx]

            if id_column in df.columns:
                split_manifests.append(pd.DataFrame({
                    "seed": seed,
                    "patient_id": df.iloc[np.concatenate([train_idx, test_idx])][id_column].to_numpy(),
                    "partition": ["train"] * len(train_idx) + ["test"] * len(test_idx),
                }))

            preprocessor = PaperFeaturePreprocessor(
                n_clusters=int(preprocess_cfg.get("kmeans_clusters", 5)),
                random_state=seed,
            )
            X_train_selected = preprocessor.fit_transform(X_train_raw)
            X_test_selected = preprocessor.transform(X_test_raw)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            x_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
            x_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

            # Plain VAE: fit only to training features, never labels.
            seed_everything(seed)
            plain_vae = TabularVAE(
                in_features=len(PAPER_SELECTED_FEATURES),
                latent_dim=int(vae_cfg.get("latent_dim", 4)),
                hidden_dim1=int(vae_cfg.get("hidden_dim1", 128)),
                hidden_dim2=int(vae_cfg.get("hidden_dim2", 64)),
                dropout_rate=float(vae_cfg.get("dropout_rate", 0.1)),
            )
            plain_trainer = VAETrainer(
                model=plain_vae,
                lr=float(vae_cfg.get("learning_rate", 1e-3)),
                weight_decay=float(vae_cfg.get("weight_decay", 1e-5)),
                beta=beta,
                device=str(vae_cfg.get("device", "cpu")),
            )
            plain_trainer.fit(
                X_train_scaled,
                epochs=int(vae_cfg.get("epochs", 100)),
                batch_size=int(vae_cfg.get("batch_size", 64)),
                verbose=False,
            )
            with torch.no_grad():
                z_train_plain = plain_vae.get_latent(x_train_tensor, deterministic=True).cpu().numpy()
                z_test_plain = plain_vae.get_latent(x_test_tensor, deterministic=True).cpu().numpy()

            plain_gb = GradientBoostingClassifier(**gb_params, random_state=seed)
            plain_gb.fit(z_train_plain, y_train)
            plain_prob = plain_gb.predict_proba(z_test_plain)[:, 1]
            plain_metrics = compute_metrics(y_test, plain_gb.predict(z_test_plain), plain_prob)
            records.append({
                "seed": seed,
                "beta": beta,
                "model": "VAE (z=4) + GB",
                **plain_metrics,
            })

            # SS-VAE: same split and inputs; training labels guide the encoder.
            seed_everything(seed)
            ss_vae = SemiSupervisedTabularVAE(
                in_features=len(PAPER_SELECTED_FEATURES),
                latent_dim=int(ss_vae_cfg.get("latent_dim", 4)),
                hidden_dim1=int(ss_vae_cfg.get("hidden_dim1", 128)),
                hidden_dim2=int(ss_vae_cfg.get("hidden_dim2", 64)),
                classifier_hidden=int(ss_vae_cfg.get("classifier_hidden", 32)),
                dropout_rate=float(ss_vae_cfg.get("dropout_rate", 0.1)),
            )
            ss_trainer = SemiSupervisedVAETrainer(
                model=ss_vae,
                lr=float(ss_vae_cfg.get("learning_rate", 1e-3)),
                weight_decay=float(ss_vae_cfg.get("weight_decay", 1e-5)),
                beta=beta,
                alpha=float(ss_vae_cfg.get("alpha", 10.0)),
                device=str(ss_vae_cfg.get("device", "cpu")),
            )
            ss_trainer.fit(
                X_train=X_train_scaled,
                y_train=y_train,
                epochs=int(ss_vae_cfg.get("epochs", 100)),
                batch_size=int(ss_vae_cfg.get("batch_size", 64)),
                verbose=False,
            )
            with torch.no_grad():
                z_train_ss = ss_vae.get_latent(x_train_tensor, deterministic=True).cpu().numpy()
                z_test_ss = ss_vae.get_latent(x_test_tensor, deterministic=True).cpu().numpy()

            ss_gb = GradientBoostingClassifier(**gb_params, random_state=seed)
            ss_gb.fit(z_train_ss, y_train)
            ss_prob = ss_gb.predict_proba(z_test_ss)[:, 1]
            ss_metrics = compute_metrics(y_test, ss_gb.predict(z_test_ss), ss_prob)
            records.append({
                "seed": seed,
                "beta": beta,
                "model": "SS-VAE (z=4) + GB",
                **ss_metrics,
            })

            # Checkpoint after every split for safe recovery and progress review.
            detail_df = pd.DataFrame(records)
            detail_df.to_csv(os.path.join(metrics_dir, "per_seed_metrics.csv"), index=False)
            paired = detail_df.pivot(index="seed", columns="model", values="roc_auc").dropna()
            if len(paired):
                delta = paired["SS-VAE (z=4) + GB"] - paired["VAE (z=4) + GB"]
                pd.DataFrame({
                    "seed": delta.index,
                    "ss_vae_minus_vae_auc": delta.values,
                }).to_csv(os.path.join(metrics_dir, "paired_auc_difference_by_seed.csv"), index=False)
            if split_manifests:
                pd.concat(split_manifests, ignore_index=True).to_csv(manifest_path, index=False)
            print(f"[{seed_index}/{len(seeds)}] completed seed={seed}", flush=True)

        details = pd.DataFrame(records)
        summary = (
            details.groupby("model", sort=False)
            .agg(
                n_seeds=("seed", "nunique"),
                auc_mean=("roc_auc", "mean"),
                auc_sd=("roc_auc", "std"),
                pr_auc_mean=("pr_auc", "mean"),
                accuracy_mean=("accuracy", "mean"),
                f1_mean=("f1_score", "mean"),
                sensitivity_mean=("recall_sensitivity", "mean"),
                specificity_mean=("specificity", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(os.path.join(metrics_dir, "summary.csv"), index=False)

        paired = details.pivot(index="seed", columns="model", values="roc_auc").dropna()
        deltas = paired["SS-VAE (z=4) + GB"] - paired["VAE (z=4) + GB"]
        pd.DataFrame([{
            "beta": beta,
            "n_seeds": int(len(deltas)),
            "mean_ss_vae_minus_vae_auc": float(deltas.mean()),
            "sd_paired_auc_difference": float(deltas.std(ddof=1)),
            "seeds_ss_vae_higher": int((deltas > 0).sum()),
            "seeds_tied": int((deltas == 0).sum()),
            "seeds_vae_higher": int((deltas < 0).sum()),
        }]).to_csv(os.path.join(metrics_dir, "paired_comparison_summary.csv"), index=False)

        plt.figure(figsize=(8, 6), dpi=180)
        sns.boxplot(data=details, x="model", y="roc_auc", color="#8fb6d8")
        sns.stripplot(data=details, x="model", y="roc_auc", color="#263746", alpha=0.45, size=3)
        plt.ylabel("Holdout ROC-AUC")
        plt.xlabel("")
        plt.title("Experiment 8: VAE + GB vs SS-VAE + GB (β=1)")
        plt.ylim(0.5, 1.0)
        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir, "holdout_auc_comparison.png"))
        plt.close()

        print(f"\nExperiment 8 summary (holdout AUC mean ± SD; beta={beta}):")
        for _, row in summary.iterrows():
            print(f"  {row['model']}: {row['auc_mean']:.4f} ± {row['auc_sd']:.4f}")
        print(f"  Paired SS-VAE minus VAE AUC: {deltas.mean():+.4f} ± {deltas.std(ddof=1):.4f}")
        return {"summary": summary, "details": details, "paired_auc_delta": deltas}


def main():
    Experiment8()()


if __name__ == "__main__":
    main()
