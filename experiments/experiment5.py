"""Experiment 5: SS-VAE + Gradient Boosting beta sweep on paper features."""

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
from model_src import SemiSupervisedTabularVAE, SemiSupervisedVAETrainer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Experiment5(BaseExperiment):
    """Compare SS-VAE latent GB across beta values and stratified split seeds."""

    experiment_name = "experiment5"

    def run(self):
        cfg = self.config
        torch.set_num_threads(int(cfg.get("runtime", {}).get("torch_num_threads", 1)))
        dataset_cfg = cfg.get("dataset", {})
        split_cfg = cfg.get("split", {})
        preprocess_cfg = cfg.get("preprocessing", {})
        vae_cfg = cfg.get("ss_vae", {})
        gb_cfg = cfg.get("gradient_boosting", {})

        source_path = self._resolve_path(
            dataset_cfg.get("raw_path", "dataset/orginal_data/gastric_cancer_dataset_raw.csv")
        )
        label_column = dataset_cfg.get("label_column", "label")
        id_column = dataset_cfg.get("id_column", "patient_id")
        df = pd.read_csv(source_path)
        if label_column not in df:
            raise ValueError(f"Label column {label_column!r} is missing from {source_path}")

        seed_start = int(split_cfg.get("seed_start", 42))
        seed_end = int(split_cfg.get("seed_end", 100))
        if seed_start > seed_end:
            raise ValueError("split.seed_start must be less than or equal to split.seed_end")
        seeds = list(range(seed_start, seed_end + 1))
        test_size = float(split_cfg.get("test_size", 0.30))
        beta_values = [float(beta) for beta in vae_cfg.get("beta_values", [0.0005, 0.005, 0.05, 0.5])]

        missing = [column for column in PAPER_INPUT_FEATURES if column not in df.columns]
        if missing:
            raise ValueError(f"Raw dataset lacks required paper predictors: {missing}")

        X_all = df[PAPER_INPUT_FEATURES]
        y_all = df[label_column].to_numpy()
        all_records: List[Dict[str, float]] = []
        split_manifests: List[pd.DataFrame] = []
        manifest_path = os.path.join(self.get_results_subdir("splits"), "patient_split_manifest.csv")
        metrics_dir = self.get_results_subdir("metrics")

        vae_gb_params = {
            "n_estimators": int(gb_cfg.get("n_estimators", 200)),
            "learning_rate": float(gb_cfg.get("learning_rate", 0.1)),
            "max_depth": int(gb_cfg.get("max_depth", 4)),
            "min_samples_split": int(gb_cfg.get("min_samples_split", 4)),
            "min_samples_leaf": int(gb_cfg.get("min_samples_leaf", 2)),
            "max_features": gb_cfg.get("max_features", "sqrt"),
        }
        raw_gb_params = {
            "n_estimators": 70,
            "learning_rate": 0.12,
            "max_depth": 5,
            "min_samples_leaf": 10,
            "min_samples_split": 2,
            "max_features": None,
        }

        print(f"Raw cohort: {len(df)} rows; input predictors: {len(PAPER_INPUT_FEATURES)}")
        print(f"Selected predictors: {PAPER_SELECTED_FEATURES}")
        print(f"Seeds: {seed_start}–{seed_end} ({len(seeds)} splits); betas: {beta_values}")
        print(f"VAE latent dimension: {vae_cfg.get('latent_dim', 4)}")

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
                split_manifests.append(
                    pd.DataFrame(
                        {
                            "seed": seed,
                            "patient_id": df.iloc[np.concatenate([train_idx, test_idx])][id_column].to_numpy(),
                            "partition": ["train"] * len(train_idx) + ["test"] * len(test_idx),
                        }
                    )
                )

            # Fit all learned preprocessing on this seed's training split only.
            preprocessor = PaperFeaturePreprocessor(
                n_clusters=int(preprocess_cfg.get("kmeans_clusters", 5)),
                random_state=seed,
            )
            X_train_selected = preprocessor.fit_transform(X_train_raw)
            X_test_selected = preprocessor.transform(X_test_raw)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)

            # Matched raw eight-feature baseline, evaluated on the same split.
            raw_model = GradientBoostingClassifier(**raw_gb_params, random_state=seed)
            raw_model.fit(X_train_selected, y_train)
            raw_prob = raw_model.predict_proba(X_test_selected)[:, 1]
            raw_pred = raw_model.predict(X_test_selected)
            raw_metrics = compute_metrics(y_test, raw_pred, raw_prob)
            all_records.append(
                {"seed": seed, "beta": "baseline", "model": "Raw 8-feature GB", **raw_metrics}
            )

            for beta in beta_values:
                # Reset to the same initialization and minibatch order for each
                # beta within a seed, making beta the intended changing factor.
                seed_everything(seed)
                model = SemiSupervisedTabularVAE(
                    in_features=len(PAPER_SELECTED_FEATURES),
                    latent_dim=int(vae_cfg.get("latent_dim", 4)),
                    hidden_dim1=int(vae_cfg.get("hidden_dim1", 128)),
                    hidden_dim2=int(vae_cfg.get("hidden_dim2", 64)),
                    classifier_hidden=int(vae_cfg.get("classifier_hidden", 32)),
                    dropout_rate=float(vae_cfg.get("dropout_rate", 0.1)),
                )
                trainer = SemiSupervisedVAETrainer(
                    model=model,
                    lr=float(vae_cfg.get("learning_rate", 1e-3)),
                    weight_decay=float(vae_cfg.get("weight_decay", 1e-5)),
                    beta=beta,
                    alpha=float(vae_cfg.get("alpha", 10.0)),
                    device=str(vae_cfg.get("device", "cpu")),
                )
                trainer.fit(
                    X_train=X_train_scaled,
                    y_train=y_train,
                    epochs=int(vae_cfg.get("epochs", 100)),
                    batch_size=int(vae_cfg.get("batch_size", 64)),
                    verbose=False,
                )

                model.eval()
                with torch.no_grad():
                    z_train = model.get_latent(
                        torch.tensor(X_train_scaled, dtype=torch.float32), deterministic=True
                    ).cpu().numpy()
                    z_test = model.get_latent(
                        torch.tensor(X_test_scaled, dtype=torch.float32), deterministic=True
                    ).cpu().numpy()

                gb = GradientBoostingClassifier(**vae_gb_params, random_state=seed)
                gb.fit(z_train, y_train)
                y_prob = gb.predict_proba(z_test)[:, 1]
                y_pred = gb.predict(z_test)
                metrics = compute_metrics(y_test, y_pred, y_prob)
                all_records.append(
                    {"seed": seed, "beta": beta, "model": "SS-VAE (z=4) + GB", **metrics}
                )

            # Checkpoint aggregate outputs after each split so a long run can resume safely.
            detail_df = pd.DataFrame(all_records)
            detail_df.to_csv(os.path.join(metrics_dir, "per_seed_metrics.csv"), index=False)
            if split_manifests:
                pd.concat(split_manifests, ignore_index=True).to_csv(manifest_path, index=False)
            print(f"[{seed_index}/{len(seeds)}] completed seed={seed}", flush=True)

        details = pd.DataFrame(all_records)
        summary = (
            details.groupby(["model", "beta"], sort=False)
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
        summary.to_csv(os.path.join(metrics_dir, "summary_by_beta.csv"), index=False)

        baseline = details[details["model"] == "Raw 8-feature GB"].set_index("seed")["roc_auc"]
        paired_rows = []
        for beta in beta_values:
            beta_df = details[(details["model"] == "SS-VAE (z=4) + GB") & (details["beta"] == beta)]
            paired_delta = beta_df.set_index("seed")["roc_auc"] - baseline
            paired_rows.append(
                {
                    "beta": beta,
                    "mean_auc_delta_vs_raw_gb": float(paired_delta.mean()),
                    "sd_auc_delta": float(paired_delta.std(ddof=1)),
                    "seeds_ssvae_higher": int((paired_delta > 0).sum()),
                    "seeds_tied": int((paired_delta == 0).sum()),
                    "seeds_ssvae_lower": int((paired_delta < 0).sum()),
                }
            )
        pd.DataFrame(paired_rows).to_csv(
            os.path.join(metrics_dir, "paired_auc_difference_vs_raw_gb.csv"), index=False
        )

        plot_data = details.copy()
        plot_data["condition"] = plot_data.apply(
            lambda row: "Raw 8-feature GB" if row["model"] == "Raw 8-feature GB" else f"SS-VAE + GB (β={row['beta']})",
            axis=1,
        )
        plt.figure(figsize=(11, 6), dpi=180)
        order = ["Raw 8-feature GB"] + [f"SS-VAE + GB (β={beta})" for beta in beta_values]
        sns.boxplot(data=plot_data, x="condition", y="roc_auc", order=order, color="#8fb6d8")
        sns.stripplot(data=plot_data, x="condition", y="roc_auc", order=order, color="#263746", alpha=0.55, size=3)
        plt.ylabel("Holdout ROC-AUC")
        plt.xlabel("")
        plt.title("Experiment 5: Holdout AUC across 70/30 split seeds")
        plt.xticks(rotation=15, ha="right")
        plt.ylim(0.5, 1.0)
        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir, "holdout_auc_by_beta.png"))
        plt.close()

        print("\nExperiment 5 summary (holdout AUC mean ± SD across seeds):")
        for _, row in summary.iterrows():
            print(f"  {row['model']} | beta={row['beta']}: {row['auc_mean']:.4f} ± {row['auc_sd']:.4f}")
        print(f"\nDetailed results saved under: {self.run_dir}")
        return {
            "metrics_dir": metrics_dir,
            "manifest_path": manifest_path,
            "summary": summary.to_dict(orient="records"),
        }


def main():
    Experiment5()()


if __name__ == "__main__":
    main()
