"""
Experiment 2: Multi-Seed & Multi-Beta Benchmark for Augmented VAE + Gradient Boosting.
Evaluates the impact of WGAN-GP synthetic patient augmentation across:
  1. Multiple random seeds (statistical stability)
  2. Multiple beta-regularization values (manifold regularization trade-off)
Evaluated on the held-out real test cohort (142 real patients).
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from typing import Dict, List, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.base_experiment import BaseExperiment
from model_src import (
    TabularVAE,
    VAETrainer,
    VAEClassifierPipeline,
    build_gradient_boosting,
)
from evaluation import compute_metrics, run_vae_evaluation


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Experiment2(BaseExperiment):
    """
    Evaluates:
      - Baseline VAE + GB (Real Only) vs. Augmented VAE + GB (Real + Syn)
      across multiple beta values: [0.0005, 0.005, 0.05, 0.5]
      and multiple random seeds.
      Also includes Standalone Gradient Boosting (Real vs. Real + Syn) as the reference anchor.
    """

    experiment_name = "experiment2"

    def __init__(
        self,
        config_path: str = None,
        config: dict = None,
        output_dir: str = None,
        timestamped: bool = True,
    ):
        super().__init__(
            config_path=config_path,
            config=config,
            output_dir=output_dir,
            timestamped=timestamped,
        )

    def run(self):
        dataset_cfg = self.config.get("dataset", {})
        vae_cfg = self.config.get("vae", {})
        gb_cfg = self.config.get("gradient_boosting", {})
        aug_cfg = self.config.get("augmentation", {})

        seeds = self.config.get("seeds", [42, 123, 456, 789, 2024])
        beta_values = self.config.get("beta_values", [0.0005, 0.005, 0.05, 0.5])
        synth_ratio = float(aug_cfg.get("synthetic_ratio", 1.0))

        # 1. Load Data
        print("[1/5] Loading datasets...")
        train_real_path = self._resolve_path(
            dataset_cfg.get("train_real_path", "dataset/processed_dataset/processed_train.csv")
        )
        train_synth_path = self._resolve_path(
            dataset_cfg.get("train_synthetic_path", "dataset/synthetic_dataset/synthetic_train.csv")
        )
        test_real_path = self._resolve_path(
            dataset_cfg.get("test_real_path", "dataset/processed_dataset/processed_test.csv")
        )
        label_col = dataset_cfg.get("label_column", "label")

        df_train_real = pd.read_csv(train_real_path)
        df_train_synth = pd.read_csv(train_synth_path)
        df_test_real = pd.read_csv(test_real_path)

        feature_cols = [c for c in df_train_real.columns if c != label_col]
        print(f"  • Real Training Set:    {len(df_train_real)} patients ({len(feature_cols)} features)")
        print(f"  • Synthetic Pool:       {len(df_train_synth)} patients")
        print(f"  • Real Holdout Test:    {len(df_test_real)} patients")
        print(f"  • Beta Values Grid:     {beta_values}")
        print(f"  • Seeds:                {seeds}")

        X_real_train = df_train_real[feature_cols].values
        y_real_train = df_train_real[label_col].values

        X_test = df_test_real[feature_cols].values
        y_test = df_test_real[label_col].values

        # Prepare subdirectories
        metrics_dir = self.get_results_subdir("metrics")
        plots_dir = self.get_results_subdir("plots")
        vae_diag_dir = self.get_results_subdir("vae_diagnostics")

        per_run_records: List[Dict[str, Any]] = []

        total_runs = len(seeds) * len(beta_values)
        run_count = 0

        print("\n[2/5] Launching Multi-Seed & Multi-Beta Grid Benchmark...")
        print("=" * 70)

        for seed in seeds:
            seed_everything(seed)

            # Subsample synthetic data if ratio < 1.0
            if synth_ratio < 1.0:
                n_sample = int(len(df_train_synth) * synth_ratio)
                df_synth_subset = df_train_synth.sample(n=n_sample, random_state=seed)
            else:
                df_synth_subset = df_train_synth

            X_synth = df_synth_subset[feature_cols].values
            y_synth = df_synth_subset[label_col].values

            # Combined Augmented Training Set
            X_aug_train = np.vstack([X_real_train, X_synth])
            y_aug_train = np.hstack([y_real_train, y_synth])

            # Standalone GB Baselines (only depends on seed, not beta)
            gb_real = build_gradient_boosting(
                n_estimators=gb_cfg.get("n_estimators", 200),
                learning_rate=gb_cfg.get("learning_rate", 0.1),
                max_depth=gb_cfg.get("max_depth", 4),
                min_samples_split=gb_cfg.get("min_samples_split", 4),
                min_samples_leaf=gb_cfg.get("min_samples_leaf", 2),
                max_features=gb_cfg.get("max_features", "sqrt"),
                random_state=seed,
            )
            gb_real.fit(X_real_train, y_real_train)
            m_gb_real = compute_metrics(y_test, gb_real.predict(X_test), gb_real.predict_proba(X_test)[:, 1])

            gb_aug = build_gradient_boosting(
                n_estimators=gb_cfg.get("n_estimators", 200),
                learning_rate=gb_cfg.get("learning_rate", 0.1),
                max_depth=gb_cfg.get("max_depth", 4),
                min_samples_split=gb_cfg.get("min_samples_split", 4),
                min_samples_leaf=gb_cfg.get("min_samples_leaf", 2),
                max_features=gb_cfg.get("max_features", "sqrt"),
                random_state=seed,
            )
            gb_aug.fit(X_aug_train, y_aug_train)
            m_gb_aug = compute_metrics(y_test, gb_aug.predict(X_test), gb_aug.predict_proba(X_test)[:, 1])

            # Record Standalone GB baselines
            per_run_records.append({
                "seed": seed,
                "beta": "N/A",
                "condition": "Baseline GB (Real)",
                "data_type": "Real",
                "model_family": "Standalone GB",
                **m_gb_real,
            })
            per_run_records.append({
                "seed": seed,
                "beta": "N/A",
                "condition": "Augmented GB (Real + Syn)",
                "data_type": "Real + Syn",
                "model_family": "Standalone GB",
                **m_gb_aug,
            })

            # Iterate through Beta Grid
            for beta in beta_values:
                run_count += 1
                print(f"  ▶️ [Run {run_count:2d}/{total_runs}] Seed={seed:<4d} | Beta={beta:<7g}...", end="", flush=True)

                # --- 1. Real VAE + GB ---
                vae_real = TabularVAE(
                    in_features=X_real_train.shape[1],
                    latent_dim=vae_cfg.get("latent_dim", 8),
                    hidden_dim1=vae_cfg.get("hidden_dim1", 128),
                    hidden_dim2=vae_cfg.get("hidden_dim2", 64),
                    dropout_rate=vae_cfg.get("dropout_rate", 0.1),
                )
                trainer_real = VAETrainer(
                    model=vae_real,
                    lr=vae_cfg.get("learning_rate", 1e-3),
                    weight_decay=vae_cfg.get("weight_decay", 1e-5),
                    beta=beta,
                )
                trainer_real.fit(
                    X_real_train,
                    epochs=vae_cfg.get("epochs", 100),
                    batch_size=vae_cfg.get("batch_size", 64),
                    verbose=False,
                )

                vae_real.eval()
                with torch.no_grad():
                    Z_real = vae_real.get_latent(
                        torch.tensor(X_real_train, dtype=torch.float32), deterministic=True
                    ).cpu().numpy()

                gb_core_real = build_gradient_boosting(
                    n_estimators=gb_cfg.get("n_estimators", 200),
                    learning_rate=gb_cfg.get("learning_rate", 0.1),
                    max_depth=gb_cfg.get("max_depth", 4),
                    min_samples_split=gb_cfg.get("min_samples_split", 4),
                    min_samples_leaf=gb_cfg.get("min_samples_leaf", 2),
                    max_features=gb_cfg.get("max_features", "sqrt"),
                    random_state=seed,
                )
                gb_core_real.fit(Z_real, y_real_train)
                pipe_real = VAEClassifierPipeline(vae_real, gb_core_real)
                m_vae_real = compute_metrics(y_test, pipe_real.predict(X_test), pipe_real.predict_proba(X_test)[:, 1])

                # --- 2. Augmented VAE + GB ---
                vae_aug = TabularVAE(
                    in_features=X_aug_train.shape[1],
                    latent_dim=vae_cfg.get("latent_dim", 8),
                    hidden_dim1=vae_cfg.get("hidden_dim1", 128),
                    hidden_dim2=vae_cfg.get("hidden_dim2", 64),
                    dropout_rate=vae_cfg.get("dropout_rate", 0.1),
                )
                trainer_aug = VAETrainer(
                    model=vae_aug,
                    lr=vae_cfg.get("learning_rate", 1e-3),
                    weight_decay=vae_cfg.get("weight_decay", 1e-5),
                    beta=beta,
                )
                trainer_aug.fit(
                    X_aug_train,
                    epochs=vae_cfg.get("epochs", 100),
                    batch_size=vae_cfg.get("batch_size", 64),
                    verbose=False,
                )

                vae_aug.eval()
                with torch.no_grad():
                    Z_aug = vae_aug.get_latent(
                        torch.tensor(X_aug_train, dtype=torch.float32), deterministic=True
                    ).cpu().numpy()

                gb_core_aug = build_gradient_boosting(
                    n_estimators=gb_cfg.get("n_estimators", 200),
                    learning_rate=gb_cfg.get("learning_rate", 0.1),
                    max_depth=gb_cfg.get("max_depth", 4),
                    min_samples_split=gb_cfg.get("min_samples_split", 4),
                    min_samples_leaf=gb_cfg.get("min_samples_leaf", 2),
                    max_features=gb_cfg.get("max_features", "sqrt"),
                    random_state=seed,
                )
                gb_core_aug.fit(Z_aug, y_aug_train)
                pipe_aug = VAEClassifierPipeline(vae_aug, gb_core_aug)
                m_vae_aug = compute_metrics(y_test, pipe_aug.predict(X_test), pipe_aug.predict_proba(X_test)[:, 1])

                per_run_records.append({
                    "seed": seed,
                    "beta": beta,
                    "condition": f"VAE + GB (Real, beta={beta})",
                    "data_type": "Real",
                    "model_family": "VAE + GB",
                    **m_vae_real,
                })
                per_run_records.append({
                    "seed": seed,
                    "beta": beta,
                    "condition": f"Augmented VAE + GB (Real+Syn, beta={beta})",
                    "data_type": "Real + Syn",
                    "model_family": "VAE + GB",
                    **m_vae_aug,
                })

                print(f" AUC: Real={m_vae_real['roc_auc']:.4f} | Aug={m_vae_aug['roc_auc']:.4f}")

                # Save VAE diagnostic plots for default beta on seed 42
                if seed == 42 and beta == 0.005:
                    run_vae_evaluation(
                        vae_model=vae_aug,
                        X_test=X_test,
                        y_test=y_test,
                        feature_names=feature_cols,
                        history=trainer_aug.history,
                        results_dir=os.path.join(vae_diag_dir, "beta_0.005_augmented"),
                    )

        # 3. Aggregate Results Across Seeds & Beta Values
        print("\n[3/5] Aggregating results across seeds and beta parameters...")
        df_all = pd.DataFrame(per_run_records)
        df_all.to_csv(os.path.join(metrics_dir, "detailed_per_run_metrics.csv"), index=False)

        metric_names = ["roc_auc", "pr_auc", "accuracy", "f1_score", "recall_sensitivity", "specificity", "brier_score"]

        # Summary grouped by Condition & Beta
        summary_rows = []
        for cond, group in df_all.groupby("condition", sort=False):
            row = {
                "Condition": cond,
                "Model Family": group["model_family"].iloc[0],
                "Data": group["data_type"].iloc[0],
                "Beta": group["beta"].iloc[0],
            }
            for m in metric_names:
                row[f"{m}_mean"] = float(group[m].mean())
                row[f"{m}_std"] = float(group[m].std())
            summary_rows.append(row)

        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(os.path.join(metrics_dir, "beta_seed_summary.csv"), index=False)

        # Create display markdown table
        display_list = []
        for r in summary_rows:
            display_list.append({
                "Condition": r["Condition"],
                "Data": r["Data"],
                "Beta": str(r["Beta"]),
                "ROC-AUC": f"{r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.4f}",
                "PR-AUC": f"{r['pr_auc_mean']:.4f} ± {r['pr_auc_std']:.4f}",
                "Accuracy": f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}",
                "F1-Score": f"{r['f1_score_mean']:.4f} ± {r['f1_score_std']:.4f}",
            })
        df_display = pd.DataFrame(display_list)

        md_path = os.path.join(metrics_dir, "beta_seed_summary.md")
        with open(md_path, "w") as f:
            f.write("# Experiment 2: Augmented VAE + Gradient Boosting (Multi-Seed & Multi-Beta)\n\n")
            f.write(f"Evaluated across {len(seeds)} random seeds on held-out test cohort (142 real patients).\n\n")
            f.write(df_display.to_markdown(index=False) + "\n")

        print("\n" + "=" * 70)
        print("📊 Multi-Beta & Multi-Seed Benchmark Summary Table (Mean ± Std)")
        print("=" * 70)
        print(df_display.to_markdown(index=False))
        print("=" * 70 + "\n")

        # 4. Visualizations
        print("[4/5] Generating beta sensitivity & multi-seed plots...")
        self._plot_beta_sensitivity(df_all, plots_dir)
        self._plot_boxplots_by_condition(df_all, plots_dir)

        print("[5/5] Experiment 2 completed successfully!")
        return {
            "detailed_metrics_csv": os.path.join(metrics_dir, "detailed_per_run_metrics.csv"),
            "summary_csv": os.path.join(metrics_dir, "beta_seed_summary.csv"),
            "summary_md": md_path,
        }

    def _plot_beta_sensitivity(self, df_all: pd.DataFrame, output_dir: str) -> None:
        """Plots ROC-AUC vs. Beta parameter for Real vs. Augmented VAE+GB."""
        df_vae = df_all[df_all["model_family"] == "VAE + GB"].copy()
        df_vae["beta_numeric"] = df_vae["beta"].astype(float)

        plt.figure(figsize=(9, 6))

        # Real VAE+GB line
        sns.lineplot(
            data=df_vae,
            x="beta_numeric",
            y="roc_auc",
            hue="data_type",
            style="data_type",
            markers=True,
            dashes=False,
            palette={"Real": "#2b5c8f", "Real + Syn": "#d95f02"},
            errorbar="sd",
            err_style="bars",
            lw=2.5,
            markersize=9,
        )

        # Add horizontal reference line for Baseline Standalone GB
        df_gb_real = df_all[df_all["condition"] == "Baseline GB (Real)"]
        mean_gb_real = df_gb_real["roc_auc"].mean()
        plt.axhline(mean_gb_real, color="#4daf4a", linestyle="--", lw=1.8, label=f"Baseline GB Real ({mean_gb_real:.3f})")

        plt.xscale("log")
        plt.xlabel("Beta Parameter (Log Scale)", fontsize=11, fontweight="bold")
        plt.ylabel("Test ROC-AUC (Mean ± Std)", fontsize=11, fontweight="bold")
        plt.title("Effect of Beta Regularization on Real vs. Augmented VAE + GB", fontsize=13, fontweight="bold")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(loc="lower left", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "beta_sensitivity_curve.png"), dpi=200)
        plt.close()

    def _plot_boxplots_by_condition(self, df_all: pd.DataFrame, output_dir: str) -> None:
        """Plots distribution of AUC across seeds for all conditions."""
        plt.figure(figsize=(13, 6))
        palette = sns.color_palette("Set2", n_colors=df_all["condition"].nunique())

        ax = sns.boxplot(
            data=df_all,
            x="condition",
            y="roc_auc",
            hue="condition",
            palette=palette,
            legend=False,
            width=0.45,
        )
        sns.stripplot(
            data=df_all,
            x="condition",
            y="roc_auc",
            color="black",
            size=5,
            jitter=0.15,
        )
        plt.xticks(rotation=35, ha="right", fontsize=9)
        plt.ylabel("ROC-AUC", fontsize=11, fontweight="bold")
        plt.title("ROC-AUC Distribution Across Random Seeds by Condition & Beta", fontsize=13, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "condition_seed_boxplots.png"), dpi=200)
        plt.close()


def main():
    exp = Experiment2()
    exp()


if __name__ == "__main__":
    main()
