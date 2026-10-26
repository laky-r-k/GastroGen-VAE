"""
Experiment 3: Semi-Supervised VAE Multi-Seed Benchmark.
Compares:
  1. SS-VAE End-to-End Neural Classifier (Direct classification head)
  2. SS-VAE + Gradient Boosting (Cancer-guided latent space Z fed to GB)
  3. Standard Unsupervised VAE + GB (Agnostic latent space Z fed to GB)
  4. Baseline Standalone Gradient Boosting (Raw clinical features)
Trained on Augmented Data (Real 567 + Synthetic 566 = 1,133 patients)
Evaluated on Held-Out Real Test Cohort (142 real patients) across multiple random seeds.
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
    SemiSupervisedTabularVAE,
    SemiSupervisedVAETrainer,
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


class Experiment3(BaseExperiment):
    """
    Evaluates Semi-Supervised VAE vs. Standard VAE vs. Standalone GB.
    """

    experiment_name = "experiment3"

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
        ss_vae_cfg = self.config.get("ss_vae", {})
        unsup_vae_cfg = self.config.get("unsupervised_vae", {})
        gb_cfg = self.config.get("gradient_boosting", {})
        aug_cfg = self.config.get("augmentation", {})

        seeds = self.config.get("seeds", [42, 123, 456, 789, 2024])
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
        print(f"  • Seeds:                {seeds}")

        X_real_train = df_train_real[feature_cols].values
        y_real_train = df_train_real[label_col].values

        X_test = df_test_real[feature_cols].values
        y_test = df_test_real[label_col].values

        metrics_dir = self.get_results_subdir("metrics")
        plots_dir = self.get_results_subdir("plots")
        diag_dir = self.get_results_subdir("diagnostics")

        arms = [
            "Baseline GB (Raw Features)",
            "Standard VAE + GB (Unsupervised Z)",
            "SS-VAE + GB (Supervised Z)",
            "SS-VAE End-to-End Neural Net",
        ]

        per_seed_records: List[Dict[str, Any]] = []
        roc_curves_data: Dict[str, List[Dict[str, Any]]] = {arm: [] for arm in arms}

        print("\n[2/5] Running Multi-Seed Benchmark across 4 Architectural Arms...")
        print("=" * 70)

        for seed_idx, seed in enumerate(seeds, 1):
            seed_everything(seed)
            print(f"\n▶️ Running Seed {seed_idx}/{len(seeds)} (seed={seed})...")

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

            # -------------------------------------------------------------
            # Arm 1: Baseline Standalone Gradient Boosting (Raw Features)
            # -------------------------------------------------------------
            gb_standalone = build_gradient_boosting(
                n_estimators=gb_cfg.get("n_estimators", 200),
                learning_rate=gb_cfg.get("learning_rate", 0.1),
                max_depth=gb_cfg.get("max_depth", 4),
                min_samples_split=gb_cfg.get("min_samples_split", 4),
                min_samples_leaf=gb_cfg.get("min_samples_leaf", 2),
                max_features=gb_cfg.get("max_features", "sqrt"),
                random_state=seed,
            )
            gb_standalone.fit(X_aug_train, y_aug_train)
            pred_gb = gb_standalone.predict(X_test)
            prob_gb = gb_standalone.predict_proba(X_test)[:, 1]
            m_gb = compute_metrics(y_test, pred_gb, prob_gb)

            # -------------------------------------------------------------
            # Arm 2: Standard Unsupervised VAE + Gradient Boosting
            # -------------------------------------------------------------
            unsup_vae = TabularVAE(
                in_features=X_aug_train.shape[1],
                latent_dim=unsup_vae_cfg.get("latent_dim", 8),
                hidden_dim1=unsup_vae_cfg.get("hidden_dim1", 128),
                hidden_dim2=unsup_vae_cfg.get("hidden_dim2", 64),
                dropout_rate=unsup_vae_cfg.get("dropout_rate", 0.1),
            )
            unsup_trainer = VAETrainer(
                model=unsup_vae,
                lr=unsup_vae_cfg.get("learning_rate", 1e-3),
                weight_decay=unsup_vae_cfg.get("weight_decay", 1e-5),
                beta=unsup_vae_cfg.get("beta", 0.005),
            )
            unsup_trainer.fit(
                X_aug_train,
                epochs=unsup_vae_cfg.get("epochs", 100),
                batch_size=unsup_vae_cfg.get("batch_size", 64),
                verbose=False,
            )

            unsup_vae.eval()
            with torch.no_grad():
                Z_unsup_train = unsup_vae.get_latent(
                    torch.tensor(X_aug_train, dtype=torch.float32), deterministic=True
                ).cpu().numpy()

            gb_unsup = build_gradient_boosting(
                n_estimators=gb_cfg.get("n_estimators", 200),
                learning_rate=gb_cfg.get("learning_rate", 0.1),
                max_depth=gb_cfg.get("max_depth", 4),
                min_samples_split=gb_cfg.get("min_samples_split", 4),
                min_samples_leaf=gb_cfg.get("min_samples_leaf", 2),
                max_features=gb_cfg.get("max_features", "sqrt"),
                random_state=seed,
            )
            gb_unsup.fit(Z_unsup_train, y_aug_train)
            pipe_unsup = VAEClassifierPipeline(unsup_vae, gb_unsup)
            pred_unsup = pipe_unsup.predict(X_test)
            prob_unsup = pipe_unsup.predict_proba(X_test)[:, 1]
            m_unsup = compute_metrics(y_test, pred_unsup, prob_unsup)

            # -------------------------------------------------------------
            # Arm 3 & 4: Semi-Supervised Tabular VAE (SS-VAE)
            # -------------------------------------------------------------
            ss_vae = SemiSupervisedTabularVAE(
                in_features=X_aug_train.shape[1],
                latent_dim=ss_vae_cfg.get("latent_dim", 8),
                hidden_dim1=ss_vae_cfg.get("hidden_dim1", 128),
                hidden_dim2=ss_vae_cfg.get("hidden_dim2", 64),
                classifier_hidden=ss_vae_cfg.get("classifier_hidden", 32),
                dropout_rate=ss_vae_cfg.get("dropout_rate", 0.1),
            )
            ss_trainer = SemiSupervisedVAETrainer(
                model=ss_vae,
                lr=ss_vae_cfg.get("learning_rate", 1e-3),
                weight_decay=ss_vae_cfg.get("weight_decay", 1e-5),
                beta=ss_vae_cfg.get("beta", 0.005),
                alpha=ss_vae_cfg.get("alpha", 10.0),
            )
            ss_trainer.fit(
                X_train=X_aug_train,
                y_train=y_aug_train,
                epochs=ss_vae_cfg.get("epochs", 100),
                batch_size=ss_vae_cfg.get("batch_size", 64),
                verbose=False,
            )

            # Arm 4: SS-VAE End-to-End Neural Classifier
            ss_vae.eval()
            with torch.no_grad():
                prob_ss_nn = ss_vae.predict_proba(torch.tensor(X_test, dtype=torch.float32)).cpu().numpy()
            pred_ss_nn = (prob_ss_nn >= 0.5).astype(int)
            m_ss_nn = compute_metrics(y_test, pred_ss_nn, prob_ss_nn)

            # Arm 3: SS-VAE + Gradient Boosting (Cancer-Guided Latent Space)
            with torch.no_grad():
                Z_ss_train = ss_vae.get_latent(
                    torch.tensor(X_aug_train, dtype=torch.float32), deterministic=True
                ).cpu().numpy()

            gb_ss = build_gradient_boosting(
                n_estimators=gb_cfg.get("n_estimators", 200),
                learning_rate=gb_cfg.get("learning_rate", 0.1),
                max_depth=gb_cfg.get("max_depth", 4),
                min_samples_split=gb_cfg.get("min_samples_split", 4),
                min_samples_leaf=gb_cfg.get("min_samples_leaf", 2),
                max_features=gb_cfg.get("max_features", "sqrt"),
                random_state=seed,
            )
            gb_ss.fit(Z_ss_train, y_aug_train)
            pipe_ss_gb = VAEClassifierPipeline(ss_vae, gb_ss)
            pred_ss_gb = pipe_ss_gb.predict(X_test)
            prob_ss_gb = pipe_ss_gb.predict_proba(X_test)[:, 1]
            m_ss_gb = compute_metrics(y_test, pred_ss_gb, prob_ss_gb)

            # Record metrics
            run_map = {
                "Baseline GB (Raw Features)": (m_gb, prob_gb),
                "Standard VAE + GB (Unsupervised Z)": (m_unsup, prob_unsup),
                "SS-VAE + GB (Supervised Z)": (m_ss_gb, prob_ss_gb),
                "SS-VAE End-to-End Neural Net": (m_ss_nn, prob_ss_nn),
            }

            for arm_name, (metric_dict, prob_arr) in run_map.items():
                per_seed_records.append({
                    "seed": seed,
                    "condition": arm_name,
                    **metric_dict,
                })
                roc_curves_data[arm_name].append({
                    "seed": seed,
                    "y_true": y_test,
                    "y_prob": prob_arr,
                })

            print(
                f"  [Seed {seed}] AUC -> Baseline GB: {m_gb['roc_auc']:.4f} | "
                f"Unsup VAE+GB: {m_unsup['roc_auc']:.4f} | "
                f"SS-VAE+GB: {m_ss_gb['roc_auc']:.4f} | "
                f"SS-VAE Neural Net: {m_ss_nn['roc_auc']:.4f}"
            )

            # On first seed, plot SS-VAE training curves
            if seed_idx == 1:
                self._plot_ss_vae_training_curves(ss_trainer.history, diag_dir)

        # 3. Aggregate Results Across Seeds
        print("\n[3/5] Aggregating results across random seeds (Mean ± Std)...")
        df_per_seed = pd.DataFrame(per_seed_records)
        df_per_seed.to_csv(os.path.join(metrics_dir, "detailed_per_seed_metrics.csv"), index=False)

        metric_names = ["roc_auc", "pr_auc", "accuracy", "f1_score", "recall_sensitivity", "specificity", "brier_score"]
        agg_rows = []

        for arm_name in arms:
            sub = df_per_seed[df_per_seed["condition"] == arm_name]
            row = {"Condition": arm_name}
            for m in metric_names:
                row[f"{m}_mean"] = float(sub[m].mean())
                row[f"{m}_std"] = float(sub[m].std())
            agg_rows.append(row)

        df_summary = pd.DataFrame(agg_rows)
        df_summary.to_csv(os.path.join(metrics_dir, "ss_vae_benchmark_summary.csv"), index=False)

        display_list = []
        for r in agg_rows:
            display_list.append({
                "Architectural Condition": r["Condition"],
                "ROC-AUC": f"{r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.4f}",
                "PR-AUC": f"{r['pr_auc_mean']:.4f} ± {r['pr_auc_std']:.4f}",
                "Accuracy": f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}",
                "F1-Score": f"{r['f1_score_mean']:.4f} ± {r['f1_score_std']:.4f}",
                "Sensitivity": f"{r['recall_sensitivity_mean']:.4f} ± {r['recall_sensitivity_std']:.4f}",
                "Specificity": f"{r['specificity_mean']:.4f} ± {r['specificity_std']:.4f}",
            })
        df_display = pd.DataFrame(display_list)

        md_path = os.path.join(metrics_dir, "ss_vae_benchmark_summary.md")
        with open(md_path, "w") as f:
            f.write("# Experiment 3: Semi-Supervised VAE vs. Gradient Boosting Benchmark\n\n")
            f.write(f"Trained on Augmented Dataset (1,133 patients). Evaluated on held-out test cohort (142 real patients) across {len(seeds)} random seeds.\n\n")
            f.write(df_display.to_markdown(index=False) + "\n")

        print("\n" + "=" * 75)
        print("📊 Benchmark Summary Table (Mean ± Std Across Seeds)")
        print("=" * 75)
        print(df_display.to_markdown(index=False))
        print("=" * 75 + "\n")

        # 4. Generate Visualizations
        print("[4/5] Generating multi-seed comparison plots...")
        self._plot_condition_boxplots(df_per_seed, plots_dir)
        self._plot_mean_roc_curves(roc_curves_data, plots_dir)

        print("[5/5] Experiment 3 completed successfully!")
        return {
            "detailed_metrics_csv": os.path.join(metrics_dir, "detailed_per_seed_metrics.csv"),
            "summary_csv": os.path.join(metrics_dir, "ss_vae_benchmark_summary.csv"),
            "summary_md": md_path,
        }

    def _plot_ss_vae_training_curves(self, history: Dict[str, List[float]], output_dir: str) -> None:
        """Plots SS-VAE multi-task training loss dynamics."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        epochs = history["epoch"]
        axes[0].plot(epochs, history["total_loss"], label="Total Joint Loss", color="black", lw=2)
        axes[0].plot(epochs, history["recon_loss"], label="Recon Loss (MSE)", color="royalblue", lw=1.5)
        axes[0].plot(epochs, history["class_loss"], label="Classification Loss (BCE)", color="crimson", lw=1.5)
        axes[0].plot(epochs, history["kl_loss"], label="KL Divergence", color="forestgreen", lw=1.5, linestyle="--")
        axes[0].set_title("SS-VAE Joint Multi-Task Loss Curves", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Epoch", fontsize=11)
        axes[0].set_ylabel("Loss", fontsize=11)
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, history["train_accuracy"], label="Training Accuracy", color="darkorange", lw=2)
        axes[1].set_title("SS-VAE Classifier Head Training Accuracy", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_ylabel("Accuracy", fontsize=11)
        axes[1].set_ylim(0.5, 1.0)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ss_vae_training_dynamics.png"), dpi=200)
        plt.close()

    def _plot_condition_boxplots(self, df_per_seed: pd.DataFrame, output_dir: str) -> None:
        """Boxplots comparing ROC-AUC and Accuracy across seeds."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        palette = ["#4daf4a", "#2b5c8f", "#d95f02", "#e41a1c"]

        sns.boxplot(
            data=df_per_seed,
            x="condition",
            y="roc_auc",
            hue="condition",
            palette=palette,
            legend=False,
            ax=axes[0],
            width=0.4,
        )
        sns.stripplot(
            data=df_per_seed,
            x="condition",
            y="roc_auc",
            color="black",
            size=6,
            jitter=0.15,
            ax=axes[0],
        )
        axes[0].set_title("ROC-AUC Comparison Across Random Seeds", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("ROC-AUC", fontsize=11)
        axes[0].set_xlabel("")
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=25, ha="right", fontsize=9)
        axes[0].grid(True, alpha=0.3)

        sns.boxplot(
            data=df_per_seed,
            x="condition",
            y="accuracy",
            hue="condition",
            palette=palette,
            legend=False,
            ax=axes[1],
            width=0.4,
        )
        sns.stripplot(
            data=df_per_seed,
            x="condition",
            y="accuracy",
            color="black",
            size=6,
            jitter=0.15,
            ax=axes[1],
        )
        axes[1].set_title("Accuracy Comparison Across Random Seeds", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Accuracy", fontsize=11)
        axes[1].set_xlabel("")
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=25, ha="right", fontsize=9)
        axes[1].grid(True, alpha=0.3)

        plt.suptitle("Semi-Supervised VAE vs. Baselines on Held-Out Test Set", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ss_vae_benchmark_boxplots.png"), dpi=200)
        plt.close()

    def _plot_mean_roc_curves(self, roc_curves_data: Dict[str, List[Dict[str, Any]]], output_dir: str) -> None:
        """Plots mean ROC curves with ±1 std deviation bands."""
        from sklearn.metrics import roc_curve, auc

        plt.figure(figsize=(9, 7))
        colors = {
            "Baseline GB (Raw Features)": "#4daf4a",
            "Standard VAE + GB (Unsupervised Z)": "#2b5c8f",
            "SS-VAE + GB (Supervised Z)": "#d95f02",
            "SS-VAE End-to-End Neural Net": "#e41a1c",
        }

        mean_fpr = np.linspace(0, 1, 100)

        for arm_name, runs in roc_curves_data.items():
            tprs = []
            aucs = []
            for run in runs:
                fpr, tpr, _ = roc_curve(run["y_true"], run["y_prob"])
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(auc(fpr, tpr))

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)

            color = colors.get(arm_name, "gray")
            plt.plot(
                mean_fpr,
                mean_tpr,
                color=color,
                lw=2.2,
                label=f"{arm_name} (AUC = {mean_auc:.4f} ± {std_auc:.4f})",
            )

            std_tpr = np.std(tprs, axis=0)
            tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
            tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
            plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color=color, alpha=0.12)

        plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Chance")
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.05])
        plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
        plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
        plt.title("Mean ROC Curves: SS-VAE vs. Tree Baselines (Multi-Seed)", fontsize=13, fontweight="bold")
        plt.legend(loc="lower right", fontsize=9.5)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ss_vae_mean_roc_curves.png"), dpi=200)
        plt.close()


def main():
    exp = Experiment3()
    exp()


if __name__ == "__main__":
    main()
