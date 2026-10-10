"""
Experiment WGAN-GP: Synthetic Tabular Biomarker Augmentation & Honest Downstream Benchmark.
Implements Objective 3 under rigorous repeated stratified cross-validation.
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE
import torch

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.base_experiment import BaseExperiment
from model_src.GAN import Generator, Critic, WGANGPTrainer
from model_src.tree.gradient_boosting import build_gradient_boosting
from model_src.VAE import TabularVAE, VAETrainer
from data_handling.fold_preprocessor import FoldPreprocessor, BASE_GEN_FEATURES
from evaluation.synthetic_fidelity import run_full_fidelity_suite
from preprocessing import PreprocessingPipeline


# Canonical 24 downstream clinical features evaluated in project
DOWNSTREAM_FEATURES = [
    "gender", "age", "log_neu_lym", "nlr_computed", "hb", "hct", "rdw",
    "hb_rdw_ratio", "plt_lym_ratio", "alb", "log_alt", "log_tb",
    "alb_tb_ratio", "cr", "log_tg", "chol", "hdl", "atherogenic_index",
    "log_lpa", "log_cea", "log_ca199", "log_ca125", "log_ca724",
    "tumor_marker_composite",
]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ExperimentWGAN(BaseExperiment):
    """
    Evaluates conditional WGAN-GP tabular clinical data augmentation.
    Compares Condition A (Real Only) vs Condition B (Real + WGAN-GP at 0.5x, 1x, 2x),
    Condition C (SMOTE), and Condition D (VAE Decoder) under 5x5 CV with paired bootstrap testing.
    """

    experiment_name = "experiment_wgan"

    def __init__(
        self,
        config_path: str = None,
        config: dict = None,
        output_dir: str = None,
        timestamped: bool = True,
        smoke_test: bool = False,
    ):
        super().__init__(
            config_path=config_path,
            config=config,
            output_dir=output_dir,
            timestamped=timestamped,
        )
        self.smoke_test = smoke_test

    def run(self):
        seed = int(self.config.get("seed", 42))
        seed_everything(seed)
        print(f"🔒 Reproducibility Seed set to: {seed}")

        # Check smoke test overrides
        smoke_cfg = self.config.get("smoke_test", {})
        is_smoke = self.smoke_test or smoke_cfg.get("enabled", False)

        gan_cfg = self.config.get("gan", {})
        eval_cfg = self.config.get("evaluation", {})
        prep_cfg = self.config.get("preprocessing", {})

        n_splits = smoke_cfg.get("cv_splits", 2) if is_smoke else eval_cfg.get("cv_splits", 5)
        n_repeats = smoke_cfg.get("cv_repeats", 1) if is_smoke else eval_cfg.get("cv_repeats", 5)
        gen_steps = smoke_cfg.get("generator_steps", 200) if is_smoke else gan_cfg.get("generator_steps", 5000)
        batch_size = gan_cfg.get("batch_size", 64)
        synth_ratios = eval_cfg.get("synthetic_ratios", [0.5, 1.0, 2.0])

        print(f"⚙️ Running mode: {'SMOKE TEST' if is_smoke else 'FULL EXPERIMENT'}")
        print(f"📊 CV Protocol: {n_splits} folds x {n_repeats} repeats ({n_splits * n_repeats} total folds)")
        print(f"⚡ Generator Steps: {gen_steps} per fold")

        # 1. Load Raw Training Data (569 patients)
        dataset_cfg = self.config.get("dataset", {})
        train_raw_path = self._resolve_path(dataset_cfg.get("train_raw_path", "dataset/gastric_cancer_train.csv"))
        test_raw_path = self._resolve_path(dataset_cfg.get("test_raw_path", "dataset/gastric_cancer_test.csv"))

        df_train_raw = pd.read_csv(train_raw_path)
        print(f"📁 Loaded Raw Train Dataset: {df_train_raw.shape[0]} patients, {df_train_raw.shape[1]} columns")

        y_train_all = df_train_raw["label"].values
        patient_indices = np.arange(len(df_train_raw))

        # Setup cross-validation
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

        # Track out-of-fold probability predictions for paired bootstrap
        # Maps condition_name -> 2D array of shape (n_repeats, n_patients)
        conditions = ["A_real_only"]
        for r in synth_ratios:
            conditions.append(f"B_wgan_{r}x")
        for r in synth_ratios:
            conditions.append(f"C_smote_{r}x")
        conditions.append("D_vae_decoder_1x")

        oof_probs = {cond: np.full((n_repeats, len(df_train_raw)), np.nan) for cond in conditions}
        fold_metrics_list = []

        # We will preserve the first fold's generator and held-out slice for comprehensive fidelity diagnostics
        first_fold_saved_assets = None

        print("\n" + "-" * 70)
        print("🔄 Beginning Cross-Validation Loop")
        print("-" * 70)

        total_folds = n_splits * n_repeats
        fold_idx = 0
        fold_times = []

        for repeat_idx in range(n_repeats):
            for split_idx, (train_idx, val_idx) in enumerate(rskf.split(patient_indices, y_train_all)):
                if repeat_idx * n_splits + split_idx != fold_idx:
                    continue  # Alignment with rskf generator

                fold_start = time.time()
                fold_num = fold_idx + 1
                print(f"\n▶️ [Fold {fold_num:2d}/{total_folds}] (Repeat {repeat_idx+1}, Split {split_idx+1})")

                df_train_fold = df_train_raw.iloc[train_idx].copy()
                df_val_fold = df_train_raw.iloc[val_idx].copy()

                y_val_real = df_val_fold["label"].values

                # 1. Fit FoldPreprocessor strictly on training fold
                preprocessor = FoldPreprocessor(
                    scaler_type=prep_cfg.get("scaler_type", "standard"),
                    log_cols=prep_cfg.get("log_columns", None),
                )
                preprocessor.fit(df_train_fold)

                # Process Real Downstream Features for Training and Validation
                df_train_proc = preprocessor.process_real_downstream(df_train_fold)
                df_val_proc = preprocessor.process_real_downstream(df_val_fold)

                X_tr_real = df_train_proc[DOWNSTREAM_FEATURES].values
                y_tr_real = df_train_proc["label"].values
                X_val_real = df_val_proc[DOWNSTREAM_FEATURES].values

                # ----------------------------------------------------
                # Condition A: Real Only Baseline
                # ----------------------------------------------------
                clf_a = build_gradient_boosting(random_state=seed)
                clf_a.fit(X_tr_real, y_tr_real)
                prob_a = clf_a.predict_proba(X_val_real)[:, 1]
                oof_probs["A_real_only"][repeat_idx, val_idx] = prob_a

                fold_metrics_list.append(self._compute_metrics_record(
                    fold_num, repeat_idx, "A_real_only", y_val_real, prob_a
                ))

                # ----------------------------------------------------
                # Train 1 WGAN-GP on Training Fold
                # ----------------------------------------------------
                X_gan_train = preprocessor.transform_for_gan(df_train_fold)
                y_gan_train = df_train_fold["label"].values
                num_classes = 2
                output_dim = X_gan_train.shape[1]

                gen = Generator(
                    latent_dim=gan_cfg.get("latent_dim", 16),
                    num_classes=num_classes,
                    output_dim=output_dim,
                    hidden_dim=gan_cfg.get("hidden_dim", 128),
                    use_layer_norm=gan_cfg.get("use_layer_norm", True),
                    use_batch_norm=gan_cfg.get("use_batch_norm", False),
                )
                critic = Critic(
                    input_dim=output_dim,
                    num_classes=num_classes,
                    hidden_dim=gan_cfg.get("hidden_dim", 128),
                    use_layer_norm=gan_cfg.get("use_layer_norm", True),
                )
                wgan_trainer = WGANGPTrainer(
                    generator=gen,
                    critic=critic,
                    lr_g=gan_cfg.get("learning_rate_g", 1e-4),
                    lr_d=gan_cfg.get("learning_rate_d", 1e-4),
                    betas=tuple(gan_cfg.get("betas", (0.5, 0.9))),
                    lambda_gp=gan_cfg.get("lambda_gp", 10.0),
                    n_critic=gan_cfg.get("n_critic", 5),
                )

                print(f"   Training WGAN-GP ({gen_steps} steps)...")
                wgan_trainer.fit(
                    X_train=X_gan_train,
                    y_train=y_gan_train,
                    generator_steps=gen_steps,
                    batch_size=batch_size,
                    checkpoint_interval=gan_cfg.get("checkpoint_interval", 500),
                    verbose=False,
                )

                # ----------------------------------------------------
                # Condition B: Real + WGAN-GP (0.5x, 1x, 2x sizes)
                # ----------------------------------------------------
                n_train_fold = len(df_train_fold)
                max_ratio = max(synth_ratios)
                max_samples_each = int(np.ceil((max_ratio * n_train_fold) / 2))

                # Generate once from this fold's generator
                all_syn_scaled_dict = wgan_trainer.generate({0: max_samples_each, 1: max_samples_each})

                for r in synth_ratios:
                    cond_name = f"B_wgan_{r}x"
                    needed_per_class = int(np.round((r * n_train_fold) / 2))

                    syn_scaled_0 = all_syn_scaled_dict[0][:needed_per_class]
                    syn_scaled_1 = all_syn_scaled_dict[1][:needed_per_class]

                    syn_scaled_combined = np.vstack([syn_scaled_0, syn_scaled_1])
                    syn_labels_combined = np.hstack([np.zeros(len(syn_scaled_0)), np.ones(len(syn_scaled_1))])

                    # Inverse transform & recompute features
                    df_syn_eng = preprocessor.inverse_transform_synthetic(syn_scaled_combined, syn_labels_combined)
                    X_syn_proc = df_syn_eng[DOWNSTREAM_FEATURES].values
                    y_syn_proc = df_syn_eng["label"].values

                    # Augmented training pool
                    X_aug = np.vstack([X_tr_real, X_syn_proc])
                    y_aug = np.hstack([y_tr_real, y_syn_proc])

                    clf_b = build_gradient_boosting(random_state=seed)
                    clf_b.fit(X_aug, y_aug)
                    prob_b = clf_b.predict_proba(X_val_real)[:, 1]
                    oof_probs[cond_name][repeat_idx, val_idx] = prob_b

                    fold_metrics_list.append(self._compute_metrics_record(
                        fold_num, repeat_idx, cond_name, y_val_real, prob_b
                    ))

                    # Save first fold 1x synthetic data for fidelity checking
                    if fold_idx == 0 and r == 1.0:
                        first_fold_saved_assets = {
                            "real_train_df": df_train_proc,
                            "held_out_val_df": df_val_proc,
                            "synthetic_df": df_syn_eng,
                            "trainer_history": wgan_trainer.history,
                        }

                # ----------------------------------------------------
                # Condition C: Real + SMOTE (Matching sizes)
                # ----------------------------------------------------
                for r in synth_ratios:
                    cond_name = f"C_smote_{r}x"
                    n_desired_total = n_train_fold + int(np.round(r * n_train_fold))
                    # Balance classes to desired total
                    n_pos_desired = n_desired_total // 2
                    n_neg_desired = n_desired_total - n_pos_desired

                    # Count real instances
                    n_pos_real = int(np.sum(y_tr_real == 1))
                    n_neg_real = int(np.sum(y_tr_real == 0))

                    sampling_strat = {
                        1: max(n_pos_real, n_pos_desired),
                        0: max(n_neg_real, n_neg_desired),
                    }
                    try:
                        smote = SMOTE(sampling_strategy=sampling_strat, random_state=seed, k_neighbors=3)
                        X_smote, y_smote = smote.fit_resample(X_tr_real, y_tr_real)
                    except Exception:
                        # Fallback if class sizes don't require SMOTE expansion
                        X_smote, y_smote = X_tr_real, y_tr_real

                    clf_c = build_gradient_boosting(random_state=seed)
                    clf_c.fit(X_smote, y_smote)
                    prob_c = clf_c.predict_proba(X_val_real)[:, 1]
                    oof_probs[cond_name][repeat_idx, val_idx] = prob_c

                    fold_metrics_list.append(self._compute_metrics_record(
                        fold_num, repeat_idx, cond_name, y_val_real, prob_c
                    ))

                # ----------------------------------------------------
                # Condition D: Real + VAE Decoder (GMM Empirical Sampling)
                # ----------------------------------------------------
                cond_name = "D_vae_decoder_1x"
                try:
                    vae = TabularVAE(in_features=X_tr_real.shape[1], latent_dim=8)
                    vae_trainer = VAETrainer(vae, lr=1e-3, beta=0.005)
                    vae_trainer.fit(X_tr_real, epochs=50 if is_smoke else 100, batch_size=64, verbose=False)

                    # Encode training points to get empirical class-conditional latent distributions
                    vae.eval()
                    with torch.no_grad():
                        z_train = vae.get_latent(torch.tensor(X_tr_real, dtype=torch.float32), deterministic=True).cpu().numpy()

                    z_pos = z_train[y_tr_real == 1]
                    z_neg = z_train[y_tr_real == 0]

                    mu_pos, cov_pos = np.mean(z_pos, axis=0), np.cov(z_pos, rowvar=False) + 1e-4 * np.eye(8)
                    mu_neg, cov_neg = np.mean(z_neg, axis=0), np.cov(z_neg, rowvar=False) + 1e-4 * np.eye(8)

                    n_gen_half = n_train_fold // 2
                    z_sample_pos = np.random.multivariate_normal(mu_pos, cov_pos, n_gen_half)
                    z_sample_neg = np.random.multivariate_normal(mu_neg, cov_neg, n_gen_half)

                    with torch.no_grad():
                        x_recon_pos = vae.decode(torch.tensor(z_sample_pos, dtype=torch.float32)).cpu().numpy()
                        x_recon_neg = vae.decode(torch.tensor(z_sample_neg, dtype=torch.float32)).cpu().numpy()

                    X_vae_syn = np.vstack([x_recon_pos, x_recon_neg])
                    y_vae_syn = np.hstack([np.ones(len(x_recon_pos)), np.zeros(len(x_recon_neg))])

                    X_aug_vae = np.vstack([X_tr_real, X_vae_syn])
                    y_aug_vae = np.hstack([y_tr_real, y_vae_syn])

                    clf_d = build_gradient_boosting(random_state=seed)
                    clf_d.fit(X_aug_vae, y_aug_vae)
                    prob_d = clf_d.predict_proba(X_val_real)[:, 1]
                    oof_probs[cond_name][repeat_idx, val_idx] = prob_d

                    fold_metrics_list.append(self._compute_metrics_record(
                        fold_num, repeat_idx, cond_name, y_val_real, prob_d
                    ))
                except Exception as e:
                    print(f"   ⚠️ VAE Condition D skipped due to: {e}")

                fold_elapsed = time.time() - fold_start
                fold_times.append(fold_elapsed)
                print(f"   ⏱️ Fold {fold_num} finished in {fold_elapsed:.1f}s (AUC Real: {roc_auc_score(y_val_real, prob_a):.4f} | WGAN-1x: {roc_auc_score(y_val_real, prob_b):.4f})")

                fold_idx += 1
                if fold_idx >= total_folds:
                    break
            if fold_idx >= total_folds:
                break

        # Log average time per fold
        avg_fold_time = float(np.mean(fold_times))
        print(f"\n⏱️ Average Wall-Clock Time Per Fold: {avg_fold_time:.2f} seconds")

        # ----------------------------------------------------
        # Run Fidelity Diagnostics on Saved Assets (Fold 1)
        # ----------------------------------------------------
        if first_fold_saved_assets is not None:
            print("\n" + "=" * 70)
            print("🔬 Running Comprehensive Fidelity Diagnostics on Fold 1 Held-Out Slice...")
            print("=" * 70)
            fid_dir = self.get_results_subdir("fidelity_diagnostics")

            # Plot training loss history
            self._plot_wgan_loss(first_fold_saved_assets["trainer_history"], fid_dir)

            run_full_fidelity_suite(
                real_train_df=first_fold_saved_assets["real_train_df"],
                held_out_real_df=first_fold_saved_assets["held_out_val_df"],
                synthetic_df=first_fold_saved_assets["synthetic_df"],
                features=DOWNSTREAM_FEATURES,
                label_col="label",
                output_dir=fid_dir,
            )

        # ----------------------------------------------------
        # Statistical Analysis: Paired Bootstrap on Pooled Out-Of-Fold Predictions
        # ----------------------------------------------------
        print("\n" + "=" * 70)
        print("📈 Statistical Analysis: Paired Patient Bootstrap (95% CI vs Baseline A)")
        print("=" * 70)

        # Average each patient's predictions across repeats
        pooled_preds = {}
        for cond, arr in oof_probs.items():
            pooled_preds[cond] = np.nanmean(arr, axis=0)

        results_table = self._compute_paired_bootstrap_comparison(
            y_true=y_train_all,
            pooled_preds=pooled_preds,
            baseline_cond="A_real_only",
            bootstrap_rounds=eval_cfg.get("bootstrap_rounds", 1000),
            seed=seed,
        )

        # Save Summary CSV and Markdown Tables
        summary_dir = self.get_results_subdir("downstream_comparison")
        results_df = pd.DataFrame(results_table)
        results_df.to_csv(os.path.join(summary_dir, "cv_augmentation_benchmark.csv"), index=False)

        md_table = results_df.to_markdown(index=False)
        with open(os.path.join(summary_dir, "cv_augmentation_benchmark.md"), "w") as f:
            f.write("# Downstream Augmentation Cross-Validation Benchmark\n\n")
            f.write(f"Protocol: {n_splits}-fold x {n_repeats}-repeats. Paired patient bootstrap over 1,000 resamples.\n\n")
            f.write(md_table + "\n")

        print("\n" + md_table + "\n")

        # Render Benchmark Box/Bar Plot
        self._plot_benchmark_comparison(results_df, summary_dir)

        # ----------------------------------------------------
        # Persistent Full Training Set Synthesis Export
        # ----------------------------------------------------
        print("\n" + "=" * 70)
        print("💾 Persisting Full Training Set Synthetic Dataset...")
        print("=" * 70)
        self._export_persistent_synthetic_dataset(df_train_raw, prep_cfg, gan_cfg, gen_steps, batch_size, seed)

        return results_table

    def _compute_metrics_record(self, fold: int, repeat: int, cond: str, y_true: np.ndarray, prob: np.ndarray) -> Dict[str, Any]:
        pred = (prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        return {
            "fold": fold,
            "repeat": repeat,
            "condition": cond,
            "roc_auc": float(roc_auc_score(y_true, prob)),
            "accuracy": float(accuracy_score(y_true, pred)),
            "sensitivity": float(recall_score(y_true, pred, zero_division=0)),
            "specificity": float(tn / (tn + fp + 1e-5)),
            "f1_score": float(f1_score(y_true, pred, zero_division=0)),
        }

    def _compute_paired_bootstrap_comparison(
        self,
        y_true: np.ndarray,
        pooled_preds: Dict[str, np.ndarray],
        baseline_cond: str = "A_real_only",
        bootstrap_rounds: int = 1000,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """
        Calculates paired bootstrap 95% Confidence Intervals for AUC difference vs Baseline A.
        """
        np.random.seed(seed)
        n_patients = len(y_true)
        records = []

        prob_base = pooled_preds[baseline_cond]
        auc_base = roc_auc_score(y_true, prob_base)

        for cond, prob_cond in pooled_preds.items():
            auc_cond = roc_auc_score(y_true, prob_cond)
            pred_cond = (prob_cond >= 0.5).astype(int)
            acc_cond = accuracy_score(y_true, pred_cond)
            sens_cond = recall_score(y_true, pred_cond, zero_division=0)
            tn, fp, _, _ = confusion_matrix(y_true, pred_cond).ravel()
            spec_cond = tn / (tn + fp + 1e-5)
            f1_val = f1_score(y_true, pred_cond, zero_division=0)

            # Paired bootstrap for AUC delta
            auc_diffs = []
            for _ in range(bootstrap_rounds):
                boot_idx = np.random.choice(n_patients, size=n_patients, replace=True)
                y_boot = y_true[boot_idx]
                if len(np.unique(y_boot)) < 2:
                    continue
                b_auc = roc_auc_score(y_boot, prob_base[boot_idx])
                c_auc = roc_auc_score(y_boot, prob_cond[boot_idx])
                auc_diffs.append(c_auc - b_auc)

            ci_lower = float(np.percentile(auc_diffs, 2.5))
            ci_upper = float(np.percentile(auc_diffs, 97.5))
            mean_diff = float(np.mean(auc_diffs))

            is_stat_sig = (ci_lower > 0.0) or (ci_upper < 0.0)
            sig_flag = "Significant" if is_stat_sig else "Not Significant"

            records.append({
                "Condition": cond,
                "ROC-AUC": round(float(auc_cond), 4),
                "Accuracy": round(float(acc_cond), 4),
                "Sensitivity": round(float(sens_cond), 4),
                "Specificity": round(float(spec_cond), 4),
                "F1-Score": round(float(f1_val), 4),
                "Delta AUC vs Baseline": round(mean_diff, 4),
                "95% CI Lower": round(ci_lower, 4),
                "95% CI Upper": round(ci_upper, 4),
                "Significance": sig_flag,
            })

        return records

    def _plot_benchmark_comparison(self, df_results: pd.DataFrame, output_dir: str) -> None:
        plt.figure(figsize=(10, 6))
        palette = ["#2b5c8f" if "real" in c.lower() else "#d95f02" if "wgan" in c.lower() else "#7570b3" if "smote" in c.lower() else "#1b9e77" for c in df_results["Condition"]]

        ax = sns.barplot(
            data=df_results,
            x="Condition",
            y="ROC-AUC",
            hue="Condition",
            palette=palette,
            legend=False,
        )
        plt.ylim(0.5, 1.0)
        plt.title("Downstream Classification Benchmark: Paired Cross-Validation ROC-AUC", fontsize=13, fontweight="bold")
        plt.xticks(rotation=30, ha="right")

        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f"{height:.4f}", (p.get_x() + p.get_width() / 2.0, height + 0.01), ha="center", fontsize=9, fontweight="bold")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "augmentation_auc_comparison.png"), dpi=200)
        plt.close()

    def _plot_wgan_loss(self, history: Dict[str, List[float]], output_dir: str) -> None:
        if not history or "step" not in history:
            return
        plt.figure(figsize=(12, 5))
        plt.plot(history["step"], history["d_loss"], label="Critic (D) Loss", color="royalblue", alpha=0.8)
        plt.plot(history["step"], history["g_loss"], label="Generator (G) Loss", color="crimson", alpha=0.8)
        plt.plot(history["step"], history["wasserstein_estimate"], label="Wasserstein Distance Estimate", color="forestgreen", linestyle="--")
        plt.xlabel("Generator Steps", fontsize=11)
        plt.ylabel("Loss / Score", fontsize=11)
        plt.title("WGAN-GP Training Dynamics", fontsize=13, fontweight="bold")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "wgan_loss_curves.png"), dpi=200)
        plt.close()

    def _export_persistent_synthetic_dataset(
        self,
        df_train_raw: pd.DataFrame,
        prep_cfg: dict,
        gan_cfg: dict,
        gen_steps: int,
        batch_size: int,
        seed: int,
    ) -> None:
        """
        Trains final WGAN-GP on the complete 569-patient training set,
        synthesizes 569 balanced profiles, and exports to dataset/synthetic_dataset/.
        """
        export_dir = os.path.join(self.project_root, "dataset", "synthetic_dataset")
        os.makedirs(export_dir, exist_ok=True)

        prep = FoldPreprocessor(
            scaler_type=prep_cfg.get("scaler_type", "standard"),
            log_cols=prep_cfg.get("log_columns", None),
        )
        prep.fit(df_train_raw)

        X_scaled = prep.transform_for_gan(df_train_raw)
        y_train = df_train_raw["label"].values

        gen = Generator(
            latent_dim=gan_cfg.get("latent_dim", 16),
            num_classes=2,
            output_dim=X_scaled.shape[1],
            hidden_dim=gan_cfg.get("hidden_dim", 128),
            use_layer_norm=gan_cfg.get("use_layer_norm", True),
            use_batch_norm=gan_cfg.get("use_batch_norm", False),
        )
        critic = Critic(
            input_dim=X_scaled.shape[1],
            num_classes=2,
            hidden_dim=gan_cfg.get("hidden_dim", 128),
            use_layer_norm=gan_cfg.get("use_layer_norm", True),
        )
        trainer = WGANGPTrainer(
            generator=gen,
            critic=critic,
            lr_g=gan_cfg.get("learning_rate_g", 1e-4),
            lr_d=gan_cfg.get("learning_rate_d", 1e-4),
            betas=tuple(gan_cfg.get("betas", (0.5, 0.9))),
            lambda_gp=gan_cfg.get("lambda_gp", 10.0),
            n_critic=gan_cfg.get("n_critic", 5),
        )

        print("  Fitting full-dataset WGAN-GP...")
        trainer.fit(
            X_train=X_scaled,
            y_train=y_train,
            generator_steps=gen_steps,
            batch_size=batch_size,
            checkpoint_interval=gan_cfg.get("checkpoint_interval", 500),
            verbose=False,
        )

        # Generate class-balanced profiles matching training size
        n_half = len(df_train_raw) // 2
        syn_scaled = trainer.generate({0: n_half, 1: n_half})
        syn_arr = np.vstack([syn_scaled[0], syn_scaled[1]])
        syn_lbl = np.hstack([np.zeros(n_half), np.ones(n_half)])

        df_synthetic = prep.inverse_transform_synthetic(syn_arr, syn_lbl)
        df_synthetic["source"] = "synthetic"

        # Reorder columns to place label and source cleanly
        cols = [c for c in df_synthetic.columns if c not in ["label", "source"]] + ["label", "source"]
        df_synthetic = df_synthetic[cols]

        csv_path = os.path.join(export_dir, "synthetic_train.csv")
        df_synthetic.to_csv(csv_path, index=False)
        print(f"  💾 Saved {len(df_synthetic)} synthetic patient rows to: {csv_path}")

        # Write strict cautionary README.md
        readme_path = os.path.join(export_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write("# Synthetic Gastric Cancer Patient Dataset (WGAN-GP)\n\n")
            f.write("⚠️ **CRITICAL PROTOCOL RESTRICTION:**\n")
            f.write("- This synthetic dataset was generated by a Conditional WGAN-GP trained strictly on `dataset/gastric_cancer_train.csv` (569 real patients).\n")
            f.write("- **It must ONLY EVER be used for TRAINING and AUGMENTATION.**\n")
            f.write("- **NEVER use this synthetic data for validation, holdout testing, or diagnostic performance evaluation.** All evaluations must be performed strictly on real clinical patient records to avoid catastrophic evaluation bias and data leakage.\n")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run fast smoke test")
    args = parser.parse_args()

    exp = ExperimentWGAN(smoke_test=args.smoke_test)
    exp()


if __name__ == "__main__":
    main()
