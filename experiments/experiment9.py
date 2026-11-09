"""Generate and quality-audit synthetic data for the paper's eight features."""

import json
import os
import random
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.base_experiment import BaseExperiment
from model_src import Generator, Critic, WGANGPTrainer, TabularVAE, VAETrainer


FEATURES = ["gender", "age", "neu_lym", "hb", "alb", "cea", "ca125", "ca199"]
LOG_FEATURES = ["neu_lym", "cea", "ca125", "ca199"]
METHODS = [
    "wgan_gp_train_proportions",
    "wgan_gp_balanced",
    "vae_decoder_train_proportions",
    "vae_decoder_balanced",
    "smote_train_proportions",
    "smote_balanced",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_labels(counts: Dict[int, int]) -> np.ndarray:
    return np.concatenate([np.full(int(counts[k]), int(k), dtype=int) for k in sorted(counts)])


def postprocess_samples(values: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """Apply explicit type/domain constraints and report every adjustment."""
    out = np.asarray(values, dtype=float).copy()
    counts: Dict[str, int] = {}

    gender_new = (out[:, 0] >= 0.5).astype(float)
    counts["gender_changed_to_binary"] = int(np.sum(out[:, 0] != gender_new))
    out[:, 0] = gender_new

    age_new = np.clip(np.round(out[:, 1]), 2, 9)
    counts["age_rounded_or_clipped"] = int(np.sum(out[:, 1] != age_new))
    out[:, 1] = age_new

    for idx, feature in enumerate(FEATURES[2:], start=2):
        below_zero = out[:, idx] < 0
        counts[f"{feature}_clipped_below_zero"] = int(below_zero.sum())
        out[below_zero, idx] = 0.0

    hb_bad = out[:, FEATURES.index("hb")] > 300.0
    alb_bad = out[:, FEATURES.index("alb")] > 100.0
    counts["hb_decimal_corrections"] = int(hb_bad.sum())
    counts["alb_decimal_corrections"] = int(alb_bad.sum())
    out[hb_bad, FEATURES.index("hb")] /= 10.0
    out[alb_bad, FEATURES.index("alb")] /= 10.0
    return out, counts


def export_frame(X: np.ndarray, y: np.ndarray, method: str) -> pd.DataFrame:
    df = pd.DataFrame(X, columns=FEATURES)
    df["label"] = y.astype(int)
    df["source"] = "synthetic"
    df["generation_method"] = method
    return df


def transform_for_generator(X: np.ndarray) -> np.ndarray:
    """Log1p-transform skewed nonnegative features before training a generator."""
    transformed = np.asarray(X, dtype=float).copy()
    for feature in LOG_FEATURES:
        column = FEATURES.index(feature)
        transformed[:, column] = np.log1p(np.clip(transformed[:, column], 0.0, None))
    return transformed


def inverse_generator_transform(X_scaled: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Return generated vectors to clinical units after the train-fitted scaler."""
    values = scaler.inverse_transform(X_scaled)
    for feature in LOG_FEATURES:
        column = FEATURES.index(feature)
        values[:, column] = np.expm1(np.clip(values[:, column], 0.0, 15.0))
    return values


def generate_smote_sets(
    X_scaled: np.ndarray,
    y: np.ndarray,
    scaler: StandardScaler,
    seed: int,
    class_counts: Dict[int, int],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Ask SMOTE to add a specified number of examples to each class."""
    train_n = len(y)
    balanced_counts = {0: train_n // 2, 1: train_n - train_n // 2}
    targets = {
        "train_proportions": {k: int(class_counts[k] * 2) for k in class_counts},
        "balanced": {k: int(class_counts[k] + balanced_counts[k]) for k in class_counts},
    }
    results = {}
    for suffix, target_counts in targets.items():
        smote = SMOTE(
            sampling_strategy=target_counts,
            random_state=seed,
            k_neighbors=min(3, min(class_counts.values()) - 1),
        )
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
        # SMOTE appends the generated rows after the original observations.
        X_synthetic = inverse_generator_transform(X_resampled[train_n:], scaler)
        y_synthetic = np.asarray(y_resampled[train_n:], dtype=int)
        X_synthetic, corrections = postprocess_samples(X_synthetic)
        results[f"smote_{suffix}"] = (X_synthetic, y_synthetic)
        results[f"smote_{suffix}_corrections"] = corrections
    return results


def nearest_neighbor_audit(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    features: List[str],
    threshold_percentile: float = 5.0,
) -> Dict[str, float]:
    """Compare novelty distances against the training cohort's own spacing."""
    X_real = real_train[features].to_numpy(dtype=float)
    X_syn = synthetic[features].to_numpy(dtype=float)
    center = X_real.mean(axis=0)
    scale = X_real.std(axis=0)
    scale[scale == 0] = 1.0
    real_z = (X_real - center) / scale
    syn_z = (X_syn - center) / scale

    real_pairwise = cdist(real_z, real_z, metric="euclidean")
    np.fill_diagonal(real_pairwise, np.inf)
    real_nn = np.min(real_pairwise, axis=1)
    syn_to_real = np.min(cdist(syn_z, real_z, metric="euclidean"), axis=1)
    threshold = float(np.percentile(real_nn, threshold_percentile))

    syn_pairwise = cdist(syn_z, syn_z, metric="euclidean")
    if len(syn_pairwise) > 1:
        np.fill_diagonal(syn_pairwise, np.inf)
        syn_nn = np.min(syn_pairwise, axis=1)
        synthetic_exact_pairs = int((syn_pairwise == 0).sum() // 2)
    else:
        syn_nn = np.array([np.nan])
        synthetic_exact_pairs = 0

    real_key = set(map(tuple, X_real.tolist()))
    exact_against_train = sum(tuple(row.tolist()) in real_key for row in X_syn)
    return {
        "exact_duplicate_rows_vs_training": int(exact_against_train),
        "exact_duplicate_fraction_vs_training": float(exact_against_train / max(1, len(X_syn))),
        "near_duplicate_threshold_train_nn_p5": threshold,
        "near_duplicate_fraction_vs_training": float(np.mean(syn_to_real <= threshold)),
        "median_synthetic_to_train_nn_distance": float(np.median(syn_to_real)),
        "median_real_to_real_nn_distance": float(np.median(real_nn)),
        "median_synthetic_to_synthetic_nn_distance": float(np.nanmedian(syn_nn)),
        "exact_duplicate_pairs_within_synthetic": synthetic_exact_pairs,
    }


def plot_feature_distributions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Plot each selected feature against real train and untouched test data."""
    os.makedirs(output_dir, exist_ok=True)
    sources = [
        ("Real training", train_df, "#266b9a"),
        ("Real held-out test", test_df, "#48a078"),
        ("WGAN-GP synthetic", synthetic_df, "#e27a35"),
    ]
    for label in [None, 0, 1]:
        fig, axes = plt.subplots(2, 4, figsize=(17, 8), dpi=180)
        axes = axes.flatten()
        for ax, feature in zip(axes, FEATURES):
            class_title = "All labels" if label is None else f"Label {label}"
            filtered = []
            for title, frame, color in sources:
                data = frame if label is None else frame.loc[frame.label == label]
                filtered.append((title, data, color))

            if feature in ("gender", "age"):
                categories = sorted({
                    int(round(float(value)))
                    for _, frame, _ in filtered
                    for value in frame[feature].dropna().unique()
                })
                width = 0.25
                x = np.arange(len(categories))
                for source_index, (title, frame, color) in enumerate(filtered):
                    proportions = frame[feature].value_counts(normalize=True)
                    heights = [float(proportions.get(category, 0.0)) for category in categories]
                    ax.bar(x + (source_index - 1) * width, heights, width=width, label=title, color=color)
                ax.set_xticks(x)
                ax.set_xticklabels(categories)
                ax.set_ylabel("Proportion")
            else:
                display_log = feature in LOG_FEATURES
                for title, frame, color in filtered:
                    values = frame[feature].dropna().to_numpy(dtype=float)
                    if display_log:
                        values = np.log1p(np.clip(values, 0.0, None))
                    if len(values):
                        ax.hist(values, bins=20, density=True, histtype="step", linewidth=1.8,
                                label=title, color=color)
                ax.set_ylabel("Density")
                ax.set_xlabel("log1p(value)" if display_log else "Value")
            ax.set_title(feature)
            ax.grid(axis="y", alpha=0.2)
            if feature == "gender":
                ax.set_xticklabels(["Female (0)" if x == 0 else "Male (1)" for x in categories], rotation=15)
            if feature == "age":
                ax.set_xlabel("Encoded age category")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.015), ncol=3, frameon=False)
        fig.suptitle(f"Eight-feature distributions: real vs WGAN-GP synthetic ({class_title})", y=1.08)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        suffix = "overall" if label is None else f"label_{label}"
        fig.savefig(os.path.join(output_dir, f"feature_distribution_{suffix}.png"), bbox_inches="tight")
        plt.close(fig)


def quality_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    synthetic_sets: Dict[str, pd.DataFrame],
    output_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assess class-conditional feature fidelity and memorization without classifiers."""
    marginal_records = []
    summary_records = []
    for method, syn_df in synthetic_sets.items():
        for label in sorted(test_df["label"].unique()):
            real_class = test_df.loc[test_df.label == label, FEATURES]
            syn_class = syn_df.loc[syn_df.label == label, FEATURES]
            for feature in FEATURES:
                real_values = real_class[feature].dropna().to_numpy(dtype=float)
                syn_values = syn_class[feature].dropna().to_numpy(dtype=float)
                if len(syn_values) == 0:
                    continue
                ks = ks_2samp(real_values, syn_values)
                marginal_records.append({
                    "method": method,
                    "label": int(label),
                    "feature": feature,
                    "real_test_mean": float(np.mean(real_values)),
                    "synthetic_mean": float(np.mean(syn_values)),
                    "real_test_std": float(np.std(real_values)),
                    "synthetic_std": float(np.std(syn_values)),
                    "ks_statistic": float(ks.statistic),
                    "ks_pvalue_descriptive": float(ks.pvalue),
                    "wasserstein_distance": float(wasserstein_distance(real_values, syn_values)),
                    "wasserstein_distance_in_train_sd": float(
                        wasserstein_distance(real_values, syn_values)
                        / max(float(train_df[feature].std(ddof=1)), 1e-12)
                    ),
                    "n_real_test": int(len(real_values)),
                    "n_synthetic": int(len(syn_values)),
                })

        corr_distances = []
        for label in sorted(test_df["label"].unique()):
            real_class = test_df.loc[test_df.label == label, FEATURES]
            syn_class = syn_df.loc[syn_df.label == label, FEATURES]
            if len(syn_class) < 3 or len(real_class) < 3:
                continue
            real_corr = real_class.corr(method="spearman").fillna(0).to_numpy()
            syn_corr = syn_class.corr(method="spearman").fillna(0).to_numpy()
            corr_distances.append(float(np.linalg.norm(real_corr - syn_corr, ord="fro")))

        nn = nearest_neighbor_audit(train_df, syn_df, FEATURES)
        syn = syn_df[FEATURES]
        valid_gender = syn.gender.isin([0, 1])
        valid_age = (syn.age.between(2, 9)) & np.isclose(syn.age, np.round(syn.age))
        valid_nonnegative = (syn[FEATURES[2:]] >= 0).all(axis=1)
        class_counts = syn_df.label.value_counts().to_dict()
        summary_records.append({
            "method": method,
            "synthetic_rows": int(len(syn_df)),
            "synthetic_class_0": int(class_counts.get(0, 0)),
            "synthetic_class_1": int(class_counts.get(1, 0)),
            "mean_class_conditional_ks": float(np.mean([
                row["ks_statistic"] for row in marginal_records if row["method"] == method
            ])),
            "mean_class_conditional_wasserstein_train_sd": float(np.mean([
                row["wasserstein_distance_in_train_sd"]
                for row in marginal_records if row["method"] == method
            ])),
            "mean_spearman_corr_frobenius_distance": float(np.mean(corr_distances)) if corr_distances else np.nan,
            "invalid_gender_rows": int((~valid_gender).sum()),
            "invalid_age_rows": int((~valid_age).sum()),
            "negative_biomarker_rows": int((~valid_nonnegative).sum()),
            **nn,
        })

    marginals = pd.DataFrame(marginal_records)
    summary = pd.DataFrame(summary_records)
    marginals.to_csv(os.path.join(output_dir, "class_conditional_marginal_metrics.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "synthetic_quality_summary.csv"), index=False)

    # PCA projection is only a visual diagnostic, not a pass/fail criterion.
    scaler = StandardScaler().fit(train_df[FEATURES])
    real_z = scaler.transform(test_df[FEATURES])
    labels = test_df.label.to_numpy()
    plot_rows = []
    for method, syn_df in synthetic_sets.items():
        syn_z = scaler.transform(syn_df[FEATURES])
        combined = np.vstack([real_z, syn_z])
        coords = PCA(n_components=2, random_state=42).fit_transform(combined)
        plot_rows.append(pd.DataFrame({
            "pc1": coords[:, 0], "pc2": coords[:, 1],
            "source": ["Real holdout"] * len(real_z) + ["Synthetic"] * len(syn_z),
            "label": np.concatenate([labels, syn_df.label.to_numpy()]),
            "method": method,
        }))
    plot_df = pd.concat(plot_rows, ignore_index=True)
    methods = list(synthetic_sets)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=160)
    for ax, method in zip(axes.flatten(), methods):
        part = plot_df[plot_df.method == method]
        for source, color in [("Real holdout", "#246a9b"), ("Synthetic", "#e07a35")]:
            rows = part[part.source == source]
            ax.scatter(rows.pc1, rows.pc2, s=14, alpha=0.55, label=source, c=color)
        ax.set_title(method.replace("_", " "))
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    if methods:
        axes.flatten()[0].legend(fontsize=8)
    for ax in axes.flatten()[len(methods):]:
        ax.axis("off")
    fig.suptitle("PCA overview: real held-out rows and synthetic rows")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pca_real_vs_synthetic.png"))
    plt.close(fig)

    # Focused eight-feature plots for the WGAN-GP training-proportion dataset.
    wgan_key = "wgan_gp_train_proportions"
    if wgan_key in synthetic_sets:
        plot_feature_distributions(
            train_df=train_df,
            test_df=test_df,
            synthetic_df=synthetic_sets[wgan_key],
            output_dir=output_dir,
        )
    return marginals, summary


class Experiment9(BaseExperiment):
    """Generate reusable eight-feature synthetic datasets and audit fidelity/novelty."""

    experiment_name = "experiment9"

    def run(self):
        cfg = self.config
        seed = int(cfg.get("seed", 42))
        seed_everything(seed)
        torch.set_num_threads(int(cfg.get("runtime", {}).get("torch_num_threads", 1)))

        ds_cfg = cfg.get("dataset", {})
        train_path = self._resolve_path(ds_cfg.get(
            "train_path", "dataset/split70_30/processed_and_selected_feature/gastric_cancer_train.csv"
        ))
        test_path = self._resolve_path(ds_cfg.get(
            "test_path", "dataset/split70_30/processed_and_selected_feature/gastric_cancer_test.csv"
        ))
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        for name, frame in (("train", train_df), ("test", test_df)):
            missing = [c for c in FEATURES + ["label"] if c not in frame]
            if missing:
                raise ValueError(f"{name} split lacks required selected features: {missing}")
            if frame[FEATURES + ["label"]].isna().any().any():
                raise ValueError(f"{name} split contains missing model values; resolve before synthesis")

        train_df = train_df[FEATURES + ["label"]].copy()
        test_df = test_df[FEATURES + ["label"]].copy()
        X_train = train_df[FEATURES].to_numpy(dtype=float)
        y_train = train_df.label.to_numpy(dtype=int)
        class_counts = {int(k): int(v) for k, v in train_df.label.value_counts().sort_index().items()}
        balanced_each = len(train_df) // 2
        train_prop_counts = class_counts.copy()
        balanced_counts = {0: balanced_each, 1: len(train_df) - balanced_each}
        X_train_transformed = transform_for_generator(X_train)
        scaler = StandardScaler().fit(X_train_transformed)
        X_scaled = scaler.transform(X_train_transformed)

        out_dir = self._resolve_path(cfg.get("synthetic_output_dir", "dataset/synthetic_8_features"))
        os.makedirs(out_dir, exist_ok=True)
        metrics_dir = self.get_results_subdir("quality")
        gen_cfg = cfg.get("wgan_gp", {})
        vae_cfg = cfg.get("vae_decoder", {})
        synthetic_sets: Dict[str, pd.DataFrame] = {}
        corrections: Dict[str, Dict[str, int]] = {}

        # Conditional WGAN-GP trained only on the saved 70% training partition.
        seed_everything(seed)
        generator = Generator(
            latent_dim=int(gen_cfg.get("latent_dim", 16)),
            num_classes=2,
            output_dim=len(FEATURES),
            hidden_dim=int(gen_cfg.get("hidden_dim", 128)),
            use_layer_norm=bool(gen_cfg.get("use_layer_norm", True)),
            use_batch_norm=False,
        )
        critic = Critic(
            input_dim=len(FEATURES), num_classes=2,
            hidden_dim=int(gen_cfg.get("hidden_dim", 128)),
            use_layer_norm=bool(gen_cfg.get("use_layer_norm", True)),
        )
        wgan = WGANGPTrainer(
            generator=generator,
            critic=critic,
            lr_g=float(gen_cfg.get("learning_rate_g", 1e-4)),
            lr_d=float(gen_cfg.get("learning_rate_d", 1e-4)),
            betas=tuple(gen_cfg.get("betas", [0.5, 0.9])),
            lambda_gp=float(gen_cfg.get("lambda_gp", 10.0)),
            n_critic=int(gen_cfg.get("n_critic", 5)),
            device=str(gen_cfg.get("device", "cpu")),
        )
        print(f"Training conditional WGAN-GP on {len(train_df)} real training rows...")
        wgan.fit(
            X_train=X_scaled,
            y_train=y_train,
            generator_steps=int(gen_cfg.get("generator_steps", 5000)),
            batch_size=int(gen_cfg.get("batch_size", 64)),
            checkpoint_interval=int(gen_cfg.get("checkpoint_interval", 500)),
            verbose=bool(gen_cfg.get("verbose", True)),
        )
        for name, counts in (("wgan_gp_train_proportions", train_prop_counts), ("wgan_gp_balanced", balanced_counts)):
            generated_scaled = wgan.generate(counts)
            x_parts = [generated_scaled[k] for k in sorted(generated_scaled)]
            y_parts = [np.full(counts[k], k, dtype=int) for k in sorted(counts)]
            generated = inverse_generator_transform(np.vstack(x_parts), scaler)
            generated, corrections[name] = postprocess_samples(generated)
            synthetic_sets[name] = export_frame(generated, np.concatenate(y_parts), name)

        # Plain VAE decoder generation: fit on features only; sample from each
        # class's empirical latent Gaussian after fitting the encoder on training rows.
        seed_everything(seed)
        vae = TabularVAE(
            in_features=len(FEATURES),
            latent_dim=int(vae_cfg.get("latent_dim", 4)),
            hidden_dim1=int(vae_cfg.get("hidden_dim1", 128)),
            hidden_dim2=int(vae_cfg.get("hidden_dim2", 64)),
            dropout_rate=float(vae_cfg.get("dropout_rate", 0.1)),
        )
        vae_trainer = VAETrainer(
            model=vae,
            lr=float(vae_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(vae_cfg.get("weight_decay", 1e-5)),
            beta=float(vae_cfg.get("beta", 0.005)),
            device=str(vae_cfg.get("device", "cpu")),
        )
        print("Training plain VAE decoder on training features (labels not used for VAE fit)...")
        vae_trainer.fit(
            X_scaled,
            epochs=int(vae_cfg.get("epochs", 100)),
            batch_size=int(vae_cfg.get("batch_size", 64)),
            verbose=False,
        )
        vae.eval()
        with torch.no_grad():
            latent = vae.get_latent(torch.tensor(X_scaled, dtype=torch.float32), deterministic=True).cpu().numpy()

        for name, counts in (("vae_decoder_train_proportions", train_prop_counts), ("vae_decoder_balanced", balanced_counts)):
            latent_draws = []
            label_draws = []
            for label in sorted(counts):
                class_latent = latent[y_train == label]
                mu = np.mean(class_latent, axis=0)
                covariance = np.atleast_2d(np.cov(class_latent, rowvar=False))
                covariance += 1e-4 * np.eye(covariance.shape[0])
                latent_draws.append(np.random.multivariate_normal(mu, covariance, int(counts[label])))
                label_draws.append(np.full(int(counts[label]), label, dtype=int))
            with torch.no_grad():
                generated_scaled = vae.decode(torch.tensor(np.vstack(latent_draws), dtype=torch.float32)).cpu().numpy()
            generated = inverse_generator_transform(generated_scaled, scaler)
            generated, corrections[name] = postprocess_samples(generated)
            synthetic_sets[name] = export_frame(generated, np.concatenate(label_draws), name)

        # SMOTE baseline; add one synthetic training-sized copy in the training
        # class proportions, and another balanced synthetic set for comparison.
        smote_results = generate_smote_sets(X_scaled, y_train, scaler, seed, class_counts)
        for name in ("smote_train_proportions", "smote_balanced"):
            generated, labels = smote_results[name]
            corrections[name] = smote_results[f"{name}_corrections"]
            synthetic_sets[name] = export_frame(generated, labels, name)

        # Save synthetic samples and a provenance record for future experiments.
        for name, synthetic in synthetic_sets.items():
            synthetic.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
        metadata = {
            "experiment": "experiment9",
            "seed": seed,
            "source_train_file": os.path.relpath(train_path, PROJECT_ROOT),
            "reference_test_file": os.path.relpath(test_path, PROJECT_ROOT),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "features": FEATURES,
            "log1p_features_for_generator_training": LOG_FEATURES,
            "label_column": "label",
            "id_column_used_for_generation": False,
            "generation_fit_scope": "70% processed training split only",
            "quality_reference": "real held-out test split (not used to fit generators)",
            "training_class_counts": class_counts,
            "generation_counts": {name: {str(k): int(v) for k, v in df.label.value_counts().sort_index().items()} for name, df in synthetic_sets.items()},
            "wgan_gp": gen_cfg,
            "vae_decoder": vae_cfg,
            "smote": {"random_state": seed, "k_neighbors": min(3, min(class_counts.values()) - 1)},
            "postprocessing_adjustments": corrections,
            "metrics_note": "No diagnostic classifier, GB model, or TSTR model was trained; fidelity and novelty are assessed statistically and with nearest-neighbor distances.",
        }
        with open(os.path.join(out_dir, "generation_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print("Computing class-conditional fidelity and duplicate/novelty checks against untouched real test data...")
        marginals, summary = quality_report(train_df, test_df, synthetic_sets, metrics_dir)
        # Mirror reports into the reusable dataset folder so files travel together.
        marginals.to_csv(os.path.join(out_dir, "class_conditional_marginal_metrics.csv"), index=False)
        summary.to_csv(os.path.join(out_dir, "synthetic_quality_summary.csv"), index=False)
        import shutil
        shutil.copy2(os.path.join(metrics_dir, "pca_real_vs_synthetic.png"), os.path.join(out_dir, "pca_real_vs_synthetic.png"))
        for suffix in ("overall", "label_0", "label_1"):
            name = f"feature_distribution_{suffix}.png"
            shutil.copy2(os.path.join(metrics_dir, name), os.path.join(out_dir, name))
        print(f"\nSynthetic datasets saved to: {out_dir}")
        print(summary[["method", "synthetic_rows", "mean_class_conditional_ks", "mean_class_conditional_wasserstein_train_sd", "near_duplicate_fraction_vs_training", "exact_duplicate_fraction_vs_training"]].to_string(index=False))
        return {"synthetic_dir": out_dir, "summary": summary, "marginals": marginals}


def main():
    Experiment9()()


if __name__ == "__main__":
    main()
