"""
Synthetic Data Fidelity and Diagnostic Suite.
Evaluates statistical alignment, feature correlation preservation,
latent space fidelity, memorization, discriminator detectability, and TSTR.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from model_src.tree.gradient_boosting import build_gradient_boosting


def evaluate_marginal_distributions(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    features: List[str],
    output_dir: str,
    top_k_plots: int = 6,
) -> pd.DataFrame:
    """
    Computes per-feature Mean, Std, KS-statistic, and KS p-value.
    Plots histogram/KDE overlays for the top mismatched features.
    """
    os.makedirs(output_dir, exist_ok=True)
    stats_records = []

    for feat in features:
        real_vals = real_df[feat].dropna().values
        syn_vals = synthetic_df[feat].dropna().values

        ks_res = ks_2samp(real_vals, syn_vals)
        stats_records.append({
            "feature": feat,
            "real_mean": float(np.mean(real_vals)),
            "real_std": float(np.std(real_vals)),
            "syn_mean": float(np.mean(syn_vals)),
            "syn_std": float(np.std(syn_vals)),
            "mean_diff_pct": float(abs(np.mean(real_vals) - np.mean(syn_vals)) / (np.std(real_vals) + 1e-5) * 100),
            "ks_statistic": float(ks_res.statistic),
            "ks_pvalue": float(ks_res.pvalue),
        })

    summary_df = pd.DataFrame(stats_records).sort_values("ks_statistic", ascending=False)
    summary_df.to_csv(os.path.join(output_dir, "ks_marginal_statistics.csv"), index=False)

    # Plot top mismatched features
    top_features = summary_df.head(top_k_plots)["feature"].tolist()
    if top_features:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes = axes.flatten()

        for i, feat in enumerate(top_features):
            ax = axes[i]
            sns.kdeplot(real_df[feat], ax=ax, label="Real (Held-Out)", color="steelblue", fill=True, alpha=0.3)
            sns.kdeplot(synthetic_df[feat], ax=ax, label="Synthetic", color="coral", fill=True, alpha=0.3)
            ks_stat = summary_df.loc[summary_df["feature"] == feat, "ks_statistic"].values[0]
            ax.set_title(f"{feat} (KS={ks_stat:.3f})", fontsize=11, fontweight="bold")
            ax.set_xlabel("")
            if i == 0:
                ax.legend(loc="upper right")

        plt.suptitle("Marginal Distribution Overlays: Top Discrepant Biomarkers", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "marginal_distribution_overlays.png"), dpi=200)
        plt.close()

    return summary_df


def evaluate_correlation_distance(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    features: List[str],
    output_dir: str,
) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """
    Computes Pearson correlation matrices for real and synthetic data,
    the Frobenius norm of their difference, and renders side-by-side heatmaps.
    """
    os.makedirs(output_dir, exist_ok=True)

    real_corr = real_df[features].corr()
    syn_corr = synthetic_df[features].corr()

    diff_matrix = (real_corr - syn_corr).abs()
    frobenius_dist = float(np.linalg.norm(real_corr.values - syn_corr.values, ord="fro"))

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    sns.heatmap(real_corr, ax=axes[0], cmap="coolwarm", vmin=-1, vmax=1, cbar=False)
    axes[0].set_title("Real Patient Correlation", fontsize=12, fontweight="bold")

    sns.heatmap(syn_corr, ax=axes[1], cmap="coolwarm", vmin=-1, vmax=1, cbar=False)
    axes[1].set_title("Synthetic Patient Correlation", fontsize=12, fontweight="bold")

    sns.heatmap(diff_matrix, ax=axes[2], cmap="Reds", vmin=0, vmax=0.6, cbar=True)
    axes[2].set_title(f"Absolute Difference (|Real - Syn|)\nFrobenius Norm: {frobenius_dist:.4f}", fontsize=12, fontweight="bold")

    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_comparison_heatmaps.png"), dpi=200)
    plt.close()

    return frobenius_dist, real_corr, syn_corr


def evaluate_pca_tsne_overlap(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    features: List[str],
    label_col: str,
    output_dir: str,
) -> None:
    """
    Overlays PCA and t-SNE 2D projections of real and synthetic data,
    colored by Source (Real vs Synthetic) and Class Label (Control vs Cancer).
    """
    os.makedirs(output_dir, exist_ok=True)

    r_feat = real_df[features].copy()
    r_feat["source"] = "Real"
    r_feat["label"] = real_df[label_col].values

    s_feat = synthetic_df[features].copy()
    s_feat["source"] = "Synthetic"
    s_feat["label"] = synthetic_df[label_col].values

    combined = pd.concat([r_feat, s_feat], ignore_index=True)
    X = combined[features].values

    # Standardize for projection
    X_norm = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-5)

    # 1. PCA
    pca = PCA(n_components=2, random_state=42)
    pca_proj = pca.fit_transform(X_norm)
    combined["pca_1"] = pca_proj[:, 0]
    combined["pca_2"] = pca_proj[:, 1]

    # 2. t-SNE
    perp = min(30, max(5, len(combined) // 10))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    tsne_proj = tsne.fit_transform(X_norm)
    combined["tsne_1"] = tsne_proj[:, 0]
    combined["tsne_2"] = tsne_proj[:, 1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # PCA by Source
    sns.scatterplot(data=combined, x="pca_1", y="pca_2", hue="source", ax=axes[0, 0], alpha=0.6, palette={"Real": "#1f77b4", "Synthetic": "#ff7f0e"})
    axes[0, 0].set_title(f"PCA Projection: Source (Var: {sum(pca.explained_variance_ratio_):.2f})", fontweight="bold")

    # PCA by Label
    sns.scatterplot(data=combined, x="pca_1", y="pca_2", hue="label", style="source", ax=axes[0, 1], alpha=0.6, palette="viridis")
    axes[0, 1].set_title("PCA Projection: Class Label", fontweight="bold")

    # t-SNE by Source
    sns.scatterplot(data=combined, x="tsne_1", y="tsne_2", hue="source", ax=axes[1, 0], alpha=0.6, palette={"Real": "#1f77b4", "Synthetic": "#ff7f0e"})
    axes[1, 0].set_title("t-SNE Projection: Source", fontweight="bold")

    # t-SNE by Label
    sns.scatterplot(data=combined, x="tsne_1", y="tsne_2", hue="label", style="source", ax=axes[1, 1], alpha=0.6, palette="viridis")
    axes[1, 1].set_title("t-SNE Projection: Class Label", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_tsne_manifold_overlap.png"), dpi=200)
    plt.close()


def evaluate_discriminator_detection_auc(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    features: List[str],
) -> Tuple[float, float]:
    """
    Trains an adversarial classifier to distinguish between Real and Synthetic rows.
    Returns:
        (mean_detection_auc, std_detection_auc)
        Near 0.50 is ideal (indistinguishable), near 1.0 indicates synthetic data is easily detected.
    """
    r_X = real_df[features].values
    s_X = synthetic_df[features].values

    X = np.vstack([r_X, s_X])
    y = np.hstack([np.zeros(len(r_X)), np.ones(len(s_X))])

    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    auc_scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    return float(np.mean(auc_scores)), float(np.std(auc_scores))


def evaluate_memorization_check(
    real_train_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    features: List[str],
    near_duplicate_percentile: float = 0.05,
) -> Dict[str, float]:
    """
    Evaluates privacy and memorization by measuring Nearest Neighbor distances in standardized space.
    Compares synthetic-to-training distances against real-to-real distances.
    """
    r_X = real_train_df[features].values
    s_X = synthetic_df[features].values

    # Standardize
    mean_val = np.mean(r_X, axis=0)
    std_val = np.std(r_X, axis=0) + 1e-5
    r_scaled = (r_X - mean_val) / std_val
    s_scaled = (s_X - mean_val) / std_val

    # Real-to-real nearest neighbors (k=2 because 1st neighbor is self)
    nn_real = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn_real.fit(r_scaled)
    r_distances, _ = nn_real.kneighbors(r_scaled)
    real_real_dist = r_distances[:, 1]  # distance to nearest distinct real neighbor

    # Synthetic-to-real nearest neighbors (k=1)
    nn_train = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn_train.fit(r_scaled)
    s_distances, _ = nn_train.kneighbors(s_scaled)
    syn_real_dist = s_distances[:, 0]

    # Threshold for "near-duplicate" based on lower percentile of real-to-real distance
    duplicate_threshold = float(np.percentile(real_real_dist, near_duplicate_percentile * 100))
    near_duplicate_fraction = float(np.mean(syn_real_dist < duplicate_threshold))

    return {
        "mean_real_to_real_nn_dist": float(np.mean(real_real_dist)),
        "median_real_to_real_nn_dist": float(np.median(real_real_dist)),
        "mean_syn_to_real_nn_dist": float(np.mean(syn_real_dist)),
        "median_syn_to_real_nn_dist": float(np.median(syn_real_dist)),
        "near_duplicate_threshold": duplicate_threshold,
        "near_duplicate_fraction": near_duplicate_fraction,
    }


def evaluate_tstr(
    synthetic_df: pd.DataFrame,
    held_out_real_df: pd.DataFrame,
    real_train_df: pd.DataFrame,
    features: List[str],
    label_col: str,
) -> Dict[str, float]:
    """
    TSTR (Train on Synthetic, Test on Real).
    Compares classifier trained purely on synthetic data vs trained on real training data,
    both evaluated on the exact same held-out real slice.
    """
    # 1. Train on Synthetic Only -> Test on Real
    clf_tstr = build_gradient_boosting(random_state=42)
    clf_tstr.fit(synthetic_df[features].values, synthetic_df[label_col].values)
    y_test_real = held_out_real_df[label_col].values
    prob_tstr = clf_tstr.predict_proba(held_out_real_df[features].values)[:, 1]
    auc_tstr = float(roc_auc_score(y_test_real, prob_tstr))

    # 2. Baseline: Train on Real Train Only -> Test on Real
    clf_trtr = build_gradient_boosting(random_state=42)
    clf_trtr.fit(real_train_df[features].values, real_train_df[label_col].values)
    prob_trtr = clf_trtr.predict_proba(held_out_real_df[features].values)[:, 1]
    auc_trtr = float(roc_auc_score(y_test_real, prob_trtr))

    return {
        "tstr_auc": auc_tstr,
        "trtr_real_baseline_auc": auc_trtr,
        "tstr_ratio": float(auc_tstr / (auc_trtr + 1e-5)),
    }


def run_full_fidelity_suite(
    real_train_df: pd.DataFrame,
    held_out_real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    features: List[str],
    label_col: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Executes the entire comprehensive fidelity evaluation suite:
    1. Marginal KS statistics & histograms
    2. Pearson correlation matrix distance & heatmaps
    3. PCA and t-SNE 2D manifold overlays
    4. Adversarial detection discriminator AUC
    5. Memorization & duplicate fraction
    6. TSTR vs TRTR benchmark
    """
    os.makedirs(output_dir, exist_ok=True)
    report: Dict[str, Any] = {}

    print("  [Fidelity] 1. Computing marginal distributions and KS statistics...")
    ks_df = evaluate_marginal_distributions(
        real_df=held_out_real_df,
        synthetic_df=synthetic_df,
        features=features,
        output_dir=output_dir,
    )
    report["mean_ks_statistic"] = float(ks_df["ks_statistic"].mean())
    report["significant_ks_fraction_p05"] = float((ks_df["ks_pvalue"] < 0.05).mean())

    print("  [Fidelity] 2. Computing Pearson correlation Frobenius distance...")
    frob_dist, _, _ = evaluate_correlation_distance(
        real_df=held_out_real_df,
        synthetic_df=synthetic_df,
        features=features,
        output_dir=output_dir,
    )
    report["correlation_frobenius_distance"] = frob_dist

    print("  [Fidelity] 3. Generating PCA & t-SNE manifold overlays...")
    evaluate_pca_tsne_overlap(
        real_df=held_out_real_df,
        synthetic_df=synthetic_df,
        features=features,
        label_col=label_col,
        output_dir=output_dir,
    )

    print("  [Fidelity] 4. Training real-vs-synthetic discriminator (detection AUC)...")
    det_auc_mean, det_auc_std = evaluate_discriminator_detection_auc(
        real_df=held_out_real_df,
        synthetic_df=synthetic_df,
        features=features,
    )
    report["detection_auc_mean"] = det_auc_mean
    report["detection_auc_std"] = det_auc_std

    print("  [Fidelity] 5. Running nearest-neighbor memorization audit...")
    mem_report = evaluate_memorization_check(
        real_train_df=real_train_df,
        synthetic_df=synthetic_df,
        features=features,
    )
    report.update(mem_report)

    print("  [Fidelity] 6. Running TSTR (Train on Synthetic, Test on Real)...")
    tstr_report = evaluate_tstr(
        synthetic_df=synthetic_df,
        held_out_real_df=held_out_real_df,
        real_train_df=real_train_df,
        features=features,
        label_col=label_col,
    )
    report.update(tstr_report)

    # Save summary dictionary
    summary_path = os.path.join(output_dir, "fidelity_summary_report.csv")
    pd.DataFrame([report]).to_csv(summary_path, index=False)
    print(f"  [Fidelity] Full fidelity suite completed. Saved to: {output_dir}")

    return report
