"""
Training Pipeline Orchestrator for Tabular VAE + RF + AdaBoost Ensembles.
Executes Stratified 5-Fold CV, Latent Compression, Out-Of-Fold Evaluation,
Production Model Retraining, and Diagnostic Export.
"""

import os
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import StratifiedKFold

from vae_model import TabularVAE
from vae_trainer import VAETrainer
from vae_preprocessing import VAEPreprocessingPipeline
from downstream_classifiers import (
    build_rf_classifier,
    build_adaboost_classifier,
    build_soft_voting,
    build_stacking,
    save_joblib,
)
from vae_evaluate import (
    compute_metrics,
    plot_vae_loss_curves,
    plot_latent_space_tsne,
    plot_reconstruction_r2,
    plot_comparison_roc,
    plot_comparison_pr,
    plot_confusion_matrices,
    save_vae_results_summary,
)


def train_vae_ensemble_pipeline(
    raw_df: pd.DataFrame,
    model_output_dir: str,
    results_output_dir: str,
    latent_dim: int = 8,
    epochs: int = 150,
    batch_size: int = 64,
    n_splits: int = 5,
    beta: float = 0.005,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes the full VAE compression and ensemble training pipeline.
    """
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(results_output_dir, exist_ok=True)

    # 1. Preprocessing & Feature Transformation
    print("[Pipeline] Preprocessing raw clinical data...")
    pipeline = VAEPreprocessingPipeline(scaler_type="standard")
    X_all, y, feature_names = pipeline.fit_transform(raw_df)
    in_features = X_all.shape[1]
    print(f"[Pipeline] Feature matrix ready: {X_all.shape[0]} samples × {in_features} features.")
    print(f"[Pipeline] Target distribution: {np.sum(y == 1)} Cancer ({np.mean(y == 1)*100:.1f}%), {np.sum(y == 0)} Control ({np.mean(y == 0)*100:.1f}%).")

    # 2. Stratified 5-Fold Cross-Validation Setup
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    evaluated_model_names = [
        "RF (Latent z)",
        "AdaBoost (Latent z)",
        "Soft Voting (Latent z)",
        "Stacking (Latent z)",
        "Soft Voting (Hybrid: X+z)",
        "Stacking (Hybrid: X+z)",
    ]

    oof_predictions = {
        name: {
            "pred": np.zeros(len(y), dtype=int),
            "prob": np.zeros(len(y), dtype=float),
        }
        for name in evaluated_model_names
    }

    fold_records = []
    print(f"\n[Pipeline] Starting {n_splits}-Fold Stratified Cross-Validation (VAE Compression -> Ensembles)...")
    cv_start = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y), 1):
        print(f"\n--- Fold {fold}/{n_splits} (Train: {len(train_idx)}, Val: {len(val_idx)}) ---")
        X_train_raw = X_all[train_idx]
        X_val_raw = X_all[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        # Train fold-specific VAE to prevent data leakage
        fold_vae = TabularVAE(in_features=in_features, latent_dim=latent_dim)
        trainer = VAETrainer(fold_vae, lr=1e-3, beta=beta, device="cpu")
        trainer.fit(X_train_raw, X_val_raw, epochs=epochs, batch_size=batch_size, verbose=False)

        # Extract compressed latent representations
        fold_vae.eval()
        with torch.no_grad():
            z_train = fold_vae.get_latent(torch.tensor(X_train_raw, dtype=torch.float32), deterministic=True).numpy()
            z_val = fold_vae.get_latent(torch.tensor(X_val_raw, dtype=torch.float32), deterministic=True).numpy()

        # Build Hybrid representations [X, z]
        X_train_hybrid = np.hstack([X_train_raw, z_train])
        X_val_hybrid = np.hstack([X_val_raw, z_val])

        # Define candidate models for this fold
        fold_models = {
            "RF (Latent z)": (build_rf_classifier(random_state), z_train, z_val),
            "AdaBoost (Latent z)": (build_adaboost_classifier(random_state), z_train, z_val),
            "Soft Voting (Latent z)": (build_soft_voting(random_state=random_state), z_train, z_val),
            "Stacking (Latent z)": (build_stacking(random_state=random_state), z_train, z_val),
            "Soft Voting (Hybrid: X+z)": (build_soft_voting(random_state=random_state), X_train_hybrid, X_val_hybrid),
            "Stacking (Hybrid: X+z)": (build_stacking(random_state=random_state), X_train_hybrid, X_val_hybrid),
        }

        for name, (model, x_tr, x_v) in fold_models.items():
            model.fit(x_tr, y_train)
            pred = model.predict(x_v)
            prob = model.predict_proba(x_v)[:, 1]

            oof_predictions[name]["pred"][val_idx] = pred
            oof_predictions[name]["prob"][val_idx] = prob

            metrics = compute_metrics(y_val, pred, prob)
            fold_records.append({"fold": fold, "model": name, **metrics})

            print(
                f"  [{name:26s}] AUC: {metrics['roc_auc']:.4f} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}"
            )

    print(f"\n[Pipeline] Cross-Validation completed in {time.time() - cv_start:.2f}s.")
    cv_fold_df = pd.DataFrame(fold_records)

    # 3. Overall Out-Of-Fold Summary
    overall_metrics = {}
    print("\n" + "=" * 90)
    print("FINAL OUT-OF-FOLD (OOF) VAE + ENSEMBLE PERFORMANCE SUMMARY")
    print("=" * 90)
    for name in evaluated_model_names:
        pred = oof_predictions[name]["pred"]
        prob = oof_predictions[name]["prob"]
        m = compute_metrics(y, pred, prob)

        sub_df = cv_fold_df[cv_fold_df["model"] == name]
        m["cv_auc_mean"] = float(sub_df["roc_auc"].mean())
        m["cv_auc_std"] = float(sub_df["roc_auc"].std())
        m["cv_acc_mean"] = float(sub_df["accuracy"].mean())
        m["cv_acc_std"] = float(sub_df["accuracy"].std())
        m["cv_f1_mean"] = float(sub_df["f1_score"].mean())
        m["cv_f1_std"] = float(sub_df["f1_score"].std())

        overall_metrics[name] = m

        print(
            f"{name:26s} | OOF AUC: {m['roc_auc']:.4f} (CV: {m['cv_auc_mean']:.4f}±{m['cv_auc_std']:.4f}) | "
            f"Acc: {m['accuracy']:.4f} | F1: {m['f1_score']:.4f} | Recall: {m['recall_sensitivity']:.4f} | Spec: {m['specificity']:.4f}"
        )
    print("=" * 90)

    # 4. Production VAE Training on 100% Data & Diagnostic Plotting
    print("\n[Pipeline] Training production VAE on 100% of dataset...")
    prod_vae = TabularVAE(in_features=in_features, latent_dim=latent_dim)
    prod_trainer = VAETrainer(prod_vae, lr=1e-3, beta=beta, device="cpu")
    history = prod_trainer.fit(X_all, epochs=epochs, batch_size=batch_size, verbose=True)

    # Extract production latent representation
    prod_vae.eval()
    with torch.no_grad():
        Z_all = prod_vae.get_latent(torch.tensor(X_all, dtype=torch.float32), deterministic=True).numpy()
    X_all_hybrid = np.hstack([X_all, Z_all])

    # Compute per-feature reconstruction fidelity
    r2_df = prod_trainer.compute_reconstruction_r2(X_all, feature_names)

    # 5. Generate and Save Visualizations
    print("\n[Pipeline] Generating diagnostic plots...")
    plot_vae_loss_curves(history, os.path.join(results_output_dir, "vae_training_loss.png"))
    plot_latent_space_tsne(Z_all, y, os.path.join(results_output_dir, "latent_space_tsne_pca.png"))
    plot_reconstruction_r2(r2_df, os.path.join(results_output_dir, "feature_reconstruction_fidelity.png"))
    plot_comparison_roc(oof_predictions, y, os.path.join(results_output_dir, "roc_curves_comparison.png"))
    plot_comparison_pr(oof_predictions, y, os.path.join(results_output_dir, "precision_recall_curves.png"))
    plot_confusion_matrices(oof_predictions, y, os.path.join(results_output_dir, "confusion_matrices.png"))

    # Save summary files
    save_vae_results_summary(overall_metrics, cv_fold_df, r2_df, results_output_dir)

    # 6. Retrain and Serialize Production Models
    print("\n[Pipeline] Serializing production models to disk...")
    # Save VAE weights
    prod_vae.save_model(os.path.join(model_output_dir, "vae_model.pt"))
    save_joblib(pipeline, os.path.join(model_output_dir, "vae_preprocessing_pipeline.joblib"))

    # Train and save downstream models on Latent z
    latent_rf = build_rf_classifier(random_state).fit(Z_all, y)
    latent_ada = build_adaboost_classifier(random_state).fit(Z_all, y)
    latent_soft_voting = build_soft_voting(random_state=random_state).fit(Z_all, y)
    latent_stacking = build_stacking(random_state=random_state).fit(Z_all, y)

    # Train and save downstream models on Hybrid [X, z]
    hybrid_soft_voting = build_soft_voting(random_state=random_state).fit(X_all_hybrid, y)
    hybrid_stacking = build_stacking(random_state=random_state).fit(X_all_hybrid, y)

    save_joblib(latent_rf, os.path.join(model_output_dir, "latent_rf_model.joblib"))
    save_joblib(latent_ada, os.path.join(model_output_dir, "latent_adaboost_model.joblib"))
    save_joblib(latent_soft_voting, os.path.join(model_output_dir, "latent_soft_voting_model.joblib"))
    save_joblib(latent_stacking, os.path.join(model_output_dir, "latent_stacking_model.joblib"))
    save_joblib(hybrid_soft_voting, os.path.join(model_output_dir, "hybrid_soft_voting_model.joblib"))
    save_joblib(hybrid_stacking, os.path.join(model_output_dir, "hybrid_stacking_model.joblib"))

    print(f"\n[Pipeline] All models saved to: {model_output_dir}")
    print(f"[Pipeline] All results and plots saved to: {results_output_dir}")

    return {
        "metrics": overall_metrics,
        "model_dir": model_output_dir,
        "results_dir": results_output_dir,
    }
