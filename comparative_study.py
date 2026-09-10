import os
import sys
import shutil
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from typing import Dict, Any

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'VAE', 'src')))
from preprocessing import PreprocessingPipeline

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Import evaluation functions
from evaluation import (
    compute_metrics,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_confusion_matrices,
    plot_feature_importance,
    plot_vae_loss_curves,
    plot_latent_space_tsne,
    plot_reconstruction_r2
)

# Import VAE components
from vae_model import TabularVAE
from vae_trainer import VAETrainer
from sklearn.metrics import r2_score

def build_models(random_state=42):
    rf = RandomForestClassifier(n_estimators=250, max_depth=6, class_weight="balanced", random_state=random_state)
    ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=150, learning_rate=0.05, random_state=random_state)
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=random_state)
    
    voting = VotingClassifier(
        estimators=[("RF", rf), ("Ada", ada), ("GB", gb)],
        voting="soft"
    )
    stacking = StackingClassifier(
        estimators=[("RF", rf), ("Ada", ada), ("GB", gb)],
        final_estimator=LogisticRegression(max_iter=500),
        cv=5
    )
    
    return {
        "Random Forest": rf,
        "AdaBoost": ada,
        "Gradient Boosting": gb,
        "Soft Voting (RF+Ada+GB)": voting,
        "Stacking (RF+Ada+GB)": stacking
    }

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(root_dir, "dataset")
    results_dir = os.path.join(root_dir, "results", "splittrainresult")
    
    # Erase old results
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
        
    rf_results_dir = os.path.join(results_dir, "Classical_Trees")
    vae_results_dir = os.path.join(results_dir, "VAE_Trees")
    
    os.makedirs(rf_results_dir, exist_ok=True)
    os.makedirs(vae_results_dir, exist_ok=True)
    
    # 1. Load Data
    train_df = pd.read_csv(os.path.join(dataset_dir, "gastric_cancer_train.csv"))
    test_df = pd.read_csv(os.path.join(dataset_dir, "gastric_cancer_test.csv"))
    
    # 2. Preprocess
    pipeline = PreprocessingPipeline(feature_mode="optimized_subset", scaler_type="robust", apply_scaling=True)
    X_train, y_train, feature_names = pipeline.fit_transform(train_df)
    X_test, y_test = pipeline.transform(test_df)
    
    # =========================================================
    # PART 1: Classical Models (Directly on Biomarkers)
    # =========================================================
    print("=== PART 1: Classical Trees on Raw Features ===")
    models = build_models()
    test_predictions = {}
    metrics_summary = {}
    rf_importances = None
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        test_predictions[name] = {"pred": y_pred, "prob": y_prob}
        metrics_summary[name] = compute_metrics(y_test, y_pred, y_prob)
        
        if name == "Random Forest":
            rf_importances = model.feature_importances_
            
    plot_roc_curves(test_predictions, y_test, os.path.join(rf_results_dir, "roc_curves.png"), "Classical Trees ROC")
    plot_precision_recall_curves(test_predictions, y_test, os.path.join(rf_results_dir, "precision_recall_curves.png"), "Classical Trees PR")
    plot_confusion_matrices(test_predictions, y_test, os.path.join(rf_results_dir, "confusion_matrices.png"))
    if rf_importances is not None:
        plot_feature_importance(feature_names, rf_importances, os.path.join(rf_results_dir, "feature_importance.png"))
        
    with open(os.path.join(rf_results_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    # =========================================================
    # PART 2: VAE + Classical Models
    # =========================================================
    print("\n=== PART 2: VAE Compression + Trees ===")
    input_dim = X_train.shape[1]
    latent_dim = 8
    
    vae = TabularVAE(in_features=input_dim, latent_dim=latent_dim)
    trainer = VAETrainer(vae, lr=1e-3, beta=0.005)
    
    print("Training VAE on train set...")
    history = trainer.fit(X_train, epochs=150, batch_size=64)
    
    plot_vae_loss_curves(history, os.path.join(vae_results_dir, "vae_training_loss.png"))
    
    # Encode to latent space
    vae.eval()
    with torch.no_grad():
        Z_train = vae.get_latent(torch.tensor(X_train, dtype=torch.float32), deterministic=True)
        Z_test = vae.get_latent(torch.tensor(X_test, dtype=torch.float32), deterministic=True)
        
        X_train_recon = vae.decode(Z_train).numpy()
        Z_train = Z_train.numpy()
        Z_test = Z_test.numpy()
        
    plot_latent_space_tsne(Z_train, y_train, os.path.join(vae_results_dir, "latent_space_tsne_pca.png"))
    
    # Feature Reconstruction Fidelity
    r2_scores = [r2_score(X_train[:, i], X_train_recon[:, i]) for i in range(input_dim)]
    r2_df = pd.DataFrame({"feature": feature_names, "r2_score": r2_scores})
    plot_reconstruction_r2(r2_df, os.path.join(vae_results_dir, "feature_reconstruction_fidelity.png"))
    r2_df.to_csv(os.path.join(vae_results_dir, "vae_reconstruction_r2.csv"), index=False)
    
    # Train Downstream Models on Z
    models_vae = build_models()
    test_predictions_vae = {}
    metrics_summary_vae = {}
    
    for name, model in models_vae.items():
        print(f"Training {name} on VAE latent space...")
        model.fit(Z_train, y_train)
        
        y_pred = model.predict(Z_test)
        y_prob = model.predict_proba(Z_test)[:, 1]
        
        test_predictions_vae[name] = {"pred": y_pred, "prob": y_prob}
        metrics_summary_vae[name] = compute_metrics(y_test, y_pred, y_prob)
        
    plot_roc_curves(test_predictions_vae, y_test, os.path.join(vae_results_dir, "roc_curves.png"), "VAE Trees ROC")
    plot_precision_recall_curves(test_predictions_vae, y_test, os.path.join(vae_results_dir, "precision_recall_curves.png"), "VAE Trees PR")
    plot_confusion_matrices(test_predictions_vae, y_test, os.path.join(vae_results_dir, "confusion_matrices.png"))
    
    with open(os.path.join(vae_results_dir, "vae_ensemble_metrics_summary.json"), "w") as f:
        json.dump(metrics_summary_vae, f, indent=4)
        
    print("\n[Complete] Comparative study finished successfully!")

if __name__ == "__main__":
    main()
