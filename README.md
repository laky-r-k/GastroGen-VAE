# Gastric Cancer Early Diagnostic Screening Pipeline

A modular machine learning and generative modeling framework for clinical diagnostic screening of gastric cancer using biochemical lab biomarkers.

---

## 🔬 Project Overview

This repository develops, benchmarks, and validates statistical, deep generative, and semi-supervised machine learning architectures for non-invasive gastric cancer detection.

### Dataset Overview
* **Cohort Size:** 709 total clinical patients (398 gastric cancer positive, 311 negative controls).
* **Biomarkers:** Complete blood counts (CBC), liver function enzymes, renal function markers, lipid profiles, and serum tumor markers (`CEA`, `CA19-9`, `CA125`, `CA724`).
* **Feature Engineering:** Log-transforms on heavy-tailed tumor markers and clinical prognostic ratios (`NLR`, `PLR`, `Hb-to-RDW`, `Atherogenic Index`, `Albumin-to-Bilirubin`).

---

## 🏛️ Architecture & Modules

```text
major_project/
├── dataset/                    # Clinical cohorts & synthetic data
│   ├── gastric_cancer_train.csv
│   ├── gastric_cancer_test.csv
│   ├── processed_dataset/      # Preprocessed train/test sets
│   └── synthetic_dataset/      # Generated WGAN-GP synthetic cohort
├── model_src/                  # Centralized model implementations
│   ├── tree/                   # Random Forest, AdaBoost, Gradient Boosting
│   ├── VAE/                    # Standard VAE & Semi-Supervised VAE
│   ├── GAN/                    # Conditional WGAN-GP (Generator & Critic)
│   └── vae_pipeline.py         # Sklearn-compatible VAE classifier wrapper
├── evaluation/                 # Modular evaluation engine
│   ├── metrics.py              # Pure statistical and mathematical metrics
│   ├── visualization.py        # ROC, PR, CM, and manifold projections
│   ├── reporting.py            # JSON/CSV result serializers
│   ├── classification_evaluation.py
│   ├── VAE_evalution.py
│   └── synthetic_fidelity.py   # KS-tests, Frobenius distance, TSTR, memorization
├── experiments/                # Config-driven experiment runner
│   ├── base_experiment.py      # Abstract base class for reproducible runs
│   ├── configs/                # YAML configuration files
│   ├── experiment1.py          # Baseline Trees vs. VAE + Trees
│   ├── experiment2.py          # WGAN-GP Augmented Multi-Seed & Multi-Beta Study
│   └── experiment3.py          # Semi-Supervised VAE vs. Baseline Models
├── results/                    # Versioned, timestamped experimental runs
├── run.py                      # Unified CLI entry point
└── resultofss_vaetest.md       # Summary of Semi-Supervised VAE findings
```

---

## 🚀 Running Experiments

All experiments are config-driven and executed via the unified CLI runner:

```bash
# 1. View all available registered experiments
python3 run.py --list

# 2. Run Experiment 1 (Baseline GB vs. VAE + GB)
python3 run.py --exp experiment1

# 3. Run Experiment 2 (WGAN-GP Augmented Multi-Seed & Multi-Beta Benchmark)
python3 run.py --exp experiment2

# 4. Run Experiment 3 (Semi-Supervised VAE vs. Gradient Boosting)
python3 run.py --exp experiment3
```

Results, metrics summaries, and plots are automatically saved in versioned folders under `results/<experiment>/run_<timestamp>/`.

---

## 📊 Summary of Research Findings

### 1. Semi-Supervised VAE Breakthrough (`experiment3`)
Standard unsupervised VAE compression created an information bottleneck ($\text{ROC-AUC} \approx 0.72$). Introducing multi-task joint supervision ($\mathcal{L}_{\text{Recon}} + \beta\mathcal{L}_{\text{KL}} + \alpha\mathcal{L}_{\text{BCE}}$) aligned the 8-dimensional latent space with the cancer diagnostic manifold, yielding a **+11.5% ROC-AUC leap**:

| Model Condition | ROC-AUC | PR-AUC | Accuracy | Specificity |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline GB (Raw Features)** | **0.8569 ± 0.0033** | **0.8839 ± 0.0019** | 0.7507 ± 0.0080 | 0.7429 ± 0.0143 |
| **Standard VAE + GB (Unsupervised Z)** | 0.7196 ± 0.0307 | 0.7246 ± 0.0316 | 0.6775 ± 0.0390 | 0.7343 ± 0.0260 |
| **SS-VAE + GB (Supervised Z)** | **0.8349 ± 0.0097** | 0.8300 ± 0.0146 | **0.7549 ± 0.0104** | 0.7571 ± 0.0101 |
| **SS-VAE End-to-End Neural Net** | **0.8330 ± 0.0111** | **0.8459 ± 0.0151** | 0.7479 ± 0.0031 | **0.7886 ± 0.0538** |

*For complete details, see [`resultofss_vaetest.md`](resultofss_vaetest.md).*
