# Experiment 3 Results: Semi-Supervised VAE vs. Baseline Models

**Date of Execution:** 2026-10-25  
**Git Branch:** `feature/semi-supervised-vae`  
**Run Directory:** `results/experiment3/run_2026-10-25_21-38-23/`  
**Training Cohort:** Augmented Pool (Real 567 + WGAN-GP Synthetic 566 = 1,133 patients)  
**Test Cohort:** 142 held-out real clinical patients (`processed_test.csv`)  
**Seeds Evaluated:** `[42, 123, 456, 789, 2024]` (5 independent seeds per condition)

---

## 1. Executive Summary & Findings

### Core Finding: Semi-Supervised VAE Solved the Latent Bottleneck
1. **Massive Performance Leap over Standard VAE (+11.5% ROC-AUC):**
   * Standard Unsupervised VAE + Gradient Boosting only reached **$\text{ROC-AUC} = 0.7196 \pm 0.0307$**.
   * **SS-VAE + Gradient Boosting** surged to **$\text{ROC-AUC} = 0.8349 \pm 0.0097$** (+11.53% gain).
   * **SS-VAE End-to-End Neural Net** surged to **$\text{ROC-AUC} = 0.8330 \pm 0.0111$** (+11.34% gain).

2. **Both Neural Classifier and Gradient Boosting Excel on Cancer-Guided Latent Space:**
   * **SS-VAE + GB:** $\text{ROC-AUC} = \mathbf{0.8349}$, $\text{Accuracy} = \mathbf{75.49\%}$, $\text{F1} = \mathbf{0.7569}$.
   * **SS-VAE Neural Net:** $\text{ROC-AUC} = \mathbf{0.8330}$, $\text{Accuracy} = \mathbf{74.79\%}$, $\text{Specificity} = \mathbf{78.86\%}$.
   * The dual-headed multi-task loss ($\mathcal{L}_{\text{Recon}} + \beta\mathcal{L}_{\text{KL}} + \alpha\mathcal{L}_{\text{BCE}}$ with $\alpha=10.0$) successfully forced the encoder to align the 8-dimensional latent space with the cancer classification boundary.

3. **Comparison Against Standalone Gradient Boosting:**
   * **Baseline GB on 24 Raw Features:** $\text{ROC-AUC} = 0.8569 \pm 0.0033$.
   * SS-VAE closes the gap from a 14% deficit (Unsupervised VAE) down to **within 2% of full raw-feature trees**, while compressing the patient into just **8 latent dimensions**.

---

## 2. Multi-Seed Benchmark Summary Table (Mean ± Std Across 5 Seeds)

| Architectural Condition | ROC-AUC | PR-AUC | Accuracy | F1-Score | Sensitivity | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline GB (Raw Features)** | **0.8569 ± 0.0033** | **0.8839 ± 0.0019** | 0.7507 ± 0.0080 | 0.7552 ± 0.0070 | **0.7583 ± 0.0076** | 0.7429 ± 0.0143 |
| **Standard VAE + GB (Unsupervised Z)** | 0.7196 ± 0.0307 | 0.7246 ± 0.0316 | 0.6775 ± 0.0390 | 0.6612 ± 0.0459 | 0.6222 ± 0.0523 | 0.7343 ± 0.0260 |
| **SS-VAE + GB (Supervised Z)** | **0.8349 ± 0.0097** | 0.8300 ± 0.0146 | **0.7549 ± 0.0104** | **0.7569 ± 0.0126** | 0.7528 ± 0.0206 | 0.7571 ± 0.0101 |
| **SS-VAE End-to-End Neural Net** | **0.8330 ± 0.0111** | **0.8459 ± 0.0151** | 0.7479 ± 0.0031 | 0.7396 ± 0.0140 | 0.7083 ± 0.0520 | **0.7886 ± 0.0538** |

---

## 3. Detailed Per-Seed Performance Breakdown

| Seed | Condition | ROC-AUC | PR-AUC | Accuracy | F1-Score | Sensitivity | Specificity |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | Baseline GB (Raw Features) | 0.8565 | 0.8844 | 0.7606 | 0.7660 | 0.7639 | 0.7571 |
| **42** | Standard VAE + GB (Unsupervised Z) | 0.7514 | 0.7676 | 0.7113 | 0.7077 | 0.6389 | 0.7857 |
| **42** | SS-VAE + GB (Supervised Z) | 0.8466 | 0.8427 | 0.7606 | 0.7660 | 0.7639 | 0.7571 |
| **42** | SS-VAE End-to-End Neural Net | 0.8450 | 0.8653 | 0.7465 | 0.7517 | 0.7500 | 0.7429 |
| **123** | Baseline GB (Raw Features) | 0.8625 | 0.8864 | 0.7465 | 0.7483 | 0.7361 | 0.7571 |
| **123** | Standard VAE + GB (Unsupervised Z) | 0.7528 | 0.7441 | 0.7113 | 0.7101 | 0.6806 | 0.7429 |
| **123** | SS-VAE + GB (Supervised Z) | 0.8276 | 0.8373 | 0.7535 | 0.7517 | 0.7361 | 0.7714 |
| **123** | SS-VAE End-to-End Neural Net | 0.8315 | 0.8488 | 0.7465 | 0.7333 | 0.6806 | 0.8143 |
| **456** | Baseline GB (Raw Features) | 0.8565 | 0.8841 | 0.7535 | 0.7586 | 0.7639 | 0.7429 |
| **456** | Standard VAE + GB (Unsupervised Z) | 0.7101 | 0.7225 | 0.6690 | 0.6667 | 0.6528 | 0.6857 |
| **456** | SS-VAE + GB (Supervised Z) | 0.8238 | 0.8062 | 0.7465 | 0.7448 | 0.7500 | 0.7429 |
| **456** | SS-VAE End-to-End Neural Net | 0.8427 | 0.8443 | 0.7535 | 0.7534 | 0.7639 | 0.7429 |
| **789** | Baseline GB (Raw Features) | 0.8546 | 0.8814 | 0.7465 | 0.7517 | 0.7639 | 0.7286 |
| **789** | Standard VAE + GB (Unsupervised Z) | 0.6889 | 0.6780 | 0.6127 | 0.5846 | 0.5278 | 0.7000 |
| **789** | SS-VAE + GB (Supervised Z) | 0.8427 | 0.8358 | 0.7465 | 0.7483 | 0.7361 | 0.7571 |
| **789** | SS-VAE End-to-End Neural Net | 0.8274 | 0.8407 | 0.7465 | 0.7429 | 0.7222 | 0.7714 |
| **2024** | Baseline GB (Raw Features) | 0.8546 | 0.8833 | 0.7465 | 0.7517 | 0.7639 | 0.7286 |
| **2024** | Standard VAE + GB (Unsupervised Z) | 0.6946 | 0.7107 | 0.6831 | 0.6364 | 0.5109 | 0.7571 |
| **2024** | SS-VAE + GB (Supervised Z) | 0.8339 | 0.8282 | 0.7676 | 0.7733 | 0.7778 | 0.7571 |
| **2024** | SS-VAE End-to-End Neural Net | 0.8183 | 0.8306 | 0.7465 | 0.7167 | 0.6250 | 0.8714 |

---

## 4. Key Artifacts & Plots
* **Training Dynamics (Joint Multi-Task Losses):**  
  `results/experiment3/run_2026-10-25_21-38-23/diagnostics/ss_vae_training_dynamics.png`
* **Mean ROC Curves (with ±1 Std Shading):**  
  `results/experiment3/run_2026-10-25_21-38-23/plots/ss_vae_mean_roc_curves.png`
* **Condition Boxplots:**  
  `results/experiment3/run_2026-10-25_21-38-23/plots/ss_vae_benchmark_boxplots.png`
* **Detailed Per-Seed CSV:**  
  `results/experiment3/run_2026-10-25_21-38-23/metrics/detailed_per_seed_metrics.csv`
* **Summary Table CSV:**  
  `results/experiment3/run_2026-10-25_21-38-23/metrics/ss_vae_benchmark_summary.csv`
