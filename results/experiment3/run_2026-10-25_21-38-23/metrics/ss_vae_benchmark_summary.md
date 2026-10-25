# Experiment 3: Semi-Supervised VAE vs. Gradient Boosting Benchmark

Trained on Augmented Dataset (1,133 patients). Evaluated on held-out test cohort (142 real patients) across 5 random seeds.

| Architectural Condition            | ROC-AUC         | PR-AUC          | Accuracy        | F1-Score        | Sensitivity     | Specificity     |
|:-----------------------------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|
| Baseline GB (Raw Features)         | 0.8569 ± 0.0033 | 0.8839 ± 0.0019 | 0.7507 ± 0.0080 | 0.7552 ± 0.0070 | 0.7583 ± 0.0076 | 0.7429 ± 0.0143 |
| Standard VAE + GB (Unsupervised Z) | 0.7196 ± 0.0307 | 0.7246 ± 0.0316 | 0.6775 ± 0.0390 | 0.6612 ± 0.0459 | 0.6222 ± 0.0523 | 0.7343 ± 0.0260 |
| SS-VAE + GB (Supervised Z)         | 0.8349 ± 0.0097 | 0.8300 ± 0.0146 | 0.7549 ± 0.0104 | 0.7569 ± 0.0126 | 0.7528 ± 0.0206 | 0.7571 ± 0.0101 |
| SS-VAE End-to-End Neural Net       | 0.8330 ± 0.0111 | 0.8459 ± 0.0151 | 0.7479 ± 0.0031 | 0.7396 ± 0.0140 | 0.7083 ± 0.0520 | 0.7886 ± 0.0538 |
