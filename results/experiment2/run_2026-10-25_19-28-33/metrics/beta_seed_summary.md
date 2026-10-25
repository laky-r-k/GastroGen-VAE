# Experiment 2: Augmented VAE + Gradient Boosting (Multi-Seed & Multi-Beta)

Evaluated across 5 random seeds on held-out test cohort (142 real patients).

| Condition                                  | Data       | Beta   | ROC-AUC         | PR-AUC          | Accuracy        | F1-Score        |
|:-------------------------------------------|:-----------|:-------|:----------------|:----------------|:----------------|:----------------|
| Baseline GB (Real)                         | Real       | N/A    | 0.8577 ± 0.0094 | 0.8871 ± 0.0093 | 0.7662 ± 0.0104 | 0.7733 ± 0.0088 |
| Augmented GB (Real + Syn)                  | Real + Syn | N/A    | 0.8569 ± 0.0033 | 0.8839 ± 0.0019 | 0.7507 ± 0.0080 | 0.7552 ± 0.0070 |
| VAE + GB (Real, beta=0.0005)               | Real       | 0.0005 | 0.7307 ± 0.0298 | 0.7524 ± 0.0188 | 0.6507 ± 0.0484 | 0.6561 ± 0.0420 |
| Augmented VAE + GB (Real+Syn, beta=0.0005) | Real + Syn | 0.0005 | 0.7443 ± 0.0226 | 0.7372 ± 0.0181 | 0.6986 ± 0.0219 | 0.6849 ± 0.0272 |
| VAE + GB (Real, beta=0.005)                | Real       | 0.005  | 0.7403 ± 0.0197 | 0.7376 ± 0.0219 | 0.6803 ± 0.0162 | 0.6868 ± 0.0141 |
| Augmented VAE + GB (Real+Syn, beta=0.005)  | Real + Syn | 0.005  | 0.7214 ± 0.0203 | 0.7228 ± 0.0151 | 0.6493 ± 0.0168 | 0.6232 ± 0.0201 |
| VAE + GB (Real, beta=0.05)                 | Real       | 0.05   | 0.7215 ± 0.0269 | 0.7244 ± 0.0257 | 0.6662 ± 0.0313 | 0.6669 ± 0.0383 |
| Augmented VAE + GB (Real+Syn, beta=0.05)   | Real + Syn | 0.05   | 0.7362 ± 0.0250 | 0.7387 ± 0.0226 | 0.6676 ± 0.0466 | 0.6532 ± 0.0607 |
| VAE + GB (Real, beta=0.5)                  | Real       | 0.5    | 0.7215 ± 0.0363 | 0.7347 ± 0.0420 | 0.6676 ± 0.0219 | 0.6786 ± 0.0140 |
| Augmented VAE + GB (Real+Syn, beta=0.5)    | Real + Syn | 0.5    | 0.7457 ± 0.0206 | 0.7498 ± 0.0114 | 0.6915 ± 0.0270 | 0.6740 ± 0.0346 |
