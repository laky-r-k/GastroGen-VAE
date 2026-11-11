# Experiment 12: corrected paired AUC significance analysis

Analyzed 59 matched stratified 70/30 splits (seeds 42–100).

No models were retrained. Split manifests were checked before the analysis; all comparisons use the same patient IDs in each train/test partition.

## Method

For each comparison, the per-seed difference is AUC(model A) − AUC(model B). We used a two-sided Nadeau–Bengio corrected resampled t-test with the repeated-holdout variance factor 1/n + n_test/n_train = 0.446385, t degrees of freedom 58. The five p-values were adjusted together with Holm's step-down method (family-wise alpha=0.05). The confidence intervals below are corrected 95% intervals and are nominal per comparison, not simultaneous family-wise intervals.

## Results

| Comparison (A − B) | Mean ΔAUC | Corrected 95% CI | Holm-adjusted p | A higher / tied / B higher |
|---|---:|---:|---:|---:|
| TabPFN real-only minus paper-config GB real-only | +0.0240 | [+0.0077, +0.0403] | 0.02306 | 58/0/1 |
| GB + WGAN minus GB real-only | +0.0051 | [-0.0132, +0.0233] | 1 | 37/0/22 |
| VAE + GB + WGAN minus VAE + GB (beta=0.0005) | +0.0017 | [-0.0275, +0.0309] | 1 | 31/0/28 |
| SS-VAE + GB + WGAN minus SS-VAE + GB (beta=0.0005) | +0.0026 | [-0.0183, +0.0235] | 1 | 35/0/24 |
| TabPFN real + WGAN minus TabPFN real-only | -0.0057 | [-0.0155, +0.0040] | 0.9807 | 12/0/47 |

## Interpretation

- **WGAN augmentation:** No statistically reliable evidence of an AUC improvement from WGAN augmentation was found in these four tested model comparisons. This does not prove that the true effect is exactly zero.
- **TabPFN vs. paper-config GB:** TabPFN real-only has significantly higher AUC than paper-config GB after Holm adjustment.

These tests compare repeated splits from one cohort. They do not establish performance on an independent hospital or patient population. A non-significant result is not proof of equivalence or of a zero augmentation effect.
