# Downstream Augmentation Cross-Validation Benchmark

Protocol: 5-fold x 5-repeats. Paired patient bootstrap over 1,000 resamples.

| Condition        |   ROC-AUC |   Accuracy |   Sensitivity |   Specificity |   F1-Score |   Delta AUC vs Baseline |   95% CI Lower |   95% CI Upper | Significance    |
|:-----------------|----------:|-----------:|--------------:|--------------:|-----------:|------------------------:|---------------:|---------------:|:----------------|
| A_real_only      |    0.878  |     0.8095 |        0.8405 |        0.7676 |     0.8354 |                  0      |         0      |         0      | Not Significant |
| B_wgan_0.5x      |    0.8856 |     0.8025 |        0.8006 |        0.805  |     0.8233 |                  0.0078 |        -0.0045 |         0.0207 | Not Significant |
| B_wgan_1.0x      |    0.8851 |     0.8078 |        0.816  |        0.7967 |     0.83   |                  0.0071 |        -0.0061 |         0.0202 | Not Significant |
| B_wgan_2.0x      |    0.8876 |     0.8025 |        0.7945 |        0.8133 |     0.8222 |                  0.0095 |        -0.0048 |         0.0244 | Not Significant |
| C_smote_0.5x     |    0.8749 |     0.7989 |        0.8313 |        0.7552 |     0.8262 |                 -0.0033 |        -0.0133 |         0.0061 | Not Significant |
| C_smote_1.0x     |    0.8733 |     0.7937 |        0.8221 |        0.7552 |     0.8208 |                 -0.0048 |        -0.0157 |         0.0067 | Not Significant |
| C_smote_2.0x     |    0.878  |     0.7937 |        0.8374 |        0.7344 |     0.8235 |                  0      |        -0.0115 |         0.0112 | Not Significant |
| D_vae_decoder_1x |    0.8677 |     0.7989 |        0.8282 |        0.7593 |     0.8257 |                 -0.0103 |        -0.0234 |         0.0014 | Not Significant |
