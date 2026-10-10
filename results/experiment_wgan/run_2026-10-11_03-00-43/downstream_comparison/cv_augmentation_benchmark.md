# Downstream Augmentation Cross-Validation Benchmark

Protocol: 2-fold x 1-repeats. Paired patient bootstrap over 1,000 resamples.

| Condition        |   ROC-AUC |   Accuracy |   Sensitivity |   Specificity |   F1-Score |   Delta AUC vs Baseline |   95% CI Lower |   95% CI Upper | Significance    |
|:-----------------|----------:|-----------:|--------------:|--------------:|-----------:|------------------------:|---------------:|---------------:|:----------------|
| A_real_only      |    0.8664 |     0.7866 |        0.8129 |        0.751  |     0.8141 |                  0      |         0      |         0      | Not Significant |
| B_wgan_0.5x      |    0.8481 |     0.7672 |        0.773  |        0.7593 |     0.7925 |                 -0.0184 |        -0.0335 |        -0.0028 | Significant     |
| B_wgan_1.0x      |    0.8384 |     0.7549 |        0.7515 |        0.7593 |     0.779  |                 -0.0274 |        -0.045  |        -0.0094 | Significant     |
| B_wgan_2.0x      |    0.8431 |     0.7619 |        0.7607 |        0.7635 |     0.7861 |                 -0.0233 |        -0.0464 |         0.0013 | Not Significant |
| C_smote_0.5x     |    0.8692 |     0.7919 |        0.8129 |        0.7635 |     0.8179 |                  0.0028 |        -0.0075 |         0.013  | Not Significant |
| C_smote_1.0x     |    0.8647 |     0.776  |        0.7883 |        0.7593 |     0.8019 |                 -0.0013 |        -0.0123 |         0.0092 | Not Significant |
| C_smote_2.0x     |    0.8697 |     0.7884 |        0.8067 |        0.7635 |     0.8142 |                  0.0034 |        -0.0089 |         0.0151 | Not Significant |
| D_vae_decoder_1x |    0.8653 |     0.7866 |        0.8252 |        0.7344 |     0.8164 |                 -0.0015 |        -0.014  |         0.01   | Not Significant |
