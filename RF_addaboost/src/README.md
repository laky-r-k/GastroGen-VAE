# RF + AdaBoost Training Pipeline & Implementation Report

**Directory:** `RF_addaboost/src/`  
**Purpose:** End-to-end Machine Learning training, feature engineering, and evaluation pipeline for Gastric Cancer diagnostic classification using **Random Forest**, **AdaBoost**, and **Hybrid Ensembles (Soft Voting & Stacking)**.

---

## 1. Architecture & Codebase Overview

The pipeline is designed with a clean, modular structure:

| File | Purpose / Role |
| :--- | :--- |
| [`data_loader.py`](file:///home/laky/Desktop/proj/major_project/RF_addaboost/src/data_loader.py) | Robust dataset resolver and loader with auto-detection for local directories and Google Colab environments (`/content/...`). |
| [`preprocessing.py`](file:///home/laky/Desktop/proj/major_project/RF_addaboost/src/preprocessing.py) | Physiological anomaly cleaning, log transformations on skewed tumor markers, clinical feature engineering, and feature selection. |
| [`models.py`](file:///home/laky/Desktop/proj/major_project/RF_addaboost/src/models.py) | Model factory for Random Forest, AdaBoost, Soft Voting, Stacking classifiers, and `joblib` serialization helpers. |
| [`evaluate.py`](file:///home/laky/Desktop/proj/major_project/RF_addaboost/src/evaluate.py) | Metric calculation (ROC-AUC, PR-AUC, Accuracy, F1, Recall/Sensitivity, Specificity, Brier Score) and generation of high-resolution plots. |
| [`train.py`](file:///home/laky/Desktop/proj/major_project/RF_addaboost/src/train.py) | Stratified 5-Fold Cross-Validation, Out-Of-Fold (OOF) prediction generation, full dataset retraining, and artifact saving. |
| [`main.py`](file:///home/laky/Desktop/proj/major_project/RF_addaboost/src/main.py) | CLI entry point with configurable arguments for running the full pipeline end-to-end. |

---

## 2. Feature Selection & Preprocessing Rationale (Based on EDA)

Based on our exploratory data analysis of [`gastric_cancer_dataset_raw.csv`](file:///home/laky/Desktop/proj/major_project/dataset/gastric_cancer_dataset_raw.csv), the following data-driven enhancements were implemented:

### A. Anomaly Cleaning & Physiological Corrections
- **Hemoglobin (`hb`):** Corrected decimal typo (`1222.0` $\rightarrow$ `122.2` $\text{g/L}$) for Patient 249.
- **Albumin (`alb`):** Corrected decimal typo (`397.0` $\rightarrow$ `39.7` $\text{g/L}$) for Patient 700.
- **Red Cell Distribution Width (`rdw`):** Clipped transcription error outliers ($>40\%$) to the 99th percentile ($18.5\%$) to prevent AdaBoost exponential loss distortion.

### B. Mitigating Skewness in Tumor Markers
All 4 serum tumor markers and selected biochemical enzymes had extreme right skew ($>10-26$). Applying $\log(1+x)$ transformations normalizes their distributions:
- `log_cea`, `log_ca199`, `log_ca125`, `log_ca724`, `log_lpa`, `log_alt`, `log_tb`, `log_tg`, `log_neu_lym`.

### C. Clinical Feature Engineering
We engineered 6 domain-informed biomarkers:
1. **`nlr_computed`:** Re-derived $\text{Neutrophil} / \text{Lymphocyte}$ ratio (systemic inflammation index).
2. **`hb_rdw_ratio`:** $\text{Hemoglobin} / \text{RDW}$ (established nutritional and anemia prognostic indicator).
3. **`plt_lym_ratio`:** Platelet-to-Lymphocyte Ratio (PLR).
4. **`tumor_marker_composite`:** Sum of log-transformed tumor markers ($\log(\text{CEA}) + \log(\text{CA199}) + \log(\text{CA724}) + \log(\text{CA125})$) representing cumulative oncological burden.
5. **`atherogenic_index`:** $\text{Cholesterol} / \text{HDL}$ ratio (metabolic dysregulation proxy).
6. **`alb_tb_ratio`:** $\text{Albumin} / \text{Total Bilirubin}$ ratio (hepatic protein synthesis proxy).

---

## 3. Model Architectures & Ensembling Strategy

1. **Random Forest Classifier:**
   - Bagging ensemble with 250 estimators, `max_depth=6`, `min_samples_split=4`, `min_samples_leaf=2`, `class_weight='balanced'`.
2. **AdaBoost Classifier:**
   - Boosting ensemble using 150 estimators with shallow decision trees (`max_depth=2`, `learning_rate=0.05`) to prevent overfitting on clinical noise.
3. **Soft Voting Hybrid Ensemble:**
   - Blends calibrated probability estimates from Random Forest and AdaBoost ($\hat{P}(y=1) = \frac{P_{\text{RF}} + P_{\text{Ada}}}{2}$), reducing variance and improving decision boundaries.
4. **Stacking Classifier:**
   - Uses RF and AdaBoost as Level-0 base learners and a regularized Logistic Regression model as the Level-1 meta-learner.

---

## 4. Execution & Usage

### Running Locally:
```bash
# Execute from project root
python3 RF_addaboost/src/main.py

# Or with custom options
python3 RF_addaboost/src/main.py --feature_mode optimized_subset --n_splits 5
```

### Running in Google Colab:
```python
# In a Google Colab notebook cell:
!python RF_addaboost/src/main.py --data_path /content/major_project/dataset/gastric_cancer_dataset_raw.csv
```

---

## 5. Generated Artifacts & Outputs

### Models Directory (`RF_addaboost/model/`)
- `random_forest_model.joblib`: Serialized Random Forest model.
- `adaboost_model.joblib`: Serialized AdaBoost model.
- `hybrid_soft_voting_model.joblib`: Serialized Soft Voting ensemble model.
- `hybrid_stacking_model.joblib`: Serialized Stacking ensemble model.
- `preprocessing_pipeline.joblib`: Complete fitted preprocessing & feature engineering pipeline.

### Results Directory (`results/RF_addaboost/`)
- `metrics_summary.json`: Full Out-of-Fold performance metrics across all models.
- `cv_fold_scores.csv`: Fold-by-fold cross-validation metrics.
- `feature_importance_ranking.csv`: Ranked clinical biomarker importance list.
- `roc_curves.png`: Multi-model ROC curves with AUC annotations.
- `precision_recall_curves.png`: Precision-Recall curves comparing models against baseline.
- `confusion_matrices.png`: Side-by-side normalized confusion matrices.
- `feature_importance.png`: Top-20 most discriminative biomarker bar chart.
