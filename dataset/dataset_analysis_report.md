# Gastric Cancer Dataset Analysis & Exploratory Data Report

**File Path:** `dataset/gastric_cancer_dataset_raw.csv`  
**Dataset Domain:** Oncology / Clinical Pathology & Biochemical Diagnostic Screening  
**Total Samples:** 709 patients | **Features:** 24 attributes (22 clinical/biochemical features, 1 target label, 1 ID)

---

## 1. Executive Summary

This report presents an exhaustive exploratory data analysis (EDA) of the raw gastric cancer diagnostic dataset. The dataset comprises **709 patient records** with **24 attributes**.

### Key Highlights
- **Completeness:** 100% complete with **0 missing values** and **0 duplicate records**.
- **Class Balance:** 398 Gastric Cancer Positive cases (56.14%) vs. 311 Control/Negative cases (43.86%).
- **Top Discriminative Predictors:**
  - **Age** ($p = 5.50 \times 10^{-38}$, $\text{AUC} = 0.774$)
  - **Albumin (`alb`)** ($p = 5.91 \times 10^{-36}$, $\text{AUC} = 0.774$)
  - **HDL Cholesterol (`hdl`)** ($p = 8.20 \times 10^{-18}$, $\text{AUC} = 0.688$)
  - **Carcinoembryonic Antigen (`cea`)** ($p = 6.60 \times 10^{-17}$, $\text{AUC} = 0.683$)
  - **Hematocrit (`hct`)** ($p = 1.13 \times 10^{-16}$, $\text{AUC} = 0.681$)
  - **Neutrophil-to-Lymphocyte Ratio (`neu_lym`)** ($p = 3.28 \times 10^{-16}$, $\text{AUC} = 0.678$)
  - **Hemoglobin (`hb`)** ($p = 4.68 \times 10^{-15}$, $\text{AUC} = 0.671$)
  - **Gender** ($p = 1.53 \times 10^{-15}$, $\text{AUC} = 0.651$)
- **Data Quality Anomalies Identified:** 5 severe data entry / typographical errors discovered in raw features (`hb = 1222.0`, `alb = 397.0`, `rdw = 118.0`, `87.7`, `50.5`) requiring cleaning prior to model training.
- **Distribution Characteristics:** Heavy right-skewness observed across all 4 serum tumor markers (`cea`, `ca125`, `ca199`, `ca724`) consistent with advanced-stage oncological dynamics.

---

## 2. Dataset Schema & Feature Taxonomy

| Column Name | Category | Data Type | Units / Range | Clinical Description |
| :--- | :--- | :--- | :--- | :--- |
| `patient_id` | Metadata | Integer | 1 – 709 | Unique patient serial identifier |
| `label` | Target | Binary (0, 1) | 0: Control, 1: Cancer | Gastric Cancer diagnosis status |
| `gender` | Demographic | Binary (0, 1) | 0: Female, 1: Male | Biological sex |
| `age` | Demographic | Ordinal / Code | 2 – 9 (Decades) | Patient age group bracket |
| `neu` | CBC / Hematology | Float | $10^9/\text{L}$ | Absolute Neutrophil Count |
| `lym` | CBC / Hematology | Float | $10^9/\text{L}$ | Absolute Lymphocyte Count |
| `neu_lym` | Inflammatory Marker | Float | Ratio | Neutrophil-to-Lymphocyte Ratio (NLR) |
| `hb` | CBC / Hematology | Float | $\text{g/L}$ | Hemoglobin concentration |
| `hct` | CBC / Hematology | Float | Proportion / % | Hematocrit (Packed cell volume) |
| `rdw` | CBC / Hematology | Float | % | Red Blood Cell Distribution Width |
| `plt` | CBC / Hematology | Float | $10^9/\text{L}$ | Platelet count |
| `alb` | Liver / Protein | Float | $\text{g/L}$ | Serum Albumin concentration |
| `alt` | Liver Function | Float | $\text{U/L}$ | Alanine Aminotransferase enzyme |
| `tb` | Liver Function | Float | $\mu\text{mol/L}$ | Total Bilirubin concentration |
| `cr` | Renal Function | Float | $\mu\text{mol/L}$ | Serum Creatinine concentration |
| `tg` | Lipid Profile | Float | $\text{mmol/L}$ | Serum Triglycerides |
| `chol` | Lipid Profile | Float | $\text{mmol/L}$ | Total Serum Cholesterol |
| `hdl` | Lipid Profile | Float | $\text{mmol/L}$ | High-Density Lipoprotein Cholesterol |
| `ldl` | Lipid Profile | Float | $\text{mmol/L}$ | Low-Density Lipoprotein Cholesterol |
| `lpa` | Lipid Profile | Float | $\text{mg/L}$ | Lipoprotein(a) |
| `cea` | Tumor Marker | Float | $\text{ng/mL}$ | Carcinoembryonic Antigen |
| `ca125` | Tumor Marker | Float | $\text{U/mL}$ | Cancer Antigen 125 |
| `ca199` | Tumor Marker | Float | $\text{U/mL}$ | Carbohydrate Antigen 19-9 |
| `ca724` | Tumor Marker | Float | $\text{U/mL}$ | Carbohydrate Antigen 72-4 |

---

## 3. Univariate & Bivariate Statistical Profile

Below is the comparative statistical breakdown between **Control ($N=311$)** and **Gastric Cancer ($N=398$)** cohorts:

| Feature | Overall Mean ± SD | Control (Mean ± SD) | Cancer (Mean ± SD) | Control Median [IQR] | Cancer Median [IQR] | MWU $p$-value | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **gender** | $0.50 \pm 0.50$ | $0.33 \pm 0.47$ | $0.63 \pm 0.48$ | 0.00 [0.00 - 1.00] | 1.00 [0.00 - 1.00] | $1.53 \times 10^{-15}$ | 0.651 |
| **age** | $6.03 \pm 1.31$ | $5.32 \pm 1.23$ | $6.58 \pm 1.09$ | 5.00 [5.00 - 6.00] | 7.00 [6.00 - 7.00] | $5.50 \times 10^{-38}$ | **0.774** |
| **neu** | $3.64 \pm 2.19$ | $3.17 \pm 0.94$ | $4.01 \pm 2.75$ | 3.00 [2.50 - 3.70] | 3.50 [2.80 - 4.70] | $2.98 \times 10^{-9}$ | 0.630 |
| **lym** | $1.66 \pm 0.61$ | $1.80 \pm 0.62$ | $1.55 \pm 0.58$ | 1.80 [1.40 - 2.15] | 1.50 [1.10 - 2.00] | $6.09 \times 10^{-8}$ | 0.618 |
| **neu_lym** | $2.59 \pm 2.19$ | $1.97 \pm 1.05$ | $3.08 \pm 2.67$ | 1.71 [1.38 - 2.30] | 2.34 [1.70 - 3.57] | $3.28 \times 10^{-16}$ | **0.678** |
| **hb** | $130.39 \pm 47.89$ | $141.31 \pm 64.21$ | $121.85 \pm 26.54$ | 137.00 [128.00 - 150.00] | 127.00 [106.00 - 141.00] | $4.68 \times 10^{-15}$ | **0.671** |
| **hct** | $0.39 \pm 0.07$ | $0.41 \pm 0.05$ | $0.37 \pm 0.07$ | 0.41 [0.39 - 0.44] | 0.38 [0.33 - 0.42] | $1.13 \times 10^{-16}$ | **0.681** |
| **rdw** | $13.49 \pm 5.33$ | $12.93 \pm 4.38$ | $13.93 \pm 5.93$ | 12.50 [12.10 - 13.00] | 13.00 [12.30 - 14.07] | $4.51 \times 10^{-12}$ | 0.651 |
| **plt** | $225.68 \pm 73.12$ | $218.84 \pm 57.62$ | $231.02 \pm 82.94$ | 214.00 [181.00 - 254.00] | 220.00 [176.00 - 269.00] | $2.14 \times 10^{-1}$ | 0.527 |
| **alb** | $40.30 \pm 14.69$ | $42.85 \pm 4.14$ | $38.31 \pm 19.03$ | 42.90 [40.30 - 45.50] | 38.20 [34.30 - 41.88] | $5.91 \times 10^{-36}$ | **0.774** |
| **alt** | $19.32 \pm 16.22$ | $20.52 \pm 13.18$ | $18.39 \pm 18.21$ | 17.00 [12.50 - 24.00] | 14.00 [11.00 - 20.00] | $1.03 \times 10^{-6}$ | 0.607 |
| **tb** | $13.93 \pm 13.43$ | $13.46 \pm 5.12$ | $14.30 \pm 17.35$ | 12.50 [9.85 - 16.05] | 11.60 [9.03 - 15.50] | $3.11 \times 10^{-2}$ | 0.547 |
| **cr** | $76.40 \pm 25.10$ | $71.67 \pm 12.23$ | $80.10 \pm 31.23$ | 69.30 [62.75 - 78.60] | 76.90 [66.43 - 88.00] | $7.91 \times 10^{-9}$ | 0.626 |
| **tg** | $1.42 \pm 1.12$ | $1.58 \pm 1.48$ | $1.30 \pm 0.70$ | 1.21 [0.92 - 1.77] | 1.14 [0.88 - 1.54] | $1.70 \times 10^{-2}$ | 0.552 |
| **chol** | $4.65 \pm 1.09$ | $4.93 \pm 0.98$ | $4.43 \pm 1.13$ | 4.84 [4.24 - 5.54] | 4.36 [3.61 - 5.17] | $3.42 \times 10^{-10}$ | 0.637 |
| **hdl** | $1.20 \pm 0.35$ | $1.33 \pm 0.35$ | $1.10 \pm 0.31$ | 1.29 [1.10 - 1.50] | 1.10 [0.89 - 1.28] | $8.20 \times 10^{-18}$ | **0.688** |
| **ldl** | $2.72 \pm 0.83$ | $2.82 \pm 0.76$ | $2.63 \pm 0.86$ | 2.78 [2.25 - 3.29] | 2.58 [2.02 - 3.09] | $5.74 \times 10^{-4}$ | 0.575 |
| **lpa** | $213.07 \pm 232.98$ | $196.12 \pm 231.85$ | $226.31 \pm 233.30$ | 105.00 [55.00 - 243.00] | 153.50 [80.50 - 298.00] | $4.85 \times 10^{-4}$ | 0.576 |
| **cea** | $37.03 \pm 723.59$ | $1.99 \pm 1.51$ | $64.41 \pm 965.41$ | 1.60 [1.10 - 2.50] | 2.40 [1.50 - 4.50] | $6.60 \times 10^{-17}$ | **0.683** |
| **ca125** | $30.50 \pm 106.60$ | $12.65 \pm 8.56$ | $44.45 \pm 140.59$ | 11.20 [8.15 - 14.75] | 12.45 [8.62 - 21.09] | $1.87 \times 10^{-5}$ | 0.594 |
| **ca199** | $91.54 \pm 995.34$ | $12.78 \pm 16.51$ | $153.09 \pm 1325.87$ | 9.65 [6.70 - 14.35] | 11.05 [6.10 - 25.93] | $8.99 \times 10^{-3}$ | 0.557 |
| **ca724** | $10.79 \pm 42.29$ | $6.12 \pm 14.51$ | $14.45 \pm 54.73$ | 1.95 [1.10 - 4.75] | 2.40 [1.21 - 7.17] | $4.39 \times 10^{-3}$ | 0.562 |

---

## 4. Clinical Insights & Physiological Interpretations

### A. Demographics (`age`, `gender`)
- **Age:** Highly skewed toward older individuals in the cancer cohort (Cancer Median: 7 [approx. 70s] vs. Control Median: 5 [approx. 50s]). Gastric carcinoma risk escalates significantly with advancing age.
- **Gender:** Male patients account for **63.32% (252/398)** of the cancer group, compared to **33.12% (103/311)** of the control group ($p < 10^{-14}$), aligning with global epidemiological data showing ~2:1 male-to-female gastric cancer incidence ratio.

### B. Hematology & Systemic Inflammation (`neu`, `lym`, `neu_lym`, `hb`, `hct`, `rdw`)
- **Neutrophil-to-Lymphocyte Ratio (`neu_lym`):** Elevated in the cancer group (Cancer Median: 2.34 vs. Control: 1.71). NLR is a well-established prognostic indicator of systemic inflammatory response and immune suppression induced by tumor microenvironments.
- **Anemia Indicators (`hb`, `hct`):** Both Hemoglobin and Hematocrit are significantly reduced in cancer patients (Cancer Median Hb: 127 g/L vs Control: 137 g/L; Hct: 0.38 vs 0.41), reflecting chronic occult gastrointestinal bleeding, bone marrow suppression, and cancer-related anemia.
- **Anisocytosis (`rdw`):** Red Cell Distribution Width is significantly elevated in cancer patients ($p = 4.51 \times 10^{-12}$), reflecting impaired erythropoiesis and nutritional deficiency.

### C. Nutrition, Liver & Lipid Biomarkers (`alb`, `hdl`, `chol`)
- **Hypoalbuminemia (`alb`):** Median albumin is significantly lower in cancer patients (38.2 g/L vs. 42.9 g/L, $\text{AUC} = 0.774$). Tumor cachexia, chronic inflammation, and poor nutritional intake reduce hepatic albumin synthesis.
- **Lipid Alteration (`hdl`, `chol`):** HDL cholesterol is markedly lower in cancer patients ($1.10\ \text{mmol/L}$ vs $1.29\ \text{mmol/L}$), reflecting cancer-induced dyslipidemia and systemic metabolic reprogramming.

### D. Serum Tumor Markers (`cea`, `ca125`, `ca199`, `ca724`)
- **`cea` (Carcinoembryonic Antigen):** The single strongest tumor marker with $\text{AUC} = 0.683$, exhibiting dramatic spikes up to 19,193 ng/mL in cancer patients.
- **`ca125`, `ca199`, `ca724`:** All show statistically significant elevations in the cancer group with long right tails, typical of gastrointestinal malignancies and peritoneal metastasis.

---

## 5. Data Quality Audit & Anomalies

An automated inspection identified **5 clear typographical/data-entry errors** that must be addressed:

| Patient ID | Feature | Raw Value | Reference / Expected Range | Probable Cause & Recommended Action |
| :--- | :--- | :--- | :--- | :--- |
| **249** | `hb` | `1222.0` | $120 - 170\ \text{g/L}$ | Misplaced decimal point. Likely `122.2` (patient has `hct=0.368`, `rdw=12.7`). Replace with `122.2` or median. |
| **700** | `alb` | `397.0` | $35 - 52\ \text{g/L}$ | Misplaced decimal point. Likely `39.7` (patient has `hb=130.0`, `hct=0.40`). Replace with `39.7` or median. |
| **653** | `rdw` | `118.0` | $11.5 - 15.5\ \%$ | Transcription error. Likely `11.8` or clip at boundary. |
| **210** | `rdw` | `87.7` | $11.5 - 15.5\ \%$ | Transcription error. Likely `8.77` / `18.77` or clip. |
| **404** | `rdw` | `50.5` | $11.5 - 15.5\ \%$ | Transcription error. Clip at 99th percentile ($18.5\%$). |

---

## 6. Preprocessing & Modeling Strategy

### A. Cleaning & Imputation Pipeline
1. **Anomaly Correction:** Fix the 5 typographical errors by dividing by 10 (`hb=1222.0 -> 122.2`, `alb=397.0 -> 39.7`) or applying physiological quantile clipping ($1\text{st} - 99\text{th}$ percentile).
2. **Deterministic Validation:** Confirm `neu_lym` matches $\text{neu} / \text{lym}$.

### B. Feature Transformations
- **Logarithmic Transformation:** Apply $\log(1 + x)$ to heavily skewed features:
  - Tumor markers: `cea`, `ca199`, `ca125`, `ca724`
  - Lipids & Enzymes: `lpa`, `alt`, `tb`, `tg`, `neu_lym`
- **Standardization:** RobustScaler or StandardScaler for distance-based and deep learning models.

### C. Generative Modeling with VAE (Variational Autoencoder)
- **Latent Dimension:** Recommended latent dimension $z \in [8, 16]$.
- **Loss Function:** Combine Reconstruction MSE with $\beta$-KL divergence ($\beta \approx 0.001 - 0.01$) to prevent posterior collapse given mixed continuous/discrete inputs.
- **Conditional VAE (C-VAE):** Condition latent space on the target `label` ($y \in \{0, 1\}$) to enable synthetic generation of balanced cancer and control profiles.

### D. Classification Modeling (Random Forest & AdaBoost)
- **Random Forest:** Hyperparameter grid: `n_estimators: [100, 300]`, `max_depth: [5, 10, None]`, `class_weight='balanced'`.
- **AdaBoost:** Hyperparameter grid: `learning_rate: [0.01, 0.1, 0.5]`, `n_estimators: [100, 200]`, base estimator with tree depth 2–3.
- **Cross-Validation:** 5-Fold Stratified Cross-Validation (`StratifiedKFold`) to ensure reliable out-of-fold metrics (ROC-AUC, F1, Precision, Recall).

---
*Report generated and stored in `dataset/dataset_analysis_report.md`.*
