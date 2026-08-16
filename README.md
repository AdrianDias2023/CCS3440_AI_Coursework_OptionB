# SmartCare AI – Patient Readmission Prediction System
## CCS3440 Artificial Intelligence Coursework (Option B)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5.0-orange.svg)](https://scikit-learn.org/)
[![Streamlit App](https://img.shields.io/badge/streamlit-1.61.1-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Executive Summary

**SmartCare AI** is an end-to-end clinical decision support system designed to predict 30-day hospital readmission risk (`readmitted_30_days`) specifically for **discharged hospital inpatients**. The project implements a leakage-free machine learning architecture using scikit-learn `Pipeline` objects, evaluated under **Stratified 5-Fold Nested Cross-Validation**, backed by **SHAP (SHapley Additive exPlanations)** interpretability and demographic subgroup fairness analysis. The solution includes an interactive Streamlit bedside risk calculator (`app.py`).

---

## 👥 Student Identification & Group Members

* **Student ID(s):** `23UG1-0018`, `23UG1-0151`, `23UG1-0072`, `23UG1-0069`, `CIT-23-02-0091`
* **Module:** CCS3440 – Artificial Intelligence Coursework
* **Task:** Option B – Patient Readmission Prediction

---

## 🎯 Problem Definition & Clinical Significance

### Problem Formulation
Predicting whether a patient will be readmitted to the hospital within 30 days of discharge is formulated as a binary classification problem:
* **`readmitted_30_days = 1`**: Patient is readmitted within 30 days of discharge.
* **`readmitted_30_days = 0`**: Patient is not readmitted.

### Target Cohort Definition (`admitted == 1`)
The clinical risk model is strictly restricted to **discharged hospital inpatients** ($N=330$). Outpatients (`admitted == 0`) have zero hospital stay duration and zero readmission events by definition. Including outpatients introduces trivial target correlation where admission status dominates prediction rather than actual post-discharge clinical risk factors.

### Clinical Significance
1. **Quality of Care:** Unplanned readmissions highlight suboptimal inpatient care or inadequate post-discharge transitional care.
2. **Financial Savings:** Reducing preventable readmissions prevents major financial penalties for healthcare providers and burden on patients.
3. **Resource Optimization:** Enables clinicians to allocate high-intensity follow-up care and telemetry monitoring to high-risk patients.

---

## 📂 Repository File Structure

```text
├── README.md                                # Comprehensive project documentation
├── app.py                                   # Streamlit interactive bedside risk calculator dashboard
├── assignment_02_coursework_optionb.py     # Complete Python execution script (Tasks 01-08)
├── Assignment_02_Coursework_OptionB.ipynb  # Interactive Jupyter Notebook version
├── requirements.txt                         # Python environment dependencies
├── best_readmission_rf_model.joblib        # Serialized production Random Forest pipeline model
├── readmission_preprocessor.joblib         # Serialized ColumnTransformer preprocessor object
├── smartcare_ai_dataset_1000.csv           # Raw SmartCare dataset (1,000 patient records)
├── smartcare_ai_dataset_data_dictionary.csv # Dataset schema & feature dictionary definitions
└── github_repository_link.txt              # GitHub version control repository link
```

---

## 🔬 Literature Review & Research Gaps

The project builds upon 5 peer-reviewed research papers:
1. **Deep Learning on EHRs (Rajkomar et al., 2018, *npj Digital Medicine*):** Showed deep models capture complex EHR dependencies (AUROC 0.75–0.76) but act as black boxes.
2. **Traditional Risk Models (Kansagara et al., 2011, *JAMA*):** Demonstrated that administrative models (LACE index) achieve poor discrimination (c-statistic 0.55–0.65).
3. **Explainable AI (Lundberg & Lee, 2017, *NeurIPS*):** Established SHAP as the mathematically consistent local feature attribution method.
4. **Intelligible Classifiers (Caruana et al., 2015, *KDD*):** Highlighted the critical need for interpretable classifiers to prevent harmful data correlations.
5. **Clinical Predictor Methodology (Amarasingham et al., 2010, *Medical Care*):** Proved that combining clinical EHR vitals with compliance barriers outperforms administrative billing codes.

### Key Research Gaps Addressed
* **Bridging Accuracy & Intelligibility:** Combining a GridSearch-tuned Random Forest ensemble with local SHAP attributions and global Permutation Importance.
* **Eliminating Data Leakage:** Restricting the cohort to discharged inpatients ($N=330$) and encapsulating feature scaling, encoding, and ANOVA feature selection inside scikit-learn `Pipeline` objects.
* **Subgroup Fairness Verification:** Testing demographic parity and equalized recall across Gender (Male vs Female) and Age Groups (<50 vs $\ge$50).

---

## ⚙️ Preprocessing & Machine Learning Pipeline

### Data Cleaning & Feature Engineering
* **Missing Value Imputation:** `room_type` missing values (recording gaps) imputed with `'None'` (Unknown/Unassigned).
* **Outlier Handling:** Retained severe clinical outliers (long ICU stays, high total bills) to preserve high-risk indicators.
* **Feature Engineering:**
  * `pulse_pressure` = `systolic_bp - diastolic_bp`
  * `missed_ratio` = `missed_previous_appointments / (previous_appointments + 1)`
  * `charge_per_day` = `total_bill_lkr / (length_of_stay_days + 1)`

### Encapsulated Pipeline Architecture
* **Numerical Features:** Centered and normalized using `StandardScaler`.
* **Categorical Features:** One-hot encoded using `OneHotEncoder(drop='first', handle_unknown='ignore')`.
* **Feature Selection:** Top 80% preprocessed features selected via univariate ANOVA F-test (`SelectPercentile(score_func=f_classif, percentile=80)`).

---

## 📊 Evaluation & Empirical Results

All models are evaluated using **Stratified 5-Fold Nested Cross-Validation** (outer 5 folds for evaluation, inner 5 folds for hyperparameter tuning) across the inpatient cohort ($N=330$).

### Stratified 5-Fold Nested Cross-Validation Summary ($N = 330$)

| Model Classifier | Nested CV Mean Accuracy | Nested CV Mean Precision | Nested CV Mean Recall | Nested CV Mean F1-Score | Nested CV Mean ROC-AUC | Out-of-Fold Confusion Matrix ($TN, FP, FN, TP$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** | $0.7030 \pm 0.0466$ | $0.7738 \pm 0.0291$ | $0.8653 \pm 0.0468$ | $0.8166 \pm 0.0324$ | $0.5595 \pm 0.1006$ | $TN=13, FP=64, FN=34, TP=219$ |
| **K-Nearest Neighbors** | $0.7212 \pm 0.0246$ | $0.7637 \pm 0.0107$ | $0.9207 \pm 0.0287$ | $0.8348 \pm 0.0175$ | $0.5442 \pm 0.1121$ | $TN=5, FP=72, FN=20, TP=233$ |
| **Random Forest (Optimized)** | **$0.7636 \pm 0.0074$** | **$0.7660 \pm 0.0070$** | **$0.9961 \pm 0.0078$** | **$0.8660 \pm 0.0048$** | **$0.5726 \pm 0.0540$** | **$TN=1, FP=76, FN=2, TP=251$** |

### Model Selection Rationale
* **Maximum Clinical Sensitivity:** Random Forest achieved **0.9961 Mean Recall** ($FN=2, TP=251$ out-of-fold), ensuring high-risk patients are not missed.
* **Top Overall Performance:** Highest F1-Score (**0.8660**) and Accuracy (**0.7636**).
* **Variance Stability:** Ensemble bagging produces extreme stability (F1 std dev of **0.0048** vs Decision Tree's **0.0324**).

---

## 🔍 Explainable AI (XAI) & Fairness Analysis

### Global & Local Interpretability
1. **Baseline Permutation Importance ($N=1000$):** Demonstrates how `admitted` dominated predictions in the un-restricted dataset due to administrative eligibility bias.
2. **Refined Inpatient Feature Importance ($N=330$):** Highlights true post-discharge clinical drivers (`length_of_stay_days`, `total_bill_lkr`, `age`, `missed_ratio`).
3. **SHAP Attributions (Lundberg & Lee, 2017):** Provides global feature impact summary plots and local waterfall attributions for individual patient predictions.

### Demographic Subgroup Fairness Performance

| Attribute | Subgroup | Sample Count ($N$) | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Gender** | Female | 173 | 0.7746 | 0.7941 | **1.0000** | 0.8852 |
| **Gender** | Male | 157 | 0.7613 | 0.7613 | **1.0000** | 0.8645 |
| **Age Group** | Age < 50 | 201 | 0.7562 | 0.7563 | **1.0000** | 0.8613 |
| **Age Group** | Age $\ge$ 50 | 129 | 0.8125 | 0.8125 | **1.0000** | 0.8966 |

---

## 🚀 Installation & Execution Guide

### 1. Prerequisites & Environment Setup
Clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/AdrianDias2023/CCS3440_AI_Coursework_OptionB
cd AI-Assignment-02    

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Python Script
To run the complete machine learning workflow (Data ingestion, preprocessing, EDA, cross-validation, XAI, and serialization):

```bash
python assignment_02_coursework_optionb.py
```

### 3. Launching the Interactive Streamlit Web App
To launch the bedside clinical decision support calculator:

```bash
streamlit run app.py
```
*or using the Python module:*
```bash
python -m streamlit run app.py
```
Access the application in your browser at `http://localhost:8501`.

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
"# CCS3440_AI_Coursework_OptionB" 
