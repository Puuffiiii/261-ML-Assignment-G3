# Machine Learning Course Project — Tabular Data

## 1. Course Information

- **Course:** Machine Learning
- **Semester:** Semester I
- **Academic Year:** 2026–2027
- **Faculty:** Faculty of Computer Science and Engineering
- **University:** Ho Chi Minh City University of Technology, VNU-HCM
- **Instructor:** [Instructor Name]
- **Group:** [Group Number]

---

## 2. Team Members

| No. | Full Name | Student ID |
|---|---|---|---|---|
| 1 | Phạm Hoàng Minh Hiếu | [Student ID] | 
| 2 | Lê Duy Nhật Minh | [Student ID] |
| 3 | Nguyên Ngô Huy | [Student ID] | 
| 4 | Nguyễn Nhật Thành | 2353102 | 

---

## 3. Project Objective

This project develops a complete traditional machine learning pipeline for a tabular dataset.

The pipeline includes:

- Exploratory Data Analysis (EDA)
- Missing-value handling
- Categorical feature encoding
- Feature scaling
- Feature selection / dimensionality reduction
- Machine learning model training
- Hyperparameter tuning
- Model evaluation and comparison
- Result analysis

An additional deep learning pipeline using an MLP will be investigated as an extension for comparison with traditional machine learning methods.

---

## 4. Dataset

- **Dataset:** HR Analytics: Job Change of Data Scientists (`aug_train.csv`)
- **Source:** [Kaggle - HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists)
- **Problem Type:** Binary Classification (`0`: Not looking for job change, `1`: Looking for job change)
- **Target Variable:** `target`
- **Number of Samples:** 19,158 rows
- **Number of Features:** 14 columns (1 identifier `enrollee_id`, 1 target, 12 feature columns)

The selected dataset contains:

- **Numerical features:** `city_development_index`, `training_hours`
- **Categorical features:**
  - *Nominal:* `gender`, `company_type`, `major_discipline`, `relevent_experience`, `enrolled_university`, `city`
  - *Ordinal:* `education_level`, `company_size`, `experience`, `last_new_job`
- **Missing values:** Present across 8 features (up to 32% in `company_type`), demonstrating clear MNAR (Missing Not At Random) patterns in company features.

These characteristics allow the rigorous implementation and comparison of imputation, categorical encoding, and scaling techniques required by the course assignment.

---

## 5. Machine Learning Pipeline

The main workflow of the project is:

Raw Data  
→ Exploratory Data Analysis  
→ Train/Test Split  
→ Missing Value Imputation  
→ Categorical Encoding  
→ Feature Scaling  
→ Feature Selection / PCA  
→ Model Training  
→ Hyperparameter Tuning  
→ Model Evaluation  
→ Result Analysis

---

## 6. Models

The traditional machine learning models considered in this project include:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

Additional baseline and experimental models may be added during development.

### Deep Learning Extension

- Multilayer Perceptron (MLP)

The MLP model will be compared with the traditional machine learning models.

---

## 7. Evaluation Metrics

Depending on the final problem formulation, the project will evaluate models using appropriate metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

Cross-validation will also be used to evaluate model stability.

---

## 8. Project Structure

```text
.
├── data/
│   ├── README.md
│   └── aug_train.csv             # HR Analytics dataset (19,158 samples)
│
├── notebooks/
│   ├── README.md
│   └── 01_eda_exploration.ipynb  # Comprehensive EDA Colab notebook (7 feature groups)
│
├── modules/
│   ├── __init__.py
│   ├── README.md
│   └── eda_viz.py                # EDA visualization & statistical analysis functions
│
├── reports/
│   ├── README.md
│   └── figures/                  # 17 high-resolution analysis figures (300 DPI)
│
├── features/
│   └── README.md
│
├── evidence/
│   └── README.md
│
├── configs/
│   └── README.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 9. Installation

The project is designed to run primarily on Google Colab.

Required Python packages are listed in:

```text
requirements.txt
```

For a local environment:

```bash
pip install -r requirements.txt
```

---

## 10. How to Run

The final notebook will be available in:

```text
notebooks/ML_Tabular_Final.ipynb
```

The notebook is designed so that it can be executed using:

```text
Runtime → Run all
```

The dataset will be downloaded automatically from its public source.

No personal Google Drive or other personal cloud storage is required.

---

## 11. Reproducibility

Experiments will use fixed random seeds where applicable.

Example:

```python
RANDOM_STATE = 42
```

The final notebook will automatically perform:

1. Environment setup
2. Dataset download
3. Data preprocessing
4. Model training
5. Model evaluation
6. Result generation

---

## 12. Features

Processed or extracted features will be stored in:

```text
features/
```

Feature files may use formats such as:

```text
.npy
.h5
```

If a feature file is too large to store directly in GitHub, its public download location will be documented in the corresponding README.

---

## 13. Report

The final project report will be available at:

```text
reports/ML_Assignment_Report.pdf
```

**Report link:** To be updated.

---

## 14. Google Colab

- **EDA Notebook (1-click Colab):**  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Puuffiiii/261-ML-Assignment-G3/blob/EDA-step/notebooks/01_eda_exploration.ipynb)

---

## 15. Key EDA Findings & Preprocessing Recommendations

The exploratory analysis performed in `notebooks/01_eda_exploration.ipynb` provides the foundation for the upcoming preprocessing and modeling pipeline:

1. **Target Imbalance:** Target variable shows ~24.9% positive class (`target=1`). Evaluation must incorporate PR-AUC, ROC-AUC, and F1-score rather than raw accuracy.
2. **Missing Not At Random (MNAR):** Candidates missing `company_size` and `company_type` exhibit significantly higher churn rates (~40.6% and ~38.3% vs. ~18% in known categories). These columns must not be dropped or filled with naive modes; an explicit `'Unknown'` category with `MissingIndicator` is recommended.
3. **Feature Redundancy (`city` vs. `city_development_index`):** Empirical analysis confirms a strict 1-to-1 relationship between `city` and `city_development_index`. Dropping the high-cardinality `city` column avoids expanding the feature space by 123 one-hot columns without losing geographical economic information.
4. **Experience Monotonic Trend:** `experience` (23 levels) demonstrates a strong monotonic decrease in job-change intention as seniority increases (>35% for <1 year down to <15% for >20 years).
5. **PCA Dimensionality Reduction:** Scree plot and cumulative explained variance ratio indicate that $k=47$ components retain 90% variance, and $k=51$ components retain 95% variance out of 62 post-encoding dimensions.

---

## 16. Current Project Status

- [x] Repository created
- [x] Initial project structure created
- [x] Dataset finalized (`aug_train.csv` - 19,158 samples)
- [x] EDA completed (`notebooks/01_eda_exploration.ipynb` with 17 publication-quality charts & MNAR / PCA findings)
- [ ] Preprocessing pipeline completed
- [ ] Traditional ML models completed
- [ ] Hyperparameter tuning completed
- [ ] Final evaluation completed
- [ ] Deep learning extension completed
- [ ] Final report completed
- [x] EDA Colab Run All test passed (34/34 cells execution verified)
