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

- **Dataset:** [Dataset Name]
- **Source:** [Public Dataset URL]
- **Problem Type:** [Classification / Regression]
- **Target Variable:** [Target]
- **Number of Samples:** [To be updated]
- **Number of Features:** [To be updated]

The selected dataset contains:

- Numerical features
- Categorical features
- Missing values

These characteristics allow the implementation and comparison of different preprocessing techniques required by the assignment.

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
├── notebooks/
│   └── ML_Tabular_Final.ipynb
│
├── modules/
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── visualization.py
│
├── reports/
│   └── ML_Assignment_Report.pdf
│
├── features/
│   └── processed feature files
│
├── evidence/
│   └── teamwork and progress evidence
│
├── configs/
│   └── experiment configuration files
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

**Main Colab Notebook:** To be updated.

---

---

## 16. Current Project Status

- [x] Repository created
- [x] Initial project structure created
- [ ] Dataset finalized
- [ ] EDA completed
- [ ] Preprocessing pipeline completed
- [ ] Traditional ML models completed
- [ ] Hyperparameter tuning completed
- [ ] Final evaluation completed
- [ ] Deep learning extension completed
- [ ] Final report completed
- [ ] Final Colab Run All test passed
