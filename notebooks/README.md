# Notebooks

This directory contains the Jupyter / Google Colab notebooks for the project.

## Available Notebooks

### 1. `01_eda_exploration.ipynb`
- **Description:** Comprehensive Exploratory Data Analysis (EDA) investigating 7 distinct feature groups of the HR Analytics dataset (`aug_train.csv`).
- **Features covered:**
  1. *Target Variable Analysis:* Imbalance ratio (~24.9% vs ~75.1%) via Bar & Donut charts.
  2. *Numerical Features:* Histogram (KDE) and boxplots for `city_development_index` and `training_hours`.
  3. *Low-Cardinality Nominal Features:* Distribution and target rate across `gender`, `company_type`, `major_discipline`, `relevent_experience`, `enrolled_university`.
  4. *Ordinal Features:* Monotonic ordering and trend analysis for `education_level`, `company_size`, `experience` (line chart with markers), `last_new_job`.
  5. *High-Cardinality Features:* Top 15 `city` bar charts and empirical proof of 1-to-1 mapping with `city_development_index`.
  6. *Missingness & Correlation:* Missing bar chart, MNAR validation for company features, and 7x7 Spearman rank correlation matrix.
  7. *PCA Exploration:* Scree plot and cumulative explained variance ratio ($k=47$ for 90%, $k=51$ for 95%).
- **Colab Link:**  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Puuffiiii/261-ML-Assignment-G3/blob/EDA-step/notebooks/01_eda_exploration.ipynb)

---

### 2. `02_preprocessing_pipeline.ipynb`
- **Description:** End-to-end Tabular Preprocessing Pipeline implementing data cleaning, stratified train-test splitting (80/20), MNAR imputation, categorical encoding, scaling comparisons, and PCA dimensionality reduction.
- **Pipeline Components:**
  1. *Data Cleaning:* Identifier removal (`enrollee_id`) and high-cardinality redundant column pruning (`city`).
  2. *Stratified Train-Test Split:* Strictly anti-leakage 80/20 train/test split stratified by binary `target`.
  3. *MNAR & Imputation:* Explicit `'Unknown'` category and `MissingIndicator` flags for `company_size` and `company_type`.
  4. *Encoding:* Ordinal integer mapping for 4 ordinal features and One-Hot Encoding for 5 nominal features.
  5. *Scaling Comparison:* Empirical comparison between `StandardScaler` (Z-score) and `MinMaxScaler` (`feature_range=(0, 1)` and `(-1, 1)`).
  6. *PCA Dimensionality Reduction:* Variance retention benchmarks at 90% ($k=21$) and 95% ($k=23$).
  7. *Feature Persistence:* Auto-exports all processed feature sets (`.npy`), feature names, and metadata to `features/`.
  8. *Integrity Checks:* Verification of shape alignments, zero NaNs, and zero Infs.
- **Colab Link:**  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Puuffiiii/261-ML-Assignment-G3/blob/feature/preprocessing/notebooks/02_preprocessing_pipeline.ipynb)
