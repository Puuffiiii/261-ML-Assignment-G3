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
