# Modules

This directory contains modularized Python source code supporting the machine learning pipeline and analysis.

## Available Modules

### 1. `eda_viz.py`
Contains reusable, modular plotting and statistical functions categorized by feature group:
- `get_missing_stats(df)`: Missing value count, percentage, and cardinality overview.
- `plot_target_distribution(df)`: Bar chart and donut chart for target imbalance.
- `plot_numerical_feature(df, col)`: Histogram (KDE dodge) and boxplot separated by target.
- `plot_nominal_feature(df, col)`: Category counts and target churn rate with baseline.
- `plot_ordinal_feature(df, col, order)`: Ordered bar charts (and line chart with markers for `experience`).
- `plot_high_cardinality_feature(df, col, top_n)`: Horizontal bar chart for Top N categories và target rates.
- `plot_missing_values(df)`: Missing percentages and sample counts across columns.
- `plot_missingness_vs_target(df)`: Compares target churn rate between present vs missing data (MNAR test).
- `plot_correlation_matrix(df)`: 7x7 Spearman rank correlation heatmap.
- `check_city_cdi_relationship(df)`: Checks and validates 1-to-1 relationship between `city` and `city_development_index`.
- `plot_pca_scree_plot(df)`: PCA scree plot and cumulative explained variance curve (identifying 90% and 95% thresholds).

### 2. `preprocessing.py`
Contains the end-to-end data preprocessing pipeline adhering strictly to the assignment specifications:
- `TabularPreprocessor`: Scikit-learn compatible transformer supporting:
  - Missing value imputation with MNAR treatment (explicit `'Unknown'` category and `MissingIndicator` for `company_size` and `company_type`).
  - Integer ordinal mapping for `education_level`, `company_size`, `experience`, `last_new_job`.
  - One-Hot Encoding for nominal features (`gender`, `relevent_experience`, `enrolled_university`, `major_discipline`, `company_type`).
  - Configurable scaling: `StandardScaler` or `MinMaxScaler(feature_range=(min, max))`.
  - Configurable dimensionality reduction via `PCA` (e.g. 90% or 95% variance retention).
- `load_raw_dataset(csv_path)`: Loads raw dataset with fallback path support.
- `split_train_test(df, test_size, random_state, drop_unnecessary)`: Stratified train-test split (80/20) with ID and redundant `city` removal.
- `save_processed_features(output_dir, feature_sets, y_train, y_test, ...)`: Exports extracted feature arrays (`.npy`) and metadata to `features/`.
- `load_processed_features(features_dir)`: Loads extracted feature arrays back into memory.
