# Modules

This directory contains modularized Python source code supporting the machine learning pipeline and analysis.

## Available Modules

### `eda_viz.py`
Contains reusable, modular plotting and statistical functions categorized by feature group:
- `get_missing_stats(df)`: Missing value count, percentage, and cardinality overview.
- `plot_target_distribution(df)`: Bar chart and donut chart for target imbalance.
- `plot_numerical_feature(df, col)`: Histogram (KDE dodge) and boxplot separated by target.
- `plot_nominal_feature(df, col)`: Category counts and target churn rate with baseline.
- `plot_ordinal_feature(df, col, order)`: Ordered bar charts (and line chart with markers for `experience`).
- `plot_high_cardinality_feature(df, col, top_n)`: Horizontal bar chart for Top N categories and target rates.
- `plot_missing_values(df)`: Missing percentages and sample counts across columns.
- `plot_missingness_vs_target(df)`: Compares target churn rate between present vs missing data (MNAR test).
- `plot_correlation_matrix(df)`: 7x7 Spearman rank correlation heatmap.
- `check_city_cdi_relationship(df)`: Checks and validates 1-to-1 relationship between `city` and `city_development_index`.
- `plot_pca_scree_plot(df)`: PCA scree plot and cumulative explained variance curve (identifying 90% and 95% thresholds).
