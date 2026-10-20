"""
Modules package for Machine Learning Pipeline & EDA utilities.
"""

from .eda_viz import (
    get_missing_stats,
    plot_target_distribution,
    plot_numerical_feature,
    plot_nominal_feature,
    plot_ordinal_feature,
    plot_high_cardinality_feature,
    plot_missing_values,
    plot_missingness_vs_target,
    plot_correlation_matrix,
    check_city_cdi_relationship,
    plot_pca_scree_plot,
)

from .preprocessing import (
    TabularPreprocessor,
    ORDINAL_MAPPINGS,
    NUMERICAL_COLS,
    ORDINAL_COLS,
    NOMINAL_COLS,
    DROP_COLS,
    MNAR_COLS,
    TARGET_COL,
    load_raw_dataset,
    split_train_test,
    save_processed_features,
    load_processed_features,
)

__all__ = [
    # EDA
    "get_missing_stats",
    "plot_target_distribution",
    "plot_numerical_feature",
    "plot_nominal_feature",
    "plot_ordinal_feature",
    "plot_high_cardinality_feature",
    "plot_missing_values",
    "plot_missingness_vs_target",
    "plot_correlation_matrix",
    "check_city_cdi_relationship",
    "plot_pca_scree_plot",
    # Preprocessing
    "TabularPreprocessor",
    "ORDINAL_MAPPINGS",
    "NUMERICAL_COLS",
    "ORDINAL_COLS",
    "NOMINAL_COLS",
    "DROP_COLS",
    "MNAR_COLS",
    "TARGET_COL",
    "load_raw_dataset",
    "split_train_test",
    "save_processed_features",
    "load_processed_features",
]
