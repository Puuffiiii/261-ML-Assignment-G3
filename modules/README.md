# Modules

## `eda_viz.py`

Các hàm thống kê và trực quan hóa tái sử dụng:

- `get_missing_stats`: thống kê kiểu dữ liệu, số lượng/tỷ lệ khuyết và cardinality;
- `plot_target_distribution`: biểu đồ cột và donut của target;
- `plot_numerical_feature`: histogram/KDE và boxplot theo target;
- `plot_nominal_feature`, `plot_ordinal_feature`: phân phối và tỷ lệ target theo nhóm;
- `plot_high_cardinality_feature`: Top-N cho biến nhiều danh mục;
- `plot_missing_values`, `plot_missingness_vs_target`: phân tích missingness/MNAR;
- `plot_correlation_matrix`: tương quan Spearman;
- `check_city_cdi_relationship`: kiểm chứng quan hệ giữa `city` và CDI;
- `plot_pca_scree_plot`: khảo sát phương sai giải thích của PCA.

## `preprocessing.py`

`TabularPreprocessor` tương thích API scikit-learn và thực hiện:

- kiểm tra schema đầu vào;
- median/mode imputation học từ train;
- xử lý `company_size`/`company_type` theo tín hiệu MNAR;
- ordinal mapping, missing indicator và One-Hot Encoding;
- 7 đặc trưng phi tuyến/tương tác và mức độ đầy đủ hồ sơ;
- StandardScaler hoặc MinMaxScaler;
- PCA theo tỷ lệ phương sai;
- xử lý danh mục chưa xuất hiện bằng `handle_unknown="ignore"`.

Các hàm workflow:

- `load_raw_dataset`;
- `split_train_test` để tương thích notebook cũ;
- `split_train_validation_test` cho workflow 70/15/15 hiện tại;
- `save_processed_features` và `load_processed_features`.

Ví dụ:

```python
from modules.preprocessing import TabularPreprocessor

preprocessor = TabularPreprocessor(scaling="standard", add_engineered_features=True)
X_train = preprocessor.fit_transform(X_train_raw)
X_validation = preprocessor.transform(X_validation_raw)
X_test = preprocessor.transform(X_test_raw)
```

Luôn `fit` trên train; chỉ gọi `transform` cho validation, test và dữ liệu suy luận.
