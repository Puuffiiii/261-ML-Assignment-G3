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

## `modeling.py`

Module huấn luyện và đánh giá hai thuật toán truyền thống:

- đọc/kiểm tra `configs/model_experiments.json`;
- chạy toàn bộ ứng viên Logistic Regression và Random Forest trên validation;
- tính Accuracy, Balanced Accuracy, Precision, Recall, Specificity, F1, ROC-AUC và PR-AUC;
- chọn threshold tối đa hóa F1 chỉ trên validation;
- chọn cấu hình theo PR-AUC, refit train+validation và đánh giá test đúng một lần;
- xuất coefficient/feature importance, ROC/PR curve và confusion matrix;
- lưu bundle gồm model + preprocessor + threshold;
- `predict_from_raw_dataframe` để suy luận trực tiếp từ schema dữ liệu thô.

## `deep_learning.py`

Pipeline PyTorch cho hai thuật toán deep learning trên dữ liệu bảng:

- `DeepTabularPreprocessor`: vocabulary học trên train, unknown/missing IDs và chuẩn hóa 13 numerical/engineered features;
- `TabularResNet`: baseline residual MLP với categorical embeddings, BatchNorm, skip connection và dropout;
- `TabM`: ensemble hiệu quả tham số gồm nhiều member nội bộ dùng chung trọng số, khác nhau qua rank-one BatchEnsemble adapters;
- AdamW, cosine schedule, gradient clipping, deterministic seed và early stopping bằng validation PR-AUC;
- chạy ba cấu hình cho mỗi thuật toán, chọn riêng ResNet và TabM tốt nhất chỉ bằng validation;
- khóa threshold tối đa hóa validation F1 rồi đánh giá mỗi mô hình trên test đúng một lần;
- lưu/nạp portable `state_dict`, preprocessor dùng chung và suy luận trực tiếp từ raw DataFrame;
- xuất learning curves, ROC/PR, confusion matrix và bảng so sánh chung.
