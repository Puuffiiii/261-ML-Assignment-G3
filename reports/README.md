# Báo cáo và hình ảnh

Thư mục lưu kết quả phân tích do notebook hợp nhất sinh ra.

## Bảng thống kê

- `categorical_target_rates.csv`: số mẫu và tỷ lệ `target=1` theo từng danh mục.
- `missingness_target_rates.csv`: so sánh tỷ lệ target giữa nhóm có dữ liệu và nhóm bị khuyết.

## Hình ảnh

`figures/` chứa 17 hình độ phân giải cao:

- `group1_target_distribution.png`: phân phối và mức mất cân bằng target;
- `group2_num_*.png`: hai biến số liên tục;
- `group3_nominal_*.png`: năm biến danh nghĩa;
- `group4_ordinal_*.png`: bốn biến thứ bậc;
- `group5_high_cardinality_city_top15.png`: Top 15 city;
- `group6_missing_values_bar.png`: tỷ lệ dữ liệu khuyết;
- `group6_missingness_vs_target.png`: missingness so với target;
- `group6_correlation_matrix.png`: tương quan Spearman;
- `group7_pca_scree_plot.png`: phương sai giải thích của PCA fit trên train.

Không chỉnh sửa thủ công các artefact này; chạy lại `notebooks/ML_Tabular_Final.ipynb` để tái tạo đồng bộ.
