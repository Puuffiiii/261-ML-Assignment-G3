# Báo cáo và hình ảnh

Thư mục lưu kết quả phân tích do notebook hợp nhất sinh ra.

## Báo cáo cuối kỳ

- `report_vi.tex`: mã nguồn LaTeX tiếng Việt, bao phủ EDA, pipeline, cấu hình thí nghiệm, kết quả traditional/deep learning, phân tích, hạn chế, cấu trúc sản phẩm và bảng phân công bắt buộc.
- `report_vi.pdf`: bản PDF tiếng Việt được sinh từ source, mô tả Logistic Regression, Random Forest, ResNet, TabM và audit độ tin cậy.
- `Makefile`: `make` để biên dịch pdfLaTeX hai lượt trên máy có Vietnamese TeX fonts; `make html-pdf` là phương án render dự phòng bằng Pandoc và Chromium.
- `report_print.css`, `report_print.js`: định dạng bản PDF dự phòng, gồm trang bìa, mục lục và trang bảng phân công nằm ngang.

Trước khi nộp, điền thông tin môn học/thành viên/phân công/minh chứng bằng dữ liệu thật, biên dịch lại và kiểm tra trực quan PDF.

## Bảng thống kê

- `categorical_target_rates.csv`: số mẫu và tỷ lệ `target=1` theo từng danh mục.
- `missingness_target_rates.csv`: so sánh tỷ lệ target giữa nhóm có dữ liệu và nhóm bị khuyết.
- `model_validation_results.csv`: toàn bộ 24 cấu hình tuning và metric validation.
- `model_test_results.csv`: kết quả cuối của hai cấu hình đã khóa.
- `model_feature_importance.csv`: hệ số Logistic Regression và impurity importance của Random Forest.
- `deep_validation_results.csv`: kết quả ba cấu hình ResNet và ba cấu hình TabM.
- `deep_training_history.csv`: loss/PR-AUC/ROC-AUC theo epoch của hai winner.
- `deep_test_results.csv`: kết quả test của ResNet và TabM đã khóa.
- `all_models_test_comparison.csv`: bảng so sánh hai traditional ML và hai deep models.
- `repeated_cv_results.csv`, `repeated_cv_summary.csv`: 5-fold CV × 3 trên train+validation cho cấu hình traditional đã khóa.
- `feature_ablation_results.csv`: validation-only ablation cho engineered features/missing indicators.
- `model_bootstrap_intervals.csv`, `model_paired_differences.csv`: CI 95% và so sánh cặp trên dự đoán test khóa kín.
- `model_calibration_*.csv`: Brier/ECE và reliability bins.
- `gender_subgroup_audit.csv`: audit lỗi theo gender/missing có đủ cỡ mẫu.
- `model_permutation_importance.csv`: permutation importance trên validation.
- `audit_provenance.json`: version, SHA-256 và chính sách không chọn lại mô hình bằng test.

## Hình ảnh

`figures/` chứa các hình độ phân giải cao:

- `group1_target_distribution.png`: phân phối và mức mất cân bằng target;
- `group2_num_*.png`: hai biến số liên tục;
- `group3_nominal_*.png`: năm biến danh nghĩa;
- `group4_ordinal_*.png`: bốn biến thứ bậc;
- `group5_high_cardinality_city_top15.png`: Top 15 city;
- `group6_missing_values_bar.png`: tỷ lệ dữ liệu khuyết;
- `group6_missingness_vs_target.png`: missingness so với target;
- `group6_correlation_matrix.png`: tương quan Spearman;
- `group7_pca_scree_plot.png`: phương sai giải thích của PCA fit trên train.
- `model_roc_pr_curves.png`: ROC và Precision–Recall của hai mô hình trên test.
- `model_confusion_matrices.png`: confusion matrix tại threshold chọn trên validation.
- `deep_models_learning_curves.png`: learning curves của ResNet và TabM winner.
- `deep_models_test_evaluation.png`: ROC, PR và confusion matrix của hai deep models.
- `all_models_metric_comparison.png`: so sánh F1, ROC-AUC và PR-AUC của cả bốn mô hình.
- `model_bootstrap_f1.png`, `model_calibration_curves.png`, `gender_subgroup_recall.png`: audit uncertainty/calibration/subgroup.

## Kết quả mô hình chính

| Mô hình | Test ROC-AUC | Test PR-AUC | Test F1 |
|---|---:|---:|---:|
| Logistic Regression | 0,7963 | 0,5288 | 0,6168 |
| Random Forest | **0,8120** | 0,5414 | 0,6389 |
| ResNet | 0,8087 | **0,5657** | **0,6438** |
| TabM | 0,8093 | 0,5576 | 0,6427 |

ResNet và TabM đều tốt hơn Random Forest trên PR-AUC và F1. ResNet là mô hình tốt nhất theo hai metric chính; TabM có Accuracy/Precision cao nhất nhưng không được chọn lại dựa trên test.

`report_vi.tex` đã được cập nhật theo ResNet/TabM và các audit. Bản PDF phải được build lại sau mọi thay đổi source và sau khi điền thông tin hành chính.

Không chỉnh sửa thủ công các artefact này; chạy lại `notebooks/ML_Tabular_Final.ipynb` để tái tạo đồng bộ.
