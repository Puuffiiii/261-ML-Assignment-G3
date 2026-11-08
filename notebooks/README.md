# Notebooks

## Notebook chính

### `ML_Tabular_Final.ipynb`

Notebook hợp nhất và là điểm chạy được khuyến nghị. Nội dung gồm:

1. kiểm tra schema, dữ liệu trùng và dữ liệu khuyết;
2. EDA cho target, biến số, biến phân loại/thứ bậc, city, missingness và tương quan;
3. split phân tầng 70/15/15;
4. imputation, ordinal mapping, one-hot encoding và xử lý MNAR;
5. tạo 7 đặc trưng mới;
6. StandardScaler, MinMaxScaler và PCA 90%/95%;
7. tính class weight, xuất artefact và kiểm tra NaN/Inf;
8. thử 24 cấu hình Logistic Regression/Random Forest trên validation;
9. khóa siêu tham số và threshold, refit train+validation, đánh giá test;
10. lưu biểu đồ, bảng kết quả và hai bundle suy luận từ dữ liệu thô;
11. fit preprocessing deep learning riêng trên train;
12. tuning ba cấu hình ResNet và ba cấu hình TabM, chọn winner/threshold trên validation;
13. đánh giá hai deep models đã khóa trên test và so sánh với hai traditional models;
14. lưu từng checkpoint `state_dict`, preprocessor dùng chung và kiểm tra raw-row inference.

[![Mở bằng Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Puuffiiii/261-ML-Assignment-G3/blob/main/notebooks/ML_Tabular_Final.ipynb)

Notebook 42 cell đã được chạy toàn bộ thành công. Các kết quả chính: 13.410 dòng train, 2.874 dòng validation, 2.874 dòng test; 39 đặc trưng truyền thống; PCA 90% còn 22 chiều và PCA 95% còn 24 chiều. ResNet đạt test PR-AUC 0,5657 và F1 0,6438; TabM đạt 0,5576 và 0,6427. Cả hai đều cao hơn Random Forest theo hai metric chính này.

## Notebook lịch sử

- `01_eda_exploration.ipynb`: phiên bản EDA tách riêng ban đầu.
- `02_preprocessing_pipeline.ipynb`: phiên bản tiền xử lý train/test ban đầu.

Hai notebook này được giữ để theo dõi lịch sử phát triển. Để tái tạo artefact hiện tại, chỉ chạy `ML_Tabular_Final.ipynb`.

## Cách chạy

Từ thư mục gốc dự án:

```bash
jupyter notebook notebooks/ML_Tabular_Final.ipynb
```

Sau đó chọn **Run All**. Có thể mở notebook khi working directory là thư mục gốc hoặc `notebooks/`; mã nguồn tự nhận diện project root.
