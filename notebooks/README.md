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
8. smoke test Logistic Regression và MLP trên validation.

[![Mở bằng Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Puuffiiii/261-ML-Assignment-G3/blob/feature/preprocessing/notebooks/ML_Tabular_Final.ipynb)

Notebook đã được chạy toàn bộ thành công. Các kết quả chính: 13.410 dòng train, 2.874 dòng validation, 2.874 dòng test; 39 đặc trưng đầy đủ; PCA 90% còn 22 chiều và PCA 95% còn 24 chiều.

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
