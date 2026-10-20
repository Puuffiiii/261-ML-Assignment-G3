# Dự án Học máy — Dự đoán nhu cầu chuyển việc

Dự án xây dựng quy trình hoàn chỉnh cho dữ liệu bảng **HR Analytics: Job Change of Data Scientists**, từ khám phá dữ liệu (EDA), tiền xử lý, tạo đặc trưng đến xuất dữ liệu sẵn sàng huấn luyện mô hình Machine Learning và Deep Learning.

Notebook chính đã được chạy kiểm chứng từ đầu đến cuối:

[![Mở bằng Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Puuffiiii/261-ML-Assignment-G3/blob/feature/preprocessing/notebooks/ML_Tabular_Final.ipynb)

## Thông tin môn học

- Môn học: Học máy
- Học kỳ: I, năm học 2026–2027
- Khoa: Khoa Khoa học và Kỹ thuật Máy tính
- Trường: Đại học Bách Khoa, ĐHQG-HCM
- Nhóm: 3

| STT | Họ và tên | MSSV |
|---:|---|---|
| 1 | Phạm Hoàng Minh Hiếu | Đang cập nhật |
| 2 | Lê Duy Nhật Minh | Đang cập nhật |
| 3 | Nguyên Ngô Huy | Đang cập nhật |
| 4 | Nguyễn Nhật Thành | 2353102 |

## Dữ liệu và bài toán

- Nguồn: [Kaggle — HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists)
- Kích thước: **19.158 dòng × 14 cột**.
- Bài toán: phân loại nhị phân.
- Biến đích: `target` (`0`: không muốn chuyển việc, `1`: muốn chuyển việc).
- Tỷ lệ lớp dương: khoảng **24,94%**, vì vậy Accuracy không nên là thước đo duy nhất.
- Có 8 cột khuyết dữ liệu; `company_type` khuyết 32,05% và `company_size` khuyết 30,99%.

## Kết quả EDA và quyết định xử lý

- `enrollee_id` là định danh duy nhất nên bị loại khỏi mô hình.
- `city` có 123 mức và ánh xạ 1-1 với `city_development_index`; loại `city` để tránh one-hot dư thừa.
- Missingness của `company_size` và `company_type` có liên hệ rõ với target. Pipeline giữ thông tin này bằng mức `Unknown`/`-1` và hai cờ missing.
- Các biến thứ bậc được ánh xạ theo trật tự nghiệp vụ; biến danh nghĩa dùng One-Hot Encoding với khả năng xử lý danh mục chưa xuất hiện ở train.
- Split phân tầng **70% train / 15% validation / 15% test**; mọi thống kê imputation, encoding, scaling và PCA chỉ được fit trên train.

## Feature engineering

Pipeline tạo thêm 7 đặc trưng không sử dụng target:

1. `training_hours_log1p`
2. `city_development_index_squared`
3. `experience_x_city_development`
4. `training_hours_per_experience`
5. `career_mobility_ratio`
6. `education_x_experience`
7. `profile_missing_count`

Sau encoding và feature engineering có **39 đặc trưng**. Dự án xuất bốn representation:

| Representation | Số chiều | Mục đích gợi ý |
|---|---:|---|
| `std` | 39 | Logistic Regression, SVM, MLP/ReLU |
| `minmax` | 39 | MLP hoặc mô hình yêu cầu miền `[0, 1]` |
| `pca90` | 22 | Mô hình tuyến tính/SVM, ưu tiên tốc độ |
| `pca95` | 24 | PCA giữ nhiều thông tin hơn |

Mọi ma trận đặc trưng được lưu ở `float32`, không chứa NaN/Inf. Smoke test trên validation đạt ROC-AUC **0,7815** với Logistic Regression và **0,7774** với MLP; đây chỉ là kiểm tra data contract, chưa phải kết quả tuning cuối cùng.

## Cấu trúc dự án

```text
.
├── data/
│   └── aug_train.csv
├── notebooks/
│   ├── ML_Tabular_Final.ipynb       # Notebook hợp nhất, nên chạy file này
│   ├── 01_eda_exploration.ipynb     # Notebook EDA lịch sử
│   └── 02_preprocessing_pipeline.ipynb
├── modules/
│   ├── eda_viz.py
│   └── preprocessing.py
├── features/                        # Mảng train/validation/test và preprocessor
├── reports/
│   ├── figures/                     # 17 biểu đồ EDA/PCA
│   ├── categorical_target_rates.csv
│   └── missingness_target_rates.csv
├── configs/
├── requirements.txt
└── README.md
```

## Cài đặt và chạy

Yêu cầu Python 3.10 trở lên. Khuyến nghị dùng môi trường ảo:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/ML_Tabular_Final.ipynb
```

Trong Jupyter/Colab, chọn **Run All**. Notebook tự tìm project root, đọc `data/aug_train.csv`, tái tạo báo cáo và ghi artefact vào `features/`.

## Nạp dữ liệu đã xử lý

```python
import numpy as np

X_train = np.load("features/X_train_std.npy")
X_validation = np.load("features/X_validation_std.npy")
X_test = np.load("features/X_test_std.npy")
y_train = np.load("features/y_train.npy")
y_validation = np.load("features/y_validation.npy")
y_test = np.load("features/y_test.npy")
```

Với Deep Learning, có thể dùng `std` hoặc `minmax`, đặt `input_dim = X_train.shape[1]`, dùng validation cho early stopping và chỉ dùng test sau khi đã khóa mô hình. TensorFlow/PyTorch không nằm trong `requirements.txt`; cài framework phù hợp với môi trường CPU/GPU sử dụng.

## Tái lập và chống rò rỉ dữ liệu

- Seed: `RANDOM_STATE = 42`.
- Preprocessor đã fit: `features/preprocessors.joblib`.
- Chỉ số dòng của từng split: `features/split_indices.npz`.
- Cấu hình, phiên bản thư viện, tỷ lệ lớp và kích thước: `features/metadata.json`.
- Không resample trước split. Nếu áp dụng SMOTE/undersampling, phải đặt thao tác này bên trong từng fold của tập train.

Xem thêm: [hướng dẫn notebook](notebooks/README.md), [mô tả đặc trưng](features/README.md), [module](modules/README.md), [báo cáo EDA](reports/README.md).
