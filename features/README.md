# Đặc trưng đã xử lý

Thư mục chứa artefact do `notebooks/ML_Tabular_Final.ipynb` sinh ra. Split được phân tầng theo target với tỷ lệ 70/15/15 và seed 42.

## Các ma trận đặc trưng

| Nhóm file | Train | Validation | Test | Kiểu dữ liệu |
|---|---:|---:|---:|---|
| `X_*_std.npy` | `(13410, 39)` | `(2874, 39)` | `(2874, 39)` | `float32` |
| `X_*_minmax.npy` | `(13410, 39)` | `(2874, 39)` | `(2874, 39)` | `float32` |
| `X_*_pca90.npy` | `(13410, 22)` | `(2874, 22)` | `(2874, 22)` | `float32` |
| `X_*_pca95.npy` | `(13410, 24)` | `(2874, 24)` | `(2874, 24)` | `float32` |
| `y_*.npy` | `(13410,)` | `(2874,)` | `(2874,)` | `int64` |

Quy ước `*` lần lượt là `train`, `validation`, `test`. Tất cả ma trận đã được kiểm tra không có NaN/Inf.

## Artefact bổ sung

- `feature_names.json`: tên 39 đặc trưng đầy đủ, 7 đặc trưng tạo mới và tên thành phần PCA.
- `metadata.json`: tỷ lệ split, tỷ lệ target, class weight, seed, cấu hình và phiên bản thư viện.
- `preprocessors.joblib`: bốn preprocessor đã fit trên train (`std`, `minmax`, `pca90`, `pca95`).
- `split_indices.npz`: index gốc của train/validation/test để audit và tái lập.

`preprocessors.joblib` phụ thuộc phiên bản scikit-learn. Khi nạp ở môi trường khác, nên dùng phiên bản ghi trong `metadata.json` hoặc chạy lại notebook.

## Ví dụ nạp dữ liệu

```python
import numpy as np

X_train = np.load("features/X_train_std.npy")
X_validation = np.load("features/X_validation_std.npy")
X_test = np.load("features/X_test_std.npy")
y_train = np.load("features/y_train.npy")
y_validation = np.load("features/y_validation.npy")
y_test = np.load("features/y_test.npy")
```

Hoặc dùng hàm tiện ích:

```python
from modules.preprocessing import load_processed_features

arrays = load_processed_features("features")
X_train = arrays["X_train_std"]
```

## Khuyến nghị sử dụng

- `std`: Logistic Regression, SVM, MLP dùng ReLU.
- `minmax`: neural network hoặc thuật toán yêu cầu đầu vào `[0, 1]`.
- `pca90`/`pca95`: thử nghiệm giảm chiều cho mô hình tuyến tính/SVM.
- Validation dùng cho tuning và early stopping; test chỉ dùng một lần để báo cáo cuối.
