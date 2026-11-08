# Dự án Học máy — Dự đoán nhu cầu chuyển việc

Pipeline hoàn chỉnh cho bộ dữ liệu bảng **HR Analytics: Job Change of Data Scientists**: EDA, xử lý missing/categorical data, feature engineering, hai mô hình truyền thống (**Logistic Regression**, **Random Forest**) và hai mô hình deep learning (**ResNet**, **TabM**), đánh giá test khóa kín và xuất artefact suy luận.

[![Mở bằng Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Puuffiiii/261-ML-Assignment-G3/blob/main/notebooks/ML_Tabular_Final.ipynb)

Notebook chính đã được kiểm tra **Run All: 42/42 cell chạy thành công, không lỗi**.

## Thông tin môn học và nhóm

- Môn học: Học máy
- Mã môn học: Chưa được cung cấp
- Học kỳ: I, năm học 2026–2027
- Khoa: Khoa Khoa học và Kỹ thuật Máy tính
- Trường: Đại học Bách Khoa, ĐHQG-HCM
- GVHD: Chưa được cung cấp
- Nhóm: 3

| STT | Họ và tên | MSSV | Email |
|---:|---|---|---|
| 1 | Phạm Hoàng Minh Hiếu | Đang cập nhật | Đang cập nhật |
| 2 | Lê Duy Nhật Minh | Đang cập nhật | Đang cập nhật |
| 3 | Nguyên Ngô Huy | Đang cập nhật | Đang cập nhật |
| 4 | Nguyễn Nhật Thành | 2353102 | Đang cập nhật |

> Nhóm xác nhận GVHD đã chấp thuận ngoại lệ bốn thành viên. Trước khi phát hành bản nộp, cần đính kèm minh chứng chấp thuận và bổ sung mã môn học, GVHD, MSSV/email, phân công và tỷ lệ đóng góp thực tế.

## Mục tiêu và dữ liệu

Mục tiêu là dự đoán một ứng viên có nhu cầu chuyển việc (`target=1`) hay không (`target=0`), đồng thời so sánh pipeline truyền thống và deep learning trong cùng một protocol có thể cấu hình và tái lập.

- Nguồn công khai: [Kaggle — HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists)
- File: `data/aug_train.csv`, **19.158 dòng × 14 cột**.
- Lớp dương: khoảng **24,94%**; vì vậy PR-AUC, ROC-AUC, F1 và Balanced Accuracy được báo cáo bên cạnh Accuracy.
- Có categorical values và 8 cột khuyết; `company_type` khuyết 32,05% và `company_size` khuyết 30,99%.

## Thiết kế pipeline

```text
Raw CSV
  → kiểm tra schema/trùng/missing + EDA
  → stratified split 70% train / 15% validation / 15% test
  → imputation + missing indicators
  → ordinal mapping + one-hot encoding
  → 7 engineered features
  → StandardScaler / MinMaxScaler / PCA 90% / PCA 95%
  → tuning Logistic Regression + Random Forest trên validation
  → chọn cấu hình theo PR-AUC, chọn threshold theo F1 validation
  → refit train+validation
  → đánh giá test đúng một lần
  → lưu model + preprocessor + threshold

Nhánh Deep Learning từ cùng split
  → train-only category vocabulary + numerical normalization
  → 10 categorical embeddings + 13 numerical/engineered features
  → ResNet baseline và TabM parameter-efficient ensemble
  → early stopping và xếp hạng 3 candidate/thuật toán trên validation
  → chọn riêng winner + threshold cho mỗi thuật toán trên validation
  → test hai mô hình đã khóa + lưu state_dict/preprocessor/threshold
```

Các quyết định chống rò rỉ dữ liệu:

- `enrollee_id` là ID duy nhất; `city` ánh xạ 1-1 với `city_development_index`, nên cả hai bị loại.
- Mọi median/mode, encoding, scaling và PCA chỉ được fit trên train.
- Missingness của `company_size`/`company_type` có tín hiệu; pipeline giữ `Unknown`/`-1` và cờ missing riêng.
- Test không được dùng để chọn representation, siêu tham số, threshold hay mô hình champion.
- Cấu hình thí nghiệm nằm tại [`configs/model_experiments.json`](configs/model_experiments.json).
- Deep-learning search space nằm tại [`configs/deep_learning_experiments.json`](configs/deep_learning_experiments.json); vocabulary/normalization cũng chỉ fit trên train.

Sau encoding và feature engineering có **39 đặc trưng**; PCA 90% còn 22 chiều và PCA 95% còn 24 chiều. Các file `.npy` tương ứng được lưu trong `features/`.

## Hai mô hình truyền thống

- **Logistic Regression**: baseline tuyến tính có regularization, nhanh và dễ diễn giải.
- **Random Forest**: mô hình cây phi tuyến, học được tương tác giữa các biến và ít phụ thuộc vào giả định tuyến tính.

Notebook thử **24 cấu hình** model/representation/hyperparameter. PR-AUC validation được dùng làm tiêu chí chọn chính vì dữ liệu lệch lớp; threshold tối đa hóa F1 được khóa trên validation trước khi mở test.

## Deep Learning: ResNet và TabM

ResNet là baseline residual MLP cho dữ liệu bảng: categorical embeddings được nối với 13 numerical/engineered features rồi đi qua các residual block có BatchNorm và dropout. TabM dùng cùng đầu vào nhưng huấn luyện nhiều member nội bộ đồng thời bằng trọng số tuyến tính dùng chung và rank-one adapters riêng, sau đó lấy trung bình xác suất khi suy luận.

- 10 categorical fields với missing/unknown token học riêng từ train.
- 13 numerical/engineered features chuẩn hóa bằng thống kê train.
- Ba cấu hình cho mỗi thuật toán; AdamW, cosine learning-rate schedule, gradient clipping và early stopping theo PR-AUC.
- Winner và threshold tối đa hóa F1 được chọn riêng cho từng thuật toán hoàn toàn trên validation.
- ResNet winner có 838.833 tham số; TabM winner có 521.149 tham số và 8 member nội bộ.

### Kết quả test khóa kín

| Mô hình | Representation | Accuracy | Balanced Acc. | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | `std` | 0,7610 | 0,7644 | 0,5139 | **0,7713** | 0,6168 | 0,7963 | 0,5288 |
| Random Forest | `std` | 0,7947 | 0,7725 | 0,5692 | 0,7280 | 0,6389 | **0,8120** | 0,5414 |
| ResNet | embeddings + numerical features | 0,7944 | **0,7778** | 0,5669 | 0,7448 | **0,6438** | 0,8087 | **0,5657** |
| TabM | embeddings + numerical features | **0,7958** | 0,7759 | **0,5702** | 0,7364 | 0,6427 | 0,8093 | 0,5576 |

Theo metric chính cho dữ liệu lệch lớp, ResNet cải thiện test PR-AUC từ **0,5414 lên 0,5657** (+0,0243) và F1 từ **0,6389 lên 0,6438**; TabM cũng cải thiện lên **0,5576** và **0,6427**. Random Forest vẫn có ROC-AUC cao nhất, TabM có Accuracy/Precision cao nhất, ResNet có Balanced Accuracy/PR-AUC/F1 cao nhất, còn Logistic Regression có Recall cao nhất. Không mô hình nào tốt nhất trên mọi metric. Xem [`reports/all_models_test_comparison.csv`](reports/all_models_test_comparison.csv) và `reports/figures/`.

### Audit độ tin cậy và trách nhiệm

- Repeated 5-fold CV × 3 trên train+validation: Random Forest PR-AUC `0,5457 ± 0,0148`, Logistic Regression `0,5245 ± 0,0115`.
- Paired bootstrap 2.000 lần: ResNet tăng PR-AUC so với Random Forest với CI 95% xấp xỉ `[0,00001; 0,04734]`; CI của chênh lệch F1 chứa 0 nên không khẳng định deep model thắng tổng quát.
- Calibration: Brier score của ResNet/TabM khoảng `0,135`, tốt hơn các model truyền thống có class weighting (`0,170–0,184`).
- Có validation ablation, permutation importance và audit theo gender/missing; đây là phân tích mô tả, không dùng để chọn lại mô hình.
- Phạm vi sử dụng, rủi ro và giới hạn triển khai được ghi tại [`MODEL_CARD.md`](MODEL_CARD.md).

## Cấu trúc dự án

```text
.
├── configs/
│   ├── model_experiments.json       # Traditional ML search space
│   └── deep_learning_experiments.json
├── data/
│   └── aug_train.csv                # Dữ liệu gốc
├── features/                        # 4 representation × 3 split, metadata, preprocessor
├── models/
│   ├── logistic_regression.joblib   # Model + preprocessor + threshold
│   ├── random_forest.joblib
│   ├── resnet.pt                    # Portable PyTorch state_dict
│   ├── tabm.pt
│   ├── deep_tabular_preprocessor.joblib
│   └── *_manifest.json
├── modules/
│   ├── eda_viz.py
│   ├── preprocessing.py
│   ├── modeling.py
│   └── deep_learning.py
├── notebooks/
│   ├── ML_Tabular_Final.ipynb       # Front-end Colab, chạy file này
│   ├── 01_eda_exploration.ipynb
│   └── 02_preprocessing_pipeline.ipynb
├── reports/
│   ├── figures/                     # EDA, PCA, ROC/PR, confusion matrices
│   ├── model_validation_results.csv
│   ├── model_test_results.csv
│   ├── deep_validation_results.csv
│   ├── deep_test_results.csv
│   └── all_models_test_comparison.csv
├── scripts/
│   ├── generate_audits.py            # Tái tạo robustness/fairness/calibration audit
│   └── update_notebook.py
├── tests/
│   ├── test_modeling.py
│   ├── test_deep_learning.py
│   └── test_audit.py
├── MODEL_CARD.md
├── requirements.txt
└── README.md
```

## Cài đặt và chạy

Yêu cầu Python 3.10 trở lên:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/ML_Tabular_Final.ipynb
```

Trong Jupyter/Colab, chọn **Runtime → Run all**. Trên Colab, notebook clone repository công khai, cài `requirements.txt`, tải CSV công khai nếu thiếu và xác minh SHA-256; không mount Google Drive. Khi chạy local, notebook tự nhận diện project root. Toàn bộ feature/model/report/audit artifact được tái tạo tự động. Deep-learning search chạy CPU để kết quả nhất quán; thời gian tùy máy.

Chạy kiểm thử module:

```bash
python -m unittest discover -s tests -v
```

## Suy luận từ dữ liệu thô — ResNet hoặc TabM

```python
import pandas as pd

from modules.deep_learning import (
    load_deep_model_artifact,
    predict_deep_from_raw_dataframe,
)

model, preprocessor, checkpoint = load_deep_model_artifact(
    "models/resnet.pt",  # đổi thành models/tabm.pt để suy luận bằng TabM
    "models/deep_tabular_preprocessor.joblib",
)
raw_rows = pd.read_csv("data/aug_train.csv").drop(columns="target").head(5)
predictions = predict_deep_from_raw_dataframe(
    model, preprocessor, checkpoint, raw_rows
)
print(predictions)
```

Hàm dùng đúng vocabulary/normalization train-only và threshold đã khóa trên validation. Với TabM, xác suất là trung bình của 8 member nội bộ. Ví dụ Random Forest vẫn có trong [`models/README.md`](models/README.md).

## Tái lập và artefact

- Seed cố định: `42`.
- Split index: `features/split_indices.npz`.
- Preprocessor: `features/preprocessors.joblib`.
- Phiên bản/thống kê dữ liệu: `features/metadata.json`.
- Mô hình cuối: `models/*.joblib`; manifest: `models/model_manifest.json`.
- Deep checkpoints: `models/resnet.pt`, `models/tabm.pt`; preprocessor dùng chung và `models/deep_models_manifest.json`.
- Deep history/results: `reports/deep_*.csv`; so sánh chung: `reports/all_models_test_comparison.csv`.
- Feature importance/coefficient: `reports/model_feature_importance.csv`.
- Audit provenance/checksum: `reports/audit_provenance.json`; locked predictions, bootstrap CI, calibration, repeated CV, ablation và subgroup audit nằm cùng thư mục.
- Không resample trước split. Nếu mở rộng bằng SMOTE/undersampling, thao tác phải nằm bên trong từng fold train.

Tài liệu chi tiết: [notebooks](notebooks/README.md), [modules](modules/README.md), [features](features/README.md), [reports](reports/README.md), [models](models/README.md).

## Liên kết nộp bài

- Notebook Colab: dùng badge ở đầu README; sau khi điền metadata thật và kiểm thử runtime sạch, thay `main` bằng tag submission bất biến.
- Báo cáo PDF tiếng Việt: [`reports/report_vi.pdf`](reports/report_vi.pdf).
- Mã nguồn LaTeX của báo cáo: [`reports/report_vi.tex`](reports/report_vi.tex). Báo cáo đã được đồng bộ sang ResNet/TabM và các audit mới; các ô thông tin hành chính/thành viên vẫn phải được nhóm điền bằng dữ liệu thật.
- Đề bài gốc: [`MLAssignment_VNv2-2.pdf`](MLAssignment_VNv2-2.pdf).
