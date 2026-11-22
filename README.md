# Phân tích dữ liệu nhân sự và dự đoán khả năng chuyển việc sau đào tạo

> **Data-driven HR analytics & post-training job-change intention prediction**

Đồ án môn **Học máy (CO3117)** xây dựng một pipeline hoàn chỉnh cho bộ dữ liệu bảng **HR Analytics: Job Change of Data Scientists**. Dự án bao gồm EDA, xử lý dữ liệu khuyết và biến phân loại, thiết kế đặc trưng, Logistic Regression, Random Forest, ResNet, TabM, đánh giá test khóa kín, audit độ tin cậy và đóng gói artefact phục vụ tái lập/suy luận.

[![Mở bằng Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Puuffiiii/261-ML-Assignment-G3/blob/main/notebooks/ML_Tabular_Final.ipynb)

Notebook chính đã được lưu với **42/42 cell chạy thành công, không có output lỗi**. Báo cáo tiếng Việt hiện hành: [`reports/G3-Report.pdf`](reports/G3-Report.pdf).

## Thông tin học phần và nhóm

- Trường: Đại học Bách Khoa, ĐHQG-HCM
- Khoa: Khoa Khoa học và Kỹ thuật Máy tính
- Môn học: Học máy
- Mã học phần: **CO3117**
- Học kỳ: **HK261** — Học kỳ I, năm học 2026–2027
- Giảng viên hướng dẫn: **TS. Trương Vĩnh Lân**
- Nhóm: **3**

| STT | Thành viên | MSSV | Email | Phụ trách chính |
|---:|---|---:|---|---|
| 1 | Phạm Hoàng Minh Hiếu | 2352740 | hieu.pham22@hcmut.edu.vn | Tiền xử lý; lựa chọn, huấn luyện và tinh chỉnh Logistic Regression; phân tích báo cáo |
| 2 | Lê Duy Nhật Minh | 2352333 | minh.leduy@hcmut.edu.vn | Thiết lập Colab, tải dữ liệu, EDA; huấn luyện, tinh chỉnh và đánh giá Random Forest |
| 3 | Nguyễn Ngô Huy | 2153384 | huy.nguyenngo13@hcmut.edu.vn | Trích xuất/lựa chọn đặc trưng; phối hợp xây dựng và tinh chỉnh ResNet, TabM |
| 4 | Nguyễn Nhật Thành | 2353102 | thanh.nguyen2353102@hcmut.edu.vn | Tích hợp notebook và thí nghiệm; phối hợp pipeline học sâu; hoàn thiện Colab Run All |

Tỷ lệ đóng góp cá nhân không được suy ra từ điểm sprint vì báo cáo chưa cung cấp tỷ lệ đã được cả nhóm xác nhận. Xem mục [Thông tin cần xác nhận trước khi nộp](#thông-tin-cần-xác-nhận-trước-khi-nộp).

## Mục tiêu và câu hỏi nghiên cứu

Dự án dự đoán một ứng viên có nhu cầu chuyển việc (`target=1`) hay không (`target=0`) và so sánh pipeline truyền thống với học sâu trên cùng một phép chia dữ liệu. Các mục tiêu chính:

1. Khảo sát chất lượng, phân phối và quan hệ giữa các biến trong hồ sơ ứng viên.
2. Xây dựng pipeline tiền xử lý có thể cấu hình, tái lập và không rò rỉ dữ liệu.
3. So sánh Logistic Regression và Random Forest trên nhiều biểu diễn/siêu tham số.
4. Đánh giá bằng các chỉ số phù hợp với dữ liệu lệch lớp, ưu tiên PR-AUC và F1.
5. Mở rộng bằng ResNet và TabM, rồi so sánh công bằng với mô hình truyền thống.
6. Lưu notebook, module, đặc trưng, mô hình, kết quả và provenance để kiểm toán.

Ba câu hỏi nghiên cứu là: cấu hình truyền thống nào tốt nhất theo validation PR-AUC; missingness/đặc trưng nào đáng kể; và ResNet/TabM có cải thiện ổn định so với Random Forest hay không.

## Dữ liệu

- Nguồn: [Kaggle — HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists)
- File: [`data/aug_train.csv`](data/aug_train.csv)
- Kích thước: **19.158 dòng × 14 cột**; không có dòng trùng hoàn toàn.
- Nhãn: 14.381 mẫu lớp 0 (75,06%) và 4.777 mẫu lớp 1 (24,94%), xấp xỉ tỷ lệ 3:1.
- `enrollee_id` có 19.158 giá trị duy nhất nên bị loại khỏi cả hai pipeline.
- `city` có 123 mức và ánh xạ 1-1 với `city_development_index`: nhánh truyền thống loại `city` để giảm dư thừa, còn nhánh deep learning **giữ `city` dưới dạng embedding**.
- Có 8 cột chứa giá trị khuyết. Hai cột thiếu nhiều nhất là `company_type` (32,05%) và `company_size` (30,99%).

Một số kết luận EDA quan trọng:

- Tỷ lệ lớp dương khi thiếu `company_size` là 40,59%, cao hơn 22,68 điểm phần trăm so với nhóm có dữ liệu; với `company_type`, chênh lệch là 20,45 điểm phần trăm. Vì vậy missingness được giữ như một tín hiệu.
- `city_development_index` phân tách hai lớp khá rõ; nhóm muốn đổi việc có xu hướng sống ở khu vực có chỉ số phát triển thấp hơn.
- `training_hours` lệch phải mạnh (median 47, mean 65,37, max 336) nhưng phân phối giữa hai lớp gần nhau; pipeline dùng `log1p` để giảm lệch và ảnh hưởng ngoại lai.
- Tỷ lệ muốn đổi việc giảm theo kinh nghiệm: khoảng 45,4% ở nhóm dưới 1 năm và 15,3% ở nhóm trên 20 năm.
- Ứng viên không có kinh nghiệm liên quan và người đang học toàn thời gian có tỷ lệ lớp dương cao hơn các nhóm đối chiếu.

## Thiết kế thí nghiệm và chống rò rỉ

Dữ liệu được chia phân tầng với seed `42`:

| Tập | Số mẫu | Tỷ lệ | Tỷ lệ lớp dương |
|---|---:|---:|---:|
| Train | 13.410 | 70% | 24,929% |
| Validation | 2.874 | 15% | 24,948% |
| Test | 2.874 | 15% | 24,948% |

```text
Raw CSV
  → kiểm tra schema, trùng lặp, missing và EDA
  → stratified split 70% train / 15% validation / 15% test
  → imputation + missing indicators
  → ordinal mapping + one-hot encoding
  → 7 engineered features
  → StandardScaler / MinMaxScaler / PCA 90% / PCA 95%
  → tuning Logistic Regression + Random Forest trên validation
  → chọn candidate theo PR-AUC và threshold theo F1 validation
  → refit estimator truyền thống trên train+validation đã biến đổi
  → đánh giá test đúng một lần và lưu artefact

Nhánh deep learning dùng cùng split
  → vocabulary categorical + median/mean/std chỉ từ train
  → 10 categorical embeddings + 13 numerical/engineered features
  → 3 candidate ResNet + 3 candidate TabM
  → AdamW + cosine schedule + gradient clipping + early stopping
  → khóa winner và threshold riêng cho từng thuật toán trên validation
  → đánh giá test và lưu state_dict/preprocessor/threshold
```

Median, mode, vocabulary, one-hot encoding, scaling, normalization và PCA đều chỉ được fit trên train. Validation dùng để chọn representation, siêu tham số, checkpoint và threshold; test chỉ được mở sau khi khóa toàn bộ quyết định. Các audit trên test chỉ mang tính mô tả, không dùng để chọn lại mô hình.

## Tiền xử lý và đặc trưng

Nhánh truyền thống loại `enrollee_id` và `city`, sau đó:

- điền median cho biến số; mode/median hoặc mức riêng cho biến thứ bậc;
- mã hóa `Unknown` và one-hot cho biến danh nghĩa với `handle_unknown="ignore"`;
- thêm cờ khuyết cho `company_size` và `company_type`;
- chuẩn hóa bằng StandardScaler hoặc MinMaxScaler;
- tùy chọn PCA giữ 90% hoặc 95% phương sai.

Bảy đặc trưng được thiết kế:

| Đặc trưng | Ý nghĩa |
|---|---|
| `training_hours_log1p` | `log(1 + training_hours)` |
| `city_development_index_squared` | Mô tả quan hệ cong của chỉ số phát triển |
| `experience_x_city_development` | Tương tác kinh nghiệm × môi trường thành phố |
| `training_hours_per_experience` | Giờ đào tạo trên `(experience + 1)` |
| `career_mobility_ratio` | Thời gian từ lần đổi việc gần nhất trên `(experience + 1)` |
| `education_x_experience` | Tương tác trình độ × kinh nghiệm |
| `profile_missing_count` | Tổng số trường hồ sơ bị thiếu |

Biểu diễn đầy đủ có **39 đặc trưng**; PCA 90% còn 22 chiều và PCA 95% còn 24 chiều. Bốn representation × ba split được lưu dưới dạng `float32` trong [`features/`](features/).

## Mô hình và chọn cấu hình

### Logistic Regression và Random Forest

Tổng cộng **24 candidate** được đánh giá: 18 Logistic Regression trên Standard/PCA90/PCA95 và 6 Random Forest trên Standard/PCA95. PR-AUC validation là tiêu chí chọn chính; threshold tối đa hóa F1 được khóa sau khi chọn candidate.

| Mô hình | Biểu diễn thắng | Validation PR-AUC | Threshold | Cấu hình thắng |
|---|---|---:|---:|---|
| Logistic Regression | Standard, 39 chiều | 0,5262 | 0,5182 | `C=10`, `class_weight=balanced` |
| Random Forest | Standard, 39 chiều | 0,5372 | 0,6148 | 250 cây, depth 8, leaf 2, `max_features=sqrt`, balanced |

PCA làm giảm hiệu năng của cả hai thuật toán trong thí nghiệm này. MinMax vẫn được hiện thực và xuất artefact để chứng minh pipeline có thể cấu hình, nhưng không nằm trong search chính.

### ResNet và TabM

Nhánh học sâu dùng 10 categorical fields (`city`, `gender`, `relevent_experience`, `enrolled_university`, `education_level`, `major_discipline`, `experience`, `company_size`, `company_type`, `last_new_job`) và 13 numerical/engineered features. ID 0 dành cho giá trị chưa thấy, ID 1 dành cho missing.

- **ResNet**: residual MLP cho dữ liệu bảng; embedding được nối với vector số rồi đi qua ba residual block.
- **TabM**: parameter-efficient ensemble; các member dùng chung trọng số lõi, có rank-one adapter riêng và trung bình xác suất khi suy luận.
- Cả hai dùng batch size 512, AdamW, cosine learning-rate schedule, gradient clipping 1,0 và early stopping theo validation PR-AUC.

| Mô hình | Cấu hình thắng | Tham số | Best epoch | Validation PR-AUC | Threshold |
|---|---|---:|---:|---:|---:|
| ResNet | embed 16, block 256, 3 blocks, dropout 0,25 | 838.833 | 7 | 0,5583 | 0,3254 |
| TabM | embed 12, block 192, 3 blocks, 8 members, dropout 0,20 | 521.149 | 7 | 0,5561 | 0,4168 |

## Kết quả test khóa kín

| Mô hình | Accuracy | Balanced Acc. | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0,7610 | 0,7644 | 0,5139 | **0,7713** | 0,6168 | 0,7963 | 0,5288 |
| Random Forest | 0,7947 | 0,7725 | 0,5692 | 0,7280 | 0,6389 | **0,8120** | 0,5414 |
| ResNet | 0,7944 | **0,7778** | 0,5669 | 0,7448 | **0,6438** | 0,8087 | **0,5657** |
| TabM | **0,7958** | 0,7759 | **0,5702** | 0,7364 | 0,6427 | 0,8093 | 0,5576 |

Confusion matrix tại threshold đã khóa:

| Mô hình | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Logistic Regression | 1.634 | 523 | 164 | 553 |
| Random Forest | 1.762 | 395 | 195 | 522 |
| ResNet | 1.749 | 408 | 183 | 534 |
| TabM | 1.759 | 398 | 189 | 528 |

So với Random Forest, ResNet tăng test PR-AUC `+0,0243` và F1 `+0,0048`; TabM tăng PR-AUC `+0,0161` và F1 `+0,0038`. Không có mô hình nào tốt nhất trên mọi metric: ResNet dẫn đầu PR-AUC/F1/Balanced Accuracy, Random Forest dẫn đầu ROC-AUC, TabM dẫn đầu Accuracy/Precision, còn Logistic Regression có Recall cao nhất.

Các bảng kết quả và hình đánh giá được notebook tái tạo khi chạy **Runtime → Run all**; kết quả tổng hợp chính được trình bày trực tiếp trong README và báo cáo PDF.

## Audit độ tin cậy, calibration và diễn giải

- Repeated stratified 5-fold CV × 3 trên train+validation: Logistic Regression PR-AUC `0,5245 ± 0,0115`; Random Forest `0,5457 ± 0,0148`.
- Paired bootstrap 2.000 lần cho ResNet − Random Forest: CI 95% của chênh lệch PR-AUC xấp xỉ `[0,00001; 0,04734]`; CI của chênh lệch F1 là `[-0,00581; 0,01613]` và chứa 0.
- Brier score: ResNet `0,1353`, TabM `0,1355`, Random Forest `0,1704`. ECE 10-bin tương ứng khoảng `0,0155`, `0,0146`, `0,1807`.
- Ablation validation của Random Forest: bỏ engineered features làm F1 giảm từ `0,6204` xuống `0,6101`; bỏ missing indicators làm PR-AUC giảm từ `0,5372` xuống `0,5295`.
- `city_development_index`, bình phương của nó, tương tác với kinh nghiệm, `company_size_isna` và `profile_missing_count` là các tín hiệu nổi bật. Đây là diễn giải mô hình, không phải quan hệ nhân quả.
- Audit giới tính trên Female (`n=214`), Male (`n=1.929`) và Missing (`n=712`) chỉ mang tính mô tả; quy mô nhóm không cân bằng và tỷ lệ thiếu cao nên chưa đủ để kết luận fairness.

Kết luận thận trọng: ResNet có bằng chứng xếp hạng lớp dương tốt hơn trên test hiện tại, nhưng chưa thể khẳng định ưu thế tổng quát về F1 hoặc trên dữ liệu ngoài mẫu. Nếu ưu tiên đơn giản, tốc độ và khả năng giải thích, Random Forest là lựa chọn thực dụng; nếu ưu tiên PR-AUC và có thêm tài nguyên, ResNet là candidate mạnh hơn trên bộ test này.

## Hạn chế và hướng phát triển

- Kết quả dựa trên một dataset và một test split; chưa có external validation hay nested CV đầy đủ cho mọi model family.
- Nhãn thể hiện **ý định** chuyển việc, không phải việc chuyển việc thực tế; có thể có selection bias.
- Feature importance và missingness có thể phản ánh quy trình thu thập dữ liệu, không chứng minh nguyên nhân.
- Audit fairness hiện tại chưa xét nhóm giao cắt, uncertainty theo nhóm hay tác động pháp lý.
- Hai deep model phức tạp và khó giải thích hơn baseline; CI của chênh lệch F1 so với Random Forest vẫn chứa 0.

Hướng tiếp theo: external validation theo thời gian/đơn vị, nested CV, calibration trên tập độc lập, so sánh CatBoost/Gradient Boosting, fairness theo nhóm giao cắt có CI, và distillation/quantization cho mô hình sâu.

## Cấu trúc dự án

```text
.
├── configs/                       # Hai search space được notebook sử dụng
├── data/
│   └── aug_train.csv              # Bản dữ liệu công khai cho Colab bootstrap
├── features/                      # Các feature/label array .npy bắt buộc
├── modules/                       # Bốn module được notebook import
├── notebooks/
│   └── ML_Tabular_Final.ipynb     # Colab front-end chính
├── reports/
│   └── G3-Report.pdf              # Báo cáo PDF hiện hành
├── .gitignore                     # Allowlist đúng phạm vi bài nộp
├── requirements.txt
└── README.md
```

Các thư mục output như `models/`, hình/CSV trung gian trong `reports/`, cache và mã sinh báo cáo được notebook tạo lại khi cần nhưng không commit.

## Cài đặt và chạy

Yêu cầu Python 3.10 trở lên:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/ML_Tabular_Final.ipynb
```

Trong Jupyter/Colab, chọn **Runtime → Run all**. Trên Colab, notebook clone repository công khai, cài `requirements.txt`, tải CSV công khai nếu thiếu và xác minh SHA-256; không mount Google Drive. Khi chạy local, notebook tự nhận diện project root. Deep-learning search chạy CPU để tăng tính nhất quán; thời gian phụ thuộc phần cứng.

## Suy luận từ dữ liệu thô

Ví dụ với ResNet; đổi checkpoint thành `models/tabm.pt` để dùng TabM:

```python
import pandas as pd

from modules.deep_learning import (
    load_deep_model_artifact,
    predict_deep_from_raw_dataframe,
)

model, preprocessor, checkpoint = load_deep_model_artifact(
    "models/resnet.pt",
    "models/deep_tabular_preprocessor.joblib",
)

raw_rows = pd.read_csv("data/aug_train.csv").drop(columns="target").head(5)
predictions = predict_deep_from_raw_dataframe(
    model, preprocessor, checkpoint, raw_rows
)
print(predictions)
```

Hàm dùng đúng vocabulary/normalization học từ train và threshold đã khóa trên validation. Với TabM, xác suất là trung bình của 8 member nội bộ. Các file trong `models/` được tạo sau khi chạy notebook và không cần commit.

## Tái lập và artifact

- Seed chia dữ liệu: `42`; các feature/label array `.npy` cần nộp nằm trong [`features/`](features/).
- Không gian thí nghiệm nằm trong `configs/model_experiments.json` và `configs/deep_learning_experiments.json`.
- Khi Run All, notebook tự tạo split index, preprocessor, model, manifest, bảng kết quả và hình đánh giá vào các thư mục output cục bộ.
- Môi trường đã ghi nhận khi sinh kết quả: Python 3.14.7, NumPy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1 và PyTorch 2.15.0+cu132.
- Không resample trước split. Nếu mở rộng bằng SMOTE/undersampling, thao tác phải nằm trong từng training fold.

## Tiến độ và minh chứng nhóm

| Sprint | Thời gian | Tiến độ | Điểm đầu việc |
|---:|---|---:|---:|
| 1 | 07–14/09/2026 | 10% | 9 |
| 2 | 19–26/10/2026 | 30% | 12 |
| 3 | 27/10–03/11/2026 | 50% | 12 |
| 4 | 04–16/11/2026 | 75% | 12 |
| 5 | 17–30/11/2026 | 100% | 12 |

Tổng điểm đầu việc ghi nhận trên bảng sprint là 57; đây không phải tỷ lệ đóng góp cá nhân. Biên bản họp và báo cáo tiến độ được lưu tại [thư mục Google Drive của nhóm](https://drive.google.com/drive/u/0/folders/1x6pSko8D-60doEsyp9dVn30dd_EKRXch).

## Báo cáo và liên kết

- Notebook Colab: [![Mở bằng Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Puuffiiii/261-ML-Assignment-G3/blob/main/notebooks/ML_Tabular_Final.ipynb)
- GitHub: [Puuffiiii/261-ML-Assignment-G3](https://github.com/Puuffiiii/261-ML-Assignment-G3)
- Báo cáo tiếng Việt hiện hành, 38 trang: [`reports/G3-Report.pdf`](reports/G3-Report.pdf)

