"""Apply the reproducibility and audit cells to the submitted notebook."""

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "ML_Tabular_Final.ipynb"
COLAB_URL = (
    "https://colab.research.google.com/github/Puuffiiii/"
    "261-ML-Assignment-G3/blob/main/notebooks/ML_Tabular_Final.ipynb"
)
DATA_SHA256 = "8b78da3482032500df40d5359c36ba4a59e28ccd6fc272b050dcf6dee37f114c"


def main() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells[0].source = f"""[![Mở bằng Google Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL})

# Pipeline Machine Learning hoàn chỉnh — HR Analytics

**Mục tiêu:** chạy trọn vẹn EDA, làm sạch, tiền xử lý, tạo đặc trưng, huấn luyện/tuning hai mô hình truyền thống (**Logistic Regression**, **Random Forest**), hai mô hình học sâu (**ResNet**, **TabM**), đánh giá khóa kín và xuất artefact suy luận.

**Nguồn dữ liệu gốc:** [Kaggle — HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists). Bản CSV đi kèm repository công khai được kiểm tra SHA-256 để notebook chạy không cần Kaggle credential và không mount Google Drive.

Notebook tuân thủ ba nguyên tắc:

1. Chia dữ liệu trước khi học mọi thống kê để tránh rò rỉ dữ liệu.
2. Chỉ `fit` imputation, encoding, scaling và PCA trên tập train.
3. Giữ riêng validation cho chọn mô hình/early stopping và test cho đánh giá cuối cùng; các audit test cuối chỉ mang tính mô tả, không được dùng để chọn lại mô hình.
"""

    setup = notebook.cells[2].source
    setup = setup.replace(
        'import warnings\nfrom pathlib import Path',
        'import warnings\nimport hashlib\nimport subprocess\nimport urllib.request\nfrom pathlib import Path',
    )
    setup = setup.replace(
        '    BRANCH = "feature/s1-final"\n    if not Path(REPO_NAME).exists():\n        !git clone -b {BRANCH} {REPO_URL}\n    PROJECT_ROOT = Path(REPO_NAME).resolve()',
        '    RELEASE_REF = "main"  # đổi thành tag submission sau khi chốt thông tin nhóm\n'
        '    if not Path(REPO_NAME).exists():\n'
        '        subprocess.run(["git", "clone", "--depth", "1", "--branch", RELEASE_REF, REPO_URL], check=True)\n'
        '    PROJECT_ROOT = Path(REPO_NAME).resolve()\n'
        '    subprocess.run(\n'
        '        [sys.executable, "-m", "pip", "install", "-q", "-r", str(PROJECT_ROOT / "requirements.txt")],\n'
        '        check=True,\n'
        '    )',
    )
    marker = 'DEEP_CONFIG_PATH = PROJECT_ROOT / "configs" / "deep_learning_experiments.json"\n'
    download = f'''DEEP_CONFIG_PATH = PROJECT_ROOT / "configs" / "deep_learning_experiments.json"
DATA_SHA256 = "{DATA_SHA256}"
DATA_PUBLIC_URL = "https://raw.githubusercontent.com/Puuffiiii/261-ML-Assignment-G3/main/data/aug_train.csv"
if not DATA_PATH.exists():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATA_PUBLIC_URL, DATA_PATH)
actual_data_sha256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
if actual_data_sha256 != DATA_SHA256:
    raise RuntimeError(
        f"Dataset checksum mismatch: expected {{DATA_SHA256}}, got {{actual_data_sha256}}"
    )
'''
    setup = setup.replace(marker, download)
    notebook.cells[2].source = setup

    audit_heading = "## 16. Audit độ tin cậy, calibration, fairness và độ ổn định"
    notebook.cells = [
        cell for cell in notebook.cells if audit_heading not in cell.source
    ]
    notebook.cells.extend(
        [
            nbformat.v4.new_markdown_cell(
                audit_heading
                + """

Các phân tích sau được tạo tự động từ các cấu hình đã khóa:

- repeated 5-fold cross-validation (3 lần lặp) chỉ trên train+validation cho hai mô hình truyền thống;
- ablation trên validation cho engineered features và missing indicators;
- bootstrap 95% CI, calibration và audit theo nhóm giới tính trên dự đoán test đã khóa;
- permutation importance trên validation và checksum phục vụ kiểm toán.

Các kết quả test bổ sung chỉ mô tả độ bất định/rủi ro; **không dùng để chọn lại mô hình hay threshold**.
"""
            ),
            nbformat.v4.new_code_cell(
                """from scripts.generate_audits import main as generate_audits

generate_audits()
display(pd.read_csv(REPORTS_DIR / "repeated_cv_summary.csv").round(4))
display(pd.read_csv(REPORTS_DIR / "feature_ablation_results.csv").round(4))
display(pd.read_csv(REPORTS_DIR / "model_calibration_metrics.csv").round(4))
display(pd.read_csv(REPORTS_DIR / "model_paired_differences.csv").round(4))
display(pd.read_csv(REPORTS_DIR / "gender_subgroup_audit.csv").round(4))
print("Audit artifacts and provenance generated successfully.")"""
            ),
            nbformat.v4.new_markdown_cell(
                """## 17. Model card rút gọn

- **Mục đích:** nghiên cứu/học tập về dự đoán ý định chuyển việc và so sánh pipeline tabular; không dùng để tự động ra quyết định tuyển dụng, sa thải hoặc giữ chân nhân sự.
- **Dữ liệu:** hồ sơ tự khai từ một bộ dữ liệu công khai; nhãn là ý định, không phải hành vi chuyển việc thực tế.
- **Rủi ro:** selection bias, missingness mang tín hiệu quy trình thu thập, sai khác hiệu năng giữa nhóm giới tính và khả năng xác suất chưa calibration.
- **Quản trị:** cần đánh giá ngoài mẫu, rà soát fairness/pháp lý và human review trước mọi ứng dụng thực tế. Không diễn giải feature importance như quan hệ nhân quả.
- **Tái lập:** seed, split indices, cấu hình, checksum dữ liệu/model, package versions và audit policy được lưu trong `features/metadata.json`, model manifest và `reports/audit_provenance.json`.
"""
            ),
        ]
    )
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK)


if __name__ == "__main__":
    main()
