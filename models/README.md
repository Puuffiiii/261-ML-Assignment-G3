# Mô hình đã huấn luyện

Notebook chính sinh artefact cho hai traditional models:

- `logistic_regression.joblib`;
- `random_forest.joblib`;
- `model_manifest.json`.

Và bốn artefact cho hai deep models:

- `resnet.pt`: `state_dict`, cấu hình, cardinality và threshold của ResNet;
- `tabm.pt`: `state_dict`, cấu hình, cardinality và threshold của TabM;
- `deep_tabular_preprocessor.joblib`: vocabulary và thống kê train-only dùng chung;
- `deep_models_manifest.json`: cấu hình, validation metrics và locked-test metrics.

Mỗi file `.joblib` chứa estimator đã refit trên train+validation, preprocessor fit trên train, tên representation, positive class và threshold được chọn bằng F1 validation. Không cần xử lý tay trước khi gọi `predict_from_raw_dataframe`.

```python
import joblib
import pandas as pd

from modules.modeling import predict_from_raw_dataframe

artifact = joblib.load("models/random_forest.joblib")
rows = pd.read_csv("data/aug_train.csv").drop(columns="target").head()
print(predict_from_raw_dataframe(artifact, rows))
```

Joblib phụ thuộc phiên bản Python/scikit-learn. Nếu môi trường khác phiên bản trong `features/metadata.json`, nên chạy lại notebook để tái tạo model.

Deep-learning inference:

```python
import pandas as pd
from modules.deep_learning import (
    load_deep_model_artifact,
    predict_deep_from_raw_dataframe,
)

model, preprocessor, checkpoint = load_deep_model_artifact(
    "models/resnet.pt",  # đổi thành models/tabm.pt để dùng TabM
    "models/deep_tabular_preprocessor.joblib",
)
rows = pd.read_csv("data/aug_train.csv").drop(columns="target").head()
print(predict_deep_from_raw_dataframe(model, preprocessor, checkpoint, rows))
```

Threshold trong từng checkpoint đã được chọn trên validation; không chọn lại threshold bằng dữ liệu test hoặc dữ liệu suy luận.
