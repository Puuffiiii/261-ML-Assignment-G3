# Features Directory

This directory stores extracted feature matrices and representations exported as NumPy binary files (`.npy`), adhering to Section 5 (Product Requirements) of the Machine Learning course project specification.

## Exported Feature Datasets

All features have been preprocessed using `modules/preprocessing.py` and generated via `notebooks/02_preprocessing_pipeline.ipynb`:

| File Name | Description | Shape | Data Type | Missing/Inf |
|:---|:---|:---:|:---:|:---:|
| `X_train_std.npy` | StandardScaler preprocessed features (Train set) | `(15326, 32)` | `float64` | None |
| `X_test_std.npy` | StandardScaler preprocessed features (Test set) | `(3832, 32)` | `float64` | None |
| `X_train_minmax.npy` | MinMaxScaler [0, 1] preprocessed features (Train set) | `(15326, 32)` | `float64` | None |
| `X_test_minmax.npy` | MinMaxScaler [0, 1] preprocessed features (Test set) | `(3832, 32)` | `float64` | None |
| `X_train_pca90.npy` | PCA retaining 90% explained variance (Train set) | `(15326, 21)` | `float64` | None |
| `X_test_pca90.npy` | PCA retaining 90% explained variance (Test set) | `(3832, 21)` | `float64` | None |
| `X_train_pca95.npy` | PCA retaining 95% explained variance (Train set) | `(15326, 23)` | `float64` | None |
| `X_test_pca95.npy` | PCA retaining 95% explained variance (Test set) | `(3832, 23)` | `float64` | None |
| `y_train.npy` | Stratified target labels (Train set, ~24.9% positive) | `(15326,)` | `int64` | None |
| `y_test.npy` | Stratified target labels (Test set, ~24.9% positive) | `(3832,)` | `int64` | None |
| `feature_names.json` | Explicit list of all 32 encoded feature names & PCA components | JSON | — | — |
| `metadata.json` | Dataset split details, sample sizes, and configuration parameters | JSON | — | — |

## Loading Example

```python
from modules.preprocessing import load_processed_features

# Load all exported feature sets into memory dictionary
features = load_processed_features("features")

X_train_std = features["X_train_std"]
y_train = features["y_train"]
X_test_std = features["X_test_std"]
y_test = features["y_test"]

print(f"Loaded train: {X_train_std.shape}, test: {X_test_std.shape}")
```
