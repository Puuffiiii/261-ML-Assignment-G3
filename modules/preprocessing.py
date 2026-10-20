"""
Module: preprocessing.py
Xây dựng pipeline tiền xử lý dữ liệu bảng (Tabular Data Preprocessing Pipeline)
tuân thủ chặt chẽ yêu cầu Bài tập lớn Học máy (Khoa KH&KT Máy tính - ĐHBK ĐHQG-HCM).

Pipeline bao gồm:
1. Data Cleaning & Feature Separation: Loại bỏ ID và cột trùng lặp (city), phân loại đặc trưng.
2. Train/Test Split: Phân tầng (Stratified) theo biến mục tiêu để ngăn ngừa rò rỉ dữ liệu (Data Leakage).
3. Missing Value Handling & MNAR Treatment: Xử lý Missing Not At Random với nhãn 'Unknown' & Missing Indicator.
4. Categorical Encoding:
   - Ordinal Encoding: Ánh xạ có thứ bậc cho education_level, experience, company_size, last_new_job.
   - One-Hot Encoding: Mã hóa danh nghĩa cho gender, relevent_experience, enrolled_university, major_discipline, company_type.
5. Feature Scaling: Tùy biến StandardScaler hoặc MinMaxScaler (với feature_range cấu hình được).
6. Dimensionality Reduction (PCA): Tùy biến giữ lại các mức phương sai tích lũy (90%, 95%, ...) theo yêu cầu đề bài.
7. Feature Persistence: Lưu trữ và nạp các ma trận đặc trưng (.npy) vào thư mục features/.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA


# ==============================================================================
# 1. ĐỊNH NGHĨA ÁNH XẠ THỨ BẬC (ORDINAL MAPPINGS) DỰA TRÊN KẾT QUẢ EDA
# ==============================================================================
ORDINAL_MAPPINGS = {
    'education_level': {
        'Primary School': 0,
        'High School': 1,
        'Graduate': 2,
        'Masters': 3,
        'Phd': 4,
    },
    'company_size': {
        '<10': 0,
        '10/49': 1,
        '50-99': 2,
        '100-500': 3,
        '500-999': 4,
        '1000-4999': 5,
        '5000-9999': 6,
        '10000+': 7,
    },
    'experience': {
        '<1': 0,
        '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
        '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
        '11': 11, '12': 12, '13': 13, '14': 14, '15': 15,
        '16': 16, '17': 17, '18': 18, '19': 19, '20': 20,
        '>20': 21
    },
    'last_new_job': {
        'never': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '>4': 5
    }
}

# Các cột mặc định
NUMERICAL_COLS = ['city_development_index', 'training_hours']
ORDINAL_COLS = ['education_level', 'company_size', 'experience', 'last_new_job']
NOMINAL_COLS = ['gender', 'relevent_experience', 'enrolled_university', 'major_discipline', 'company_type']
DROP_COLS = ['enrollee_id', 'city']
TARGET_COL = 'target'
MNAR_COLS = ['company_size', 'company_type']


# ==============================================================================
# 2. BỘ TIỀN XỬ LÝ ĐẶC TRƯNG CHÍNH (TABULAR PREPROCESSOR)
# ==============================================================================
class TabularPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible preprocessor cho bài toán dự đoán nhảy việc (HR Analytics).
    
    Parameters:
    -----------
    scaling : str or None, default='standard'
        Phương pháp chuẩn hóa: 'standard' (StandardScaler), 'minmax' (MinMaxScaler), hoặc None.
    feature_range : tuple of (float, float), default=(0, 1)
        Phạm vi co giãn đặc trưng khi sử dụng scaling='minmax' (Ví dụ: (0, 1) hoặc (-1, 1)).
    pca_variance : float, int or None, default=None
        Tỷ lệ phương sai tích lũy cần giữ lại khi áp dụng PCA (ví dụ: 0.90 hoặc 0.95),
        hoặc số lượng thành phần chính cố định (int), hoặc None nếu không giảm chiều.
    add_missing_indicators : bool, default=True
        Thêm cột cờ chỉ báo dữ liệu khuyết cho các thuộc tính MNAR (company_size, company_type).
    random_state : int, default=42
        Random seed cho tính tái lập (Reproducibility).
    """

    def __init__(
        self,
        scaling: Optional[str] = 'standard',
        feature_range: Tuple[float, float] = (0, 1),
        pca_variance: Optional[Union[float, int]] = None,
        add_missing_indicators: bool = True,
        random_state: int = 42
    ):
        self.scaling = scaling
        self.feature_range = feature_range
        self.pca_variance = pca_variance
        self.add_missing_indicators = add_missing_indicators
        self.random_state = random_state

        # State lưu lại sau khi fit
        self.num_imputer_medians_ = {}
        self.ord_imputer_modes_ = {}
        self.ohe_ = None
        self.scaler_ = None
        self.pca_ = None
        self.feature_names_out_ = []
        self.raw_feature_names_ = []

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit toàn bộ các thành phần tiền xử lý trên tập dữ liệu X (Train set).
        Tuyệt đối không được fit trên Test set để chống Data Leakage.
        """
        X = X.copy()

        # 1. Học giá trị điền khuyết cho Numerical features
        for col in NUMERICAL_COLS:
            if col in X.columns:
                self.num_imputer_medians_[col] = X[col].median()

        # 2. Học giá trị điền khuyết cho Ordinal features
        for col in ORDINAL_COLS:
            if col in X.columns:
                mode_val = X[col].mode().dropna()
                self.ord_imputer_modes_[col] = mode_val.iloc[0] if len(mode_val) > 0 else list(ORDINAL_MAPPINGS[col].keys())[0]

        # 3. Fit One-Hot Encoder cho Nominal features
        X_nom_imputed = self._impute_nominal(X)
        self.ohe_ = OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False,
            drop=None  # Giữ toàn bộ để phục vụ phân tích hoặc MLP/Linear regularization
        )
        self.ohe_.fit(X_nom_imputed[NOMINAL_COLS])

        # 4. Tạo ma trận đặc trưng trước khi Scale để Fit Scaler
        X_transformed_df = self._transform_without_scaling_pca(X)
        self.raw_feature_names_ = list(X_transformed_df.columns)

        # 5. Fit Scaler
        if self.scaling == 'standard':
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X_transformed_df.values)
        elif self.scaling == 'minmax':
            self.scaler_ = MinMaxScaler(feature_range=self.feature_range)
            X_scaled = self.scaler_.fit_transform(X_transformed_df.values)
        elif self.scaling is None:
            self.scaler_ = None
            X_scaled = X_transformed_df.values
        else:
            raise ValueError(f"Phương pháp scaling không hợp lệ: '{self.scaling}'. Chọn 'standard', 'minmax', hoặc None.")

        # 6. Fit PCA (nếu có cấu hình)
        if self.pca_variance is not None:
            self.pca_ = PCA(n_components=self.pca_variance, random_state=self.random_state)
            self.pca_.fit(X_scaled)
            n_comps = self.pca_.n_components_
            self.feature_names_out_ = [f"PC_{i+1}" for i in range(n_comps)]
        else:
            self.pca_ = None
            self.feature_names_out_ = self.raw_feature_names_

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Áp dụng toàn bộ quy trình biến đổi đã học lên dữ liệu X (Train hoặc Test).
        """
        X = X.copy()
        X_transformed_df = self._transform_without_scaling_pca(X)

        # Scale
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X_transformed_df.values)
        else:
            X_scaled = X_transformed_df.values

        # PCA
        if self.pca_ is not None:
            X_out = self.pca_.transform(X_scaled)
        else:
            X_out = X_scaled

        return X_out

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def _impute_nominal(self, X: pd.DataFrame) -> pd.DataFrame:
        """Điền khuyết cho các biến danh nghĩa theo chiến lược MNAR ('Unknown')."""
        X_nom = X[NOMINAL_COLS].copy()
        for col in NOMINAL_COLS:
            X_nom[col] = X_nom[col].fillna('Unknown').astype(str)
        return X_nom

    def _transform_without_scaling_pca(self, X: pd.DataFrame) -> pd.DataFrame:
        """Thực hiện Imputation, Ordinal Mapping, Missing Indicator và OHE tạo DataFrame thô."""
        result_dfs = []

        # A. Numerical Features
        X_num = pd.DataFrame(index=X.index)
        for col in NUMERICAL_COLS:
            median_val = self.num_imputer_medians_.get(col, 0.0)
            X_num[col] = X[col].fillna(median_val).astype(float)
        result_dfs.append(X_num)

        # B. Ordinal Features (Ánh xạ số nguyên có thứ bậc)
        X_ord = pd.DataFrame(index=X.index)
        for col in ORDINAL_COLS:
            mode_val = self.ord_imputer_modes_.get(col, list(ORDINAL_MAPPINGS[col].keys())[0])
            mapped_series = X[col].fillna(mode_val).map(ORDINAL_MAPPINGS[col])
            # Nếu có giá trị ngoài bảng mapping thì điền theo mode
            fallback_val = ORDINAL_MAPPINGS[col][mode_val]
            X_ord[col] = mapped_series.fillna(fallback_val).astype(float)
        result_dfs.append(X_ord)

        # C. Missing Indicators (Bảo tồn tín hiệu MNAR cho company_size và company_type)
        if self.add_missing_indicators:
            X_mnar = pd.DataFrame(index=X.index)
            for col in MNAR_COLS:
                if col in X.columns:
                    X_mnar[f"{col}_isna"] = X[col].isna().astype(float)
            result_dfs.append(X_mnar)

        # D. Nominal Features (One-Hot Encoded)
        X_nom_imputed = self._impute_nominal(X)
        nom_encoded = self.ohe_.transform(X_nom_imputed[NOMINAL_COLS])
        ohe_col_names = self.ohe_.get_feature_names_out(NOMINAL_COLS)
        X_ohe = pd.DataFrame(nom_encoded, columns=ohe_col_names, index=X.index)
        result_dfs.append(X_ohe)

        # Ghép toàn bộ lại thành 1 DataFrame
        return pd.concat(result_dfs, axis=1)

    def get_feature_names_out(self) -> List[str]:
        """Trả về danh sách tên các đặc trưng sau khi xử lý (hoặc tên Principal Components)."""
        return self.feature_names_out_

    def get_explained_variance_ratio(self) -> Optional[np.ndarray]:
        """Trả về tỷ lệ phương sai giải thích nếu đã chạy PCA."""
        if self.pca_ is not None:
            return self.pca_.explained_variance_ratio_
        return None

    def get_cumulative_explained_variance(self) -> Optional[float]:
        """Trả về tổng phương sai tích lũy được giải thích nếu đã chạy PCA."""
        if self.pca_ is not None:
            return float(np.sum(self.pca_.explained_variance_ratio_))
        return None


# ==============================================================================
# 3. CÁC HÀM TIỆN ÍCH DÀNH CHO WORKFLOW & COLAB
# ==============================================================================
def load_raw_dataset(csv_path: str = "data/aug_train.csv") -> pd.DataFrame:
    """Nạp tập dữ liệu thô và kiểm tra tính toàn vẹn."""
    if not os.path.exists(csv_path):
        # Thử đường dẫn từ thư mục notebooks (lùi 1 cấp)
        alt_path = os.path.join("..", csv_path)
        if os.path.exists(alt_path):
            csv_path = alt_path
        else:
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại '{csv_path}' hoặc '{alt_path}'")
    df = pd.read_csv(csv_path)
    return df


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    drop_unnecessary: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Tách dữ liệu thành Train và Test set có phân tầng (Stratified) theo biến target.
    Loại bỏ enrollee_id và city theo khuyến nghị EDA.
    """
    df = df.copy()
    if TARGET_COL not in df.columns:
        raise ValueError(f"Không tìm thấy cột mục tiêu '{TARGET_COL}' trong DataFrame.")

    # Tách X và y
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])

    if drop_unnecessary:
        cols_to_drop = [c for c in DROP_COLS if c in X.columns]
        X = X.drop(columns=cols_to_drop)

    # Phân tầng theo y để bảo toàn tỷ lệ 24.9% positive class
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def save_processed_features(
    output_dir: str,
    feature_sets: Dict[str, np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names_dict: Optional[Dict[str, List[str]]] = None,
    metadata: Optional[Dict] = None
):
    """
    Lưu trữ các ma trận đặc trưng (.npy) và siêu dữ liệu (.json) vào thư mục features/
    tuân thủ Mục 5 trang 5 đề bài PDF.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Lưu biến mục tiêu
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    # Lưu từng tập đặc trưng
    for name, arr in feature_sets.items():
        filename = f"{name}.npy"
        np.save(os.path.join(output_dir, filename), arr)
        print(f"-> Đã lưu: {os.path.join(output_dir, filename)} [shape: {arr.shape}]")

    # Lưu tên cột
    if feature_names_dict is not None:
        names_path = os.path.join(output_dir, "feature_names.json")
        with open(names_path, "w", encoding="utf-8") as f:
            json.dump(feature_names_dict, f, indent=2, ensure_ascii=False)
        print(f"-> Đã lưu: {names_path}")

    # Lưu metadata tổng quan
    if metadata is not None:
        meta_path = os.path.join(output_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"-> Đã lưu: {meta_path}")


def load_processed_features(features_dir: str = "features") -> Dict[str, np.ndarray]:
    """
    Nạp toàn bộ các file đặc trưng (.npy) từ thư mục features/ lên bộ nhớ.
    """
    if not os.path.exists(features_dir):
        alt = os.path.join("..", features_dir)
        if os.path.exists(alt):
            features_dir = alt
        else:
            raise FileNotFoundError(f"Thư mục '{features_dir}' không tồn tại.")

    data = {}
    for f in os.listdir(features_dir):
        if f.endswith(".npy"):
            key = f[:-4]
            data[key] = np.load(os.path.join(features_dir, f))
    return data
