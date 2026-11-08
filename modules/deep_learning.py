"""Leakage-safe ResNet and TabM pipelines for the HR Analytics dataset.

The module intentionally consumes raw split DataFrames. Categorical vocabularies
and numerical normalization statistics are learned from the training split only.
Hyperparameters, early stopping checkpoints, and decision thresholds are selected
on validation data; the held-out test set is evaluated only after those choices
are locked.

Two deep-learning algorithms are implemented:

* ``ResNet``: a strong residual-MLP baseline for tabular data.
* ``TabM``: a parameter-efficient ensemble of MLPs implemented with rank-one
  BatchEnsemble adapters. One shared network produces ``k`` member predictions,
  which are averaged at inference time.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from modules.modeling import binary_classification_metrics, select_f1_threshold
from modules.preprocessing import ORDINAL_MAPPINGS


SUPPORTED_DEEP_MODELS = ("resnet", "tabm")
DEEP_MODEL_LABELS = {"resnet": "ResNet", "tabm": "TabM"}

DEEP_CATEGORICAL_COLS = [
    "city",
    "gender",
    "relevent_experience",
    "enrolled_university",
    "education_level",
    "major_discipline",
    "experience",
    "company_size",
    "company_type",
    "last_new_job",
]

DEEP_NUMERICAL_FEATURES = [
    "city_development_index",
    "training_hours",
    "training_hours_log1p",
    "city_development_index_squared",
    "education_level_ordinal",
    "experience_ordinal",
    "company_size_ordinal",
    "last_new_job_ordinal",
    "experience_x_city_development",
    "training_hours_per_experience",
    "career_mobility_ratio",
    "education_x_experience",
    "profile_missing_count",
]


def set_deep_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible CPU/GPU training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


class DeepTabularPreprocessor(BaseEstimator, TransformerMixin):
    """Encode categories as IDs and standardize engineered numerical features.

    Vocabulary, medians, means, and standard deviations are learned from train
    only. Category ID 0 is reserved for unseen values and ID 1 for missing
    values so inference remains safe when a new category appears.
    """

    unknown_token = "__UNKNOWN__"
    missing_token = "__MISSING__"

    def __init__(self):
        self.category_maps_: Dict[str, Dict[str, int]] = {}
        self.numeric_medians_: Dict[str, float] = {}
        self.numeric_means_: Dict[str, float] = {}
        self.numeric_stds_: Dict[str, float] = {}
        self.cardinalities_: list[int] = []

    @staticmethod
    def _validate_columns(X: pd.DataFrame) -> None:
        required = set(DEEP_CATEGORICAL_COLS + ["city_development_index", "training_hours"])
        missing = sorted(required.difference(X.columns))
        if missing:
            raise ValueError(f"Missing deep-learning input columns: {missing}")

    @staticmethod
    def _category_series(series: pd.Series) -> pd.Series:
        return series.fillna(DeepTabularPreprocessor.missing_token).astype(str)

    def _engineer_numerical(self, X: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=X.index)
        cdi = pd.to_numeric(X["city_development_index"], errors="coerce")
        hours = pd.to_numeric(X["training_hours"], errors="coerce").clip(lower=0)

        ordinal_values: Dict[str, pd.Series] = {}
        for column in ["education_level", "experience", "company_size", "last_new_job"]:
            ordinal_values[column] = X[column].map(ORDINAL_MAPPINGS[column]).astype(float)

        experience = ordinal_values["experience"]
        education = ordinal_values["education_level"]
        company_size = ordinal_values["company_size"]
        last_job = ordinal_values["last_new_job"]

        output["city_development_index"] = cdi
        output["training_hours"] = hours
        output["training_hours_log1p"] = np.log1p(hours)
        output["city_development_index_squared"] = cdi.pow(2)
        output["education_level_ordinal"] = education
        output["experience_ordinal"] = experience
        output["company_size_ordinal"] = company_size
        output["last_new_job_ordinal"] = last_job
        output["experience_x_city_development"] = experience * cdi
        output["training_hours_per_experience"] = hours / (experience + 1.0)
        output["career_mobility_ratio"] = last_job / (experience + 1.0)
        output["education_x_experience"] = education * experience
        profile_columns = DEEP_CATEGORICAL_COLS + [
            "city_development_index",
            "training_hours",
        ]
        output["profile_missing_count"] = X[profile_columns].isna().sum(axis=1).astype(float)
        return output.replace([np.inf, -np.inf], np.nan)

    def fit(self, X: pd.DataFrame, y=None):
        self._validate_columns(X)
        self.category_maps_ = {}
        for column in DEEP_CATEGORICAL_COLS:
            values = self._category_series(X[column])
            observed = sorted(
                value for value in values.unique() if value != self.missing_token
            )
            mapping = {self.unknown_token: 0, self.missing_token: 1}
            mapping.update({value: index + 2 for index, value in enumerate(observed)})
            self.category_maps_[column] = mapping
        self.cardinalities_ = [
            len(self.category_maps_[column]) for column in DEEP_CATEGORICAL_COLS
        ]

        numerical = self._engineer_numerical(X)
        self.numeric_medians_ = {
            column: float(numerical[column].median())
            if numerical[column].notna().any()
            else 0.0
            for column in DEEP_NUMERICAL_FEATURES
        }
        imputed = numerical.fillna(self.numeric_medians_)
        self.numeric_means_ = {
            column: float(imputed[column].mean()) for column in DEEP_NUMERICAL_FEATURES
        }
        self.numeric_stds_ = {}
        for column in DEEP_NUMERICAL_FEATURES:
            std = float(imputed[column].std(ddof=0))
            self.numeric_stds_[column] = std if std > 1e-8 else 1.0
        return self

    def transform(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.category_maps_:
            raise RuntimeError("DeepTabularPreprocessor must be fit before transform().")
        self._validate_columns(X)

        categorical = np.zeros((len(X), len(DEEP_CATEGORICAL_COLS)), dtype=np.int64)
        for column_index, column in enumerate(DEEP_CATEGORICAL_COLS):
            mapping = self.category_maps_[column]
            categorical[:, column_index] = (
                self._category_series(X[column])
                .map(mapping)
                .fillna(0)
                .to_numpy(dtype=np.int64)
            )

        numerical = self._engineer_numerical(X).fillna(self.numeric_medians_)
        for column in DEEP_NUMERICAL_FEATURES:
            numerical[column] = (
                numerical[column] - self.numeric_means_[column]
            ) / self.numeric_stds_[column]
        numerical_array = numerical[DEEP_NUMERICAL_FEATURES].to_numpy(dtype=np.float32)
        if not np.isfinite(numerical_array).all():
            raise ValueError("Deep numerical features contain NaN or infinite values.")
        return categorical, numerical_array

    def fit_transform(self, X: pd.DataFrame, y=None) -> Tuple[np.ndarray, np.ndarray]:
        return self.fit(X, y).transform(X)


class CategoricalNumericalEncoder(nn.Module):
    """Concatenate learned categorical embeddings and normalized numerics."""

    def __init__(
        self,
        cardinalities: Sequence[int],
        n_numerical: int,
        embedding_dim: int = 12,
        embedding_dropout: float = 0.0,
    ):
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive.")
        self.cardinalities = list(cardinalities)
        self.n_numerical = int(n_numerical)
        self.embedding_dim = int(embedding_dim)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, embedding_dim) for cardinality in cardinalities]
        )
        self.embedding_dropout = nn.Dropout(float(embedding_dropout))
        self.output_dim = len(cardinalities) * embedding_dim + n_numerical
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, std=0.02)

    def forward(self, categorical: torch.Tensor, numerical: torch.Tensor) -> torch.Tensor:
        embedded = torch.cat(
            [
                embedding(categorical[:, index])
                for index, embedding in enumerate(self.embeddings)
            ],
            dim=1,
        )
        return torch.cat([self.embedding_dropout(embedded), numerical], dim=1)


class ResidualMLPBlock(nn.Module):
    """Pre-normalized residual MLP block for the ResNet baseline."""

    def __init__(
        self,
        d_block: int,
        d_hidden: int,
        dropout: float,
        residual_dropout: float,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_block)
        self.linear1 = nn.Linear(d_block, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_block)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.norm(x)
        residual = self.linear1(residual)
        residual = self.activation(residual)
        residual = self.dropout(residual)
        residual = self.linear2(residual)
        residual = self.residual_dropout(residual)
        return x + residual


class TabularResNet(nn.Module):
    """Strong residual-MLP baseline for mixed tabular features."""

    algorithm = "resnet"

    def __init__(
        self,
        cardinalities: Sequence[int],
        n_numerical: int,
        embedding_dim: int = 12,
        embedding_dropout: float = 0.0,
        d_block: int = 192,
        n_blocks: int = 3,
        d_hidden_multiplier: float = 2.0,
        dropout: float = 0.2,
        residual_dropout: float = 0.05,
    ):
        super().__init__()
        self.encoder = CategoricalNumericalEncoder(
            cardinalities, n_numerical, embedding_dim, embedding_dropout
        )
        d_hidden = max(d_block, int(round(d_block * d_hidden_multiplier)))
        self.input_projection = nn.Linear(self.encoder.output_dim, d_block)
        self.blocks = nn.ModuleList(
            [
                ResidualMLPBlock(
                    d_block=d_block,
                    d_hidden=d_hidden,
                    dropout=dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_block),
            nn.ReLU(),
            nn.Linear(d_block, 1),
        )

    def forward(self, categorical: torch.Tensor, numerical: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(self.encoder(categorical, numerical))
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


class BatchEnsembleLinear(nn.Module):
    """Shared linear layer with trainable rank-one adapters for each member.

    Input has shape ``[batch, members, in_features]`` and output has shape
    ``[batch, members, out_features]``. Shared weights provide parameter
    efficiency, while member-specific input/output scales create diverse MLPs.
    """

    def __init__(self, in_features: int, out_features: int, member_count: int):
        super().__init__()
        if member_count < 2:
            raise ValueError("TabM member_count must be at least 2.")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.member_count = int(member_count)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.input_scale = nn.Parameter(torch.empty(member_count, in_features))
        self.output_scale = nn.Parameter(torch.empty(member_count, out_features))
        self.member_bias = nn.Parameter(torch.zeros(member_count, out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.bias, -bound, bound)
        with torch.no_grad():
            input_signs = torch.randint(0, 2, self.input_scale.shape, dtype=torch.float32)
            output_signs = torch.randint(0, 2, self.output_scale.shape, dtype=torch.float32)
            self.input_scale.copy_(input_signs.mul_(2).sub_(1))
            self.output_scale.copy_(output_signs.mul_(2).sub_(1))
            self.input_scale.add_(0.05 * torch.randn_like(self.input_scale))
            self.output_scale.add_(0.05 * torch.randn_like(self.output_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.member_count:
            raise ValueError(
                f"Expected [batch, {self.member_count}, {self.in_features}], got {tuple(x.shape)}"
            )
        scaled_input = x * self.input_scale.unsqueeze(0)
        output = torch.einsum("bki,oi->bko", scaled_input, self.weight)
        output = output + self.bias.view(1, 1, -1)
        return output * self.output_scale.unsqueeze(0) + self.member_bias.unsqueeze(0)


class TabMResidualBlock(nn.Module):
    """Residual block operating on all implicit TabM members in parallel."""

    def __init__(
        self,
        d_block: int,
        d_hidden: int,
        member_count: int,
        dropout: float,
        residual_dropout: float,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_block)
        self.linear1 = BatchEnsembleLinear(d_block, d_hidden, member_count)
        self.linear2 = BatchEnsembleLinear(d_hidden, d_block, member_count)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.norm(x)
        residual = self.linear1(residual)
        residual = self.activation(residual)
        residual = self.dropout(residual)
        residual = self.linear2(residual)
        residual = self.residual_dropout(residual)
        return x + residual


class TabM(nn.Module):
    """Parameter-efficient ensemble of residual MLPs for tabular data."""

    algorithm = "tabm"

    def __init__(
        self,
        cardinalities: Sequence[int],
        n_numerical: int,
        embedding_dim: int = 12,
        embedding_dropout: float = 0.0,
        member_count: int = 8,
        d_block: int = 160,
        n_blocks: int = 2,
        d_hidden_multiplier: float = 2.0,
        dropout: float = 0.15,
        residual_dropout: float = 0.05,
    ):
        super().__init__()
        self.member_count = int(member_count)
        self.encoder = CategoricalNumericalEncoder(
            cardinalities, n_numerical, embedding_dim, embedding_dropout
        )
        d_hidden = max(d_block, int(round(d_block * d_hidden_multiplier)))
        self.input_projection = BatchEnsembleLinear(
            self.encoder.output_dim, d_block, self.member_count
        )
        self.blocks = nn.ModuleList(
            [
                TabMResidualBlock(
                    d_block=d_block,
                    d_hidden=d_hidden,
                    member_count=self.member_count,
                    dropout=dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        self.head_norm = nn.LayerNorm(d_block)
        self.head_activation = nn.ReLU()
        self.head = BatchEnsembleLinear(d_block, 1, self.member_count)

    def forward(self, categorical: torch.Tensor, numerical: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(categorical, numerical)
        x = encoded.unsqueeze(1).expand(-1, self.member_count, -1)
        x = self.input_projection(x)
        for block in self.blocks:
            x = block(x)
        x = self.head_activation(self.head_norm(x))
        return self.head(x).squeeze(-1)


def build_deep_model(
    algorithm: str,
    cardinalities: Sequence[int],
    n_numerical: int,
    config: Mapping[str, Any],
) -> nn.Module:
    """Build one supported deep model from a serializable configuration."""
    if algorithm not in SUPPORTED_DEEP_MODELS:
        raise ValueError(f"Unsupported deep model '{algorithm}'. Choose from {SUPPORTED_DEEP_MODELS}.")
    common_keys = {
        "embedding_dim",
        "embedding_dropout",
        "d_block",
        "n_blocks",
        "d_hidden_multiplier",
        "dropout",
        "residual_dropout",
    }
    architecture = {key: value for key, value in config.items() if key in common_keys}
    if algorithm == "resnet":
        return TabularResNet(cardinalities, n_numerical, **architecture)
    architecture["member_count"] = int(config.get("member_count", 8))
    return TabM(cardinalities, n_numerical, **architecture)


def _make_loader(
    categorical: np.ndarray,
    numerical: np.ndarray,
    labels: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    tensors = [torch.from_numpy(categorical), torch.from_numpy(numerical)]
    if labels is not None:
        tensors.append(torch.from_numpy(np.asarray(labels, dtype=np.float32)))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


def _mean_probability_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    return probabilities.mean(dim=1) if probabilities.ndim == 2 else probabilities


def _loss_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    if logits.ndim == 2:
        labels = labels.unsqueeze(1).expand_as(logits)
    return criterion(logits, labels)


@torch.no_grad()
def predict_deep_model(
    model: nn.Module,
    categorical: np.ndarray,
    numerical: np.ndarray,
    batch_size: int = 1024,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Predict positive-class probabilities; TabM members are averaged."""
    device = torch.device(device)
    model.eval()
    loader = _make_loader(categorical, numerical, None, batch_size, False, 0)
    outputs = []
    for categorical_batch, numerical_batch in loader:
        logits = model(categorical_batch.to(device), numerical_batch.to(device))
        outputs.append(_mean_probability_from_logits(logits).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def train_deep_model(
    algorithm: str,
    train_data: Tuple[np.ndarray, np.ndarray],
    y_train: Sequence[int],
    validation_data: Tuple[np.ndarray, np.ndarray],
    y_validation: Sequence[int],
    cardinalities: Sequence[int],
    config: Mapping[str, Any],
    device: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Train one ResNet/TabM candidate with validation-only early stopping."""
    seed = int(config.get("seed", 42))
    set_deep_seed(seed)
    device = torch.device(device)
    categorical_train, numerical_train = train_data
    categorical_validation, numerical_validation = validation_data
    model = build_deep_model(
        algorithm, cardinalities, numerical_train.shape[1], config
    ).to(device)

    batch_size = int(config.get("batch_size", 512))
    train_loader = _make_loader(
        categorical_train,
        numerical_train,
        np.asarray(y_train),
        batch_size,
        True,
        seed,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    max_epochs = int(config.get("max_epochs", 100))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=float(config.get("minimum_learning_rate", 1e-6))
    )
    pos_weight = torch.tensor(float(config.get("pos_weight", 1.0)), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_epoch = 0
    best_pr_auc = -np.inf
    best_roc_auc = -np.inf
    patience = int(config.get("patience", 15))
    min_delta = float(config.get("min_delta", 1e-5))
    epochs_without_improvement = 0
    history = []
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        cumulative_loss = 0.0
        examples = 0
        for categorical_batch, numerical_batch, labels_batch in train_loader:
            categorical_batch = categorical_batch.to(device)
            numerical_batch = numerical_batch.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(categorical_batch, numerical_batch)
            loss = _loss_from_logits(logits, labels_batch, criterion)
            loss.backward()
            gradient_clip = float(config.get("gradient_clip", 1.0))
            if gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            cumulative_loss += float(loss.detach().cpu()) * len(labels_batch)
            examples += len(labels_batch)
        scheduler.step()

        validation_probability = predict_deep_model(
            model,
            categorical_validation,
            numerical_validation,
            batch_size=batch_size * 2,
            device=device,
        )
        validation_pr_auc = average_precision_score(
            y_validation, validation_probability
        )
        validation_roc_auc = roc_auc_score(y_validation, validation_probability)
        history.append(
            {
                "epoch": epoch,
                "train_loss": cumulative_loss / max(examples, 1),
                "validation_pr_auc": float(validation_pr_auc),
                "validation_roc_auc": float(validation_roc_auc),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

        improved = validation_pr_auc > best_pr_auc + min_delta or (
            abs(validation_pr_auc - best_pr_auc) <= min_delta
            and validation_roc_auc > best_roc_auc
        )
        if improved:
            best_pr_auc = float(validation_pr_auc)
            best_roc_auc = float(validation_roc_auc)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError(f"{DEEP_MODEL_LABELS[algorithm]} training produced no checkpoint.")
    model.load_state_dict(best_state)
    validation_probability = predict_deep_model(
        model,
        categorical_validation,
        numerical_validation,
        batch_size=batch_size * 2,
        device=device,
    )
    threshold, _ = select_f1_threshold(y_validation, validation_probability)
    metrics = binary_classification_metrics(
        y_validation, validation_probability, threshold
    )
    return {
        "algorithm": algorithm,
        "model": model.cpu(),
        "state_dict": {
            key: value.detach().cpu() for key, value in best_state.items()
        },
        "config": dict(config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "threshold": threshold,
        "validation_probabilities": validation_probability,
        "validation_metrics": metrics,
        "history": pd.DataFrame(history),
        "fit_seconds": time.perf_counter() - started,
    }


def run_deep_validation_experiments(
    train_data: Tuple[np.ndarray, np.ndarray],
    y_train: Sequence[int],
    validation_data: Tuple[np.ndarray, np.ndarray],
    y_validation: Sequence[int],
    cardinalities: Sequence[int],
    model_candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    device: str | torch.device = "cpu",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Train configured ResNet and TabM candidates on validation."""
    unknown = sorted(set(model_candidates) - set(SUPPORTED_DEEP_MODELS))
    if unknown:
        raise ValueError(f"Unsupported deep model(s): {unknown}")
    rows = []
    fitted: Dict[str, Dict[str, Any]] = {}
    total = sum(len(candidates) for candidates in model_candidates.values())
    completed = 0
    for algorithm in SUPPORTED_DEEP_MODELS:
        candidates = model_candidates.get(algorithm, [])
        if not candidates:
            raise ValueError(f"Deep model '{algorithm}' has no candidates.")
        for index, config in enumerate(candidates, start=1):
            completed += 1
            candidate_id = f"{algorithm}__{index:02d}"
            if verbose:
                print(
                    f"[{completed}/{total}] Training {candidate_id} "
                    f"({DEEP_MODEL_LABELS[algorithm]}) ..."
                )
            result = train_deep_model(
                algorithm,
                train_data,
                y_train,
                validation_data,
                y_validation,
                cardinalities,
                config,
                device,
            )
            metrics = result["validation_metrics"]
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "algorithm": DEEP_MODEL_LABELS[algorithm],
                    "algorithm_key": algorithm,
                    "parameters": json.dumps(dict(config), sort_keys=True),
                    "parameter_count": result["parameter_count"],
                    "best_epoch": result["best_epoch"],
                    "fit_seconds": result["fit_seconds"],
                    "validation_threshold": result["threshold"],
                    "validation_accuracy": metrics["accuracy"],
                    "validation_balanced_accuracy": metrics["balanced_accuracy"],
                    "validation_precision": metrics["precision"],
                    "validation_recall": metrics["recall"],
                    "validation_f1": metrics["f1"],
                    "validation_roc_auc": metrics["roc_auc"],
                    "validation_pr_auc": metrics["pr_auc"],
                }
            )
            fitted[candidate_id] = result
            if verbose:
                print(
                    f"  best_epoch={result['best_epoch']} | "
                    f"PR-AUC={metrics['pr_auc']:.4f} | "
                    f"ROC-AUC={metrics['roc_auc']:.4f} | F1={metrics['f1']:.4f}"
                )
    results = pd.DataFrame(rows).sort_values(
        ["algorithm_key", "validation_pr_auc", "validation_roc_auc", "validation_f1"],
        ascending=[True, False, False, False],
    )
    return results.reset_index(drop=True), fitted


def select_best_deep_candidates(validation_results: pd.DataFrame) -> pd.DataFrame:
    """Lock the best validation candidate independently for each algorithm."""
    required = {
        "candidate_id",
        "algorithm",
        "algorithm_key",
        "validation_pr_auc",
        "validation_roc_auc",
        "validation_f1",
    }
    missing = sorted(required.difference(validation_results.columns))
    if missing:
        raise ValueError(f"Validation results missing columns: {missing}")
    ordered = validation_results.sort_values(
        ["algorithm_key", "validation_pr_auc", "validation_roc_auc", "validation_f1"],
        ascending=[True, False, False, False],
    )
    return ordered.groupby("algorithm_key", as_index=False, sort=True).head(1).reset_index(drop=True)


def evaluate_deep_models_on_test(
    best_candidates: pd.DataFrame,
    fitted_candidates: Mapping[str, Mapping[str, Any]],
    test_data: Tuple[np.ndarray, np.ndarray],
    y_test: Sequence[int],
    device: str | torch.device = "cpu",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Evaluate each validation-locked algorithm winner once on test."""
    categorical_test, numerical_test = test_data
    rows = []
    final_models: Dict[str, Dict[str, Any]] = {}
    for row in best_candidates.sort_values("algorithm_key").to_dict(orient="records"):
        candidate_id = row["candidate_id"]
        result = dict(fitted_candidates[candidate_id])
        algorithm = result["algorithm"]
        model = result["model"].to(device)
        probabilities = predict_deep_model(
            model,
            categorical_test,
            numerical_test,
            batch_size=int(result["config"].get("batch_size", 512)) * 2,
            device=device,
        )
        metrics = binary_classification_metrics(
            y_test, probabilities, result["threshold"]
        )
        result["model"] = model.cpu()
        result["test_probabilities"] = probabilities
        result["test_metrics"] = metrics
        final_models[algorithm] = result
        rows.append(
            {
                "algorithm": DEEP_MODEL_LABELS[algorithm],
                "algorithm_key": algorithm,
                "candidate_id": candidate_id,
                "parameters": json.dumps(result["config"], sort_keys=True),
                "parameter_count": result["parameter_count"],
                "best_epoch": result["best_epoch"],
                **{f"test_{key}": value for key, value in metrics.items()},
            }
        )
    return pd.DataFrame(rows), final_models


def save_deep_model_artifacts(
    output_dir: str | Path,
    preprocessor: DeepTabularPreprocessor,
    final_models: Mapping[str, Mapping[str, Any]],
) -> Dict[str, str]:
    """Save portable checkpoints, one shared preprocessor, and a manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = output_dir / "deep_tabular_preprocessor.joblib"
    manifest_path = output_dir / "deep_models_manifest.json"
    joblib.dump(preprocessor, preprocessor_path)

    paths: Dict[str, str] = {"preprocessor": str(preprocessor_path)}
    manifest_models: Dict[str, Any] = {}
    for algorithm in SUPPORTED_DEEP_MODELS:
        if algorithm not in final_models:
            continue
        result = final_models[algorithm]
        checkpoint_path = output_dir / f"{algorithm}.pt"
        checkpoint = {
            "algorithm": algorithm,
            "state_dict": result["state_dict"],
            "cardinalities": list(preprocessor.cardinalities_),
            "n_numerical": len(DEEP_NUMERICAL_FEATURES),
            "config": result["config"],
            "threshold": float(result["threshold"]),
            "best_epoch": int(result["best_epoch"]),
        }
        torch.save(checkpoint, checkpoint_path)
        paths[algorithm] = str(checkpoint_path)
        manifest_models[algorithm] = {
            "algorithm": DEEP_MODEL_LABELS[algorithm],
            "checkpoint": checkpoint_path.name,
            "threshold": float(result["threshold"]),
            "best_epoch": int(result["best_epoch"]),
            "parameter_count": int(result["parameter_count"]),
            "parameters": result["config"],
            "validation_metrics": result["validation_metrics"],
            "test_metrics": result.get("test_metrics"),
        }

    manifest = {
        "preprocessor": preprocessor_path.name,
        "selection_metric": "validation_pr_auc",
        "categorical_features": DEEP_CATEGORICAL_COLS,
        "numerical_features": DEEP_NUMERICAL_FEATURES,
        "models": manifest_models,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    paths["manifest"] = str(manifest_path)
    return paths


def load_deep_model_artifact(
    checkpoint_path: str | Path,
    preprocessor_path: str | Path,
    device: str | torch.device = "cpu",
) -> Tuple[nn.Module, DeepTabularPreprocessor, Dict[str, Any]]:
    """Load one saved ResNet or TabM inference contract."""
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    preprocessor = joblib.load(preprocessor_path)
    model = build_deep_model(
        checkpoint["algorithm"],
        checkpoint["cardinalities"],
        checkpoint["n_numerical"],
        checkpoint["config"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, preprocessor, checkpoint


def predict_deep_from_raw_dataframe(
    model: nn.Module,
    preprocessor: DeepTabularPreprocessor,
    checkpoint: Mapping[str, Any],
    raw_features: pd.DataFrame,
    device: str | torch.device = "cpu",
) -> pd.DataFrame:
    """Predict probabilities/classes directly from raw HR rows."""
    X = raw_features.drop(columns=["enrollee_id", "target"], errors="ignore")
    categorical, numerical = preprocessor.transform(X)
    probabilities = predict_deep_model(
        model, categorical, numerical, device=device
    )
    predictions = (probabilities >= float(checkpoint["threshold"])).astype(int)
    return pd.DataFrame(
        {"probability_target_1": probabilities, "prediction": predictions},
        index=raw_features.index,
    )


def plot_deep_model_results(
    final_models: Mapping[str, Mapping[str, Any]],
    y_test: Sequence[int],
    output_dir: str | Path,
) -> None:
    """Save learning curves and locked-test evaluations for ResNet and TabM."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    algorithms = [key for key in SUPPORTED_DEEP_MODELS if key in final_models]

    fig, axes = plt.subplots(len(algorithms), 2, figsize=(13, 4.2 * len(algorithms)))
    if len(algorithms) == 1:
        axes = np.asarray([axes])
    for row_index, algorithm in enumerate(algorithms):
        result = final_models[algorithm]
        history = result["history"]
        label = DEEP_MODEL_LABELS[algorithm]
        axes[row_index, 0].plot(history["epoch"], history["train_loss"])
        axes[row_index, 0].axvline(
            result["best_epoch"], linestyle="--", color="tab:red", label="Best epoch"
        )
        axes[row_index, 0].set(
            xlabel="Epoch", ylabel="BCE loss", title=f"{label}: training loss"
        )
        axes[row_index, 0].legend()
        axes[row_index, 1].plot(
            history["epoch"], history["validation_pr_auc"], label="PR-AUC"
        )
        axes[row_index, 1].plot(
            history["epoch"], history["validation_roc_auc"], label="ROC-AUC"
        )
        axes[row_index, 1].axvline(
            result["best_epoch"], linestyle="--", color="tab:red"
        )
        axes[row_index, 1].set(
            xlabel="Epoch", ylabel="Score", title=f"{label}: validation curves"
        )
        axes[row_index, 1].legend()
        for axis in axes[row_index]:
            axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        output_dir / "deep_models_learning_curves.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    fig, axes = plt.subplots(len(algorithms), 3, figsize=(16, 4.4 * len(algorithms)))
    if len(algorithms) == 1:
        axes = np.asarray([axes])
    for row_index, algorithm in enumerate(algorithms):
        result = final_models[algorithm]
        label = DEEP_MODEL_LABELS[algorithm]
        probabilities = result["test_probabilities"]
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        precision, recall, _ = precision_recall_curve(y_test, probabilities)
        axes[row_index, 0].plot(
            fpr, tpr, label=f"AUC={roc_auc_score(y_test, probabilities):.3f}"
        )
        axes[row_index, 0].plot([0, 1], [0, 1], "--", color="grey")
        axes[row_index, 0].set(
            xlabel="False Positive Rate", ylabel="True Positive Rate", title=f"{label}: test ROC"
        )
        axes[row_index, 0].legend()
        axes[row_index, 1].plot(
            recall,
            precision,
            label=f"AP={average_precision_score(y_test, probabilities):.3f}",
        )
        axes[row_index, 1].axhline(np.mean(y_test), linestyle="--", color="grey")
        axes[row_index, 1].set(
            xlabel="Recall", ylabel="Precision", title=f"{label}: test Precision–Recall"
        )
        axes[row_index, 1].legend()
        predictions = (probabilities >= float(result["threshold"])).astype(int)
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        axes[row_index, 2].imshow(matrix, cmap="Purples")
        for (matrix_row, column), value in np.ndenumerate(matrix):
            axes[row_index, 2].text(column, matrix_row, str(value), ha="center", va="center")
        axes[row_index, 2].set(
            xticks=[0, 1],
            yticks=[0, 1],
            xlabel="Predicted",
            ylabel="Actual",
            title=f"{label}: confusion (t={result['threshold']:.3f})",
        )
        for axis in axes[row_index]:
            axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        output_dir / "deep_models_test_evaluation.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)
