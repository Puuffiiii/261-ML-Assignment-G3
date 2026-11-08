"""Training, selection, evaluation, and persistence for the tabular ML models.

The module deliberately keeps model selection on the validation split and exposes
the test split only after both hyperparameters and decision thresholds are locked.
It supports the two traditional algorithms used by the assignment notebook:
regularized Logistic Regression and Random Forest.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


SUPPORTED_MODELS = ("logistic_regression", "random_forest")


def load_experiment_config(path: str | Path) -> Dict[str, Any]:
    """Load and minimally validate the JSON experiment configuration."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    if "models" not in config or not isinstance(config["models"], dict):
        raise ValueError("Configuration must contain a 'models' object.")
    unknown = sorted(set(config["models"]) - set(SUPPORTED_MODELS))
    if unknown:
        raise ValueError(f"Unsupported model(s): {unknown}")

    for model_name, model_config in config["models"].items():
        if not model_config.get("representations"):
            raise ValueError(f"{model_name} must define at least one representation.")
        if not model_config.get("parameter_grid"):
            raise ValueError(f"{model_name} must define a non-empty parameter_grid.")
    return config


def build_model(
    model_name: str,
    parameters: Mapping[str, Any],
    random_state: int = 42,
    n_jobs: int = -1,
) -> BaseEstimator:
    """Construct one supported estimator from a serializable parameter mapping."""
    params = dict(parameters)
    if model_name == "logistic_regression":
        defaults = {
            "solver": "liblinear",
            "max_iter": 2000,
            "random_state": random_state,
        }
        defaults.update(params)
        return LogisticRegression(**defaults)
    if model_name == "random_forest":
        defaults = {
            "n_estimators": 300,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
        defaults.update(params)
        return RandomForestClassifier(**defaults)
    raise ValueError(f"Unsupported model '{model_name}'. Choose from {SUPPORTED_MODELS}.")


def predict_positive_probability(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """Return P(y=1); fail clearly if an estimator cannot provide probabilities."""
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{type(model).__name__} does not implement predict_proba().")
    probabilities = np.asarray(model.predict_proba(X))
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Expected binary predict_proba output with two columns.")
    return probabilities[:, 1]


def binary_classification_metrics(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, float | int]:
    """Compute threshold-free and threshold-dependent binary metrics."""
    y_true_array = np.asarray(y_true, dtype=int)
    probability_array = np.asarray(probabilities, dtype=float)
    predictions = (probability_array >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_array, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_array, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_array, predictions)),
        "precision": float(precision_score(y_true_array, predictions, zero_division=0)),
        "recall": float(recall_score(y_true_array, predictions, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true_array, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_array, probability_array)),
        "pr_auc": float(average_precision_score(y_true_array, probability_array)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_f1_threshold(
    y_true: Sequence[int], probabilities: Sequence[float]
) -> Tuple[float, float]:
    """Choose the probability threshold that maximizes validation F1.

    ``precision_recall_curve`` returns one more precision/recall value than
    threshold value, so the last metric point is intentionally excluded.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if thresholds.size == 0:
        return 0.5, 0.0
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


def run_validation_experiments(
    feature_sets: Mapping[str, np.ndarray],
    y_train: Sequence[int],
    y_validation: Sequence[int],
    config: Mapping[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Fit every configured candidate and score it on validation only.

    Parameters
    ----------
    feature_sets:
        Mapping containing ``X_train_<representation>`` and
        ``X_validation_<representation>`` arrays.

    Returns
    -------
    results:
        One row per model/representation/hyperparameter candidate.
    fitted_candidates:
        Candidate metadata and fitted estimator, keyed by candidate id.
    """
    random_state = int(config.get("random_state", 42))
    n_jobs = int(config.get("n_jobs", -1))
    rows = []
    fitted_candidates: Dict[str, Dict[str, Any]] = {}

    for model_name, model_config in config["models"].items():
        candidate_number = 0
        for representation in model_config["representations"]:
            train_key = f"X_train_{representation}"
            validation_key = f"X_validation_{representation}"
            if train_key not in feature_sets or validation_key not in feature_sets:
                raise KeyError(
                    f"Missing feature arrays '{train_key}'/'{validation_key}'."
                )
            X_train = feature_sets[train_key]
            X_validation = feature_sets[validation_key]

            for parameters in model_config["parameter_grid"]:
                candidate_number += 1
                candidate_id = f"{model_name}__{candidate_number:02d}"
                estimator = build_model(
                    model_name,
                    parameters,
                    random_state=random_state,
                    n_jobs=n_jobs,
                )
                started = time.perf_counter()
                estimator.fit(X_train, y_train)
                elapsed = time.perf_counter() - started
                validation_probability = predict_positive_probability(
                    estimator, X_validation
                )
                threshold, _ = select_f1_threshold(
                    y_validation, validation_probability
                )
                metrics_default = binary_classification_metrics(
                    y_validation, validation_probability, threshold=0.5
                )
                metrics_tuned = binary_classification_metrics(
                    y_validation, validation_probability, threshold=threshold
                )

                row = {
                    "candidate_id": candidate_id,
                    "model": model_name,
                    "representation": representation,
                    "parameters": json.dumps(
                        parameters, sort_keys=True, ensure_ascii=False
                    ),
                    "fit_seconds": float(elapsed),
                    "validation_threshold": threshold,
                    "validation_roc_auc": metrics_tuned["roc_auc"],
                    "validation_pr_auc": metrics_tuned["pr_auc"],
                    "validation_accuracy_at_0_5": metrics_default["accuracy"],
                    "validation_f1_at_0_5": metrics_default["f1"],
                    "validation_precision": metrics_tuned["precision"],
                    "validation_recall": metrics_tuned["recall"],
                    "validation_f1": metrics_tuned["f1"],
                    "validation_balanced_accuracy": metrics_tuned[
                        "balanced_accuracy"
                    ],
                }
                rows.append(row)
                fitted_candidates[candidate_id] = {
                    "model": estimator,
                    "model_name": model_name,
                    "representation": representation,
                    "parameters": dict(parameters),
                    "threshold": threshold,
                    "validation_probabilities": validation_probability,
                }

    results = pd.DataFrame(rows).sort_values(
        ["model", "validation_pr_auc", "validation_roc_auc"],
        ascending=[True, False, False],
    )
    return results.reset_index(drop=True), fitted_candidates


def select_best_candidates(
    validation_results: pd.DataFrame,
    primary_metric: str = "validation_pr_auc",
) -> pd.DataFrame:
    """Select one locked configuration per algorithm from validation results."""
    if primary_metric not in validation_results.columns:
        raise KeyError(f"Unknown selection metric '{primary_metric}'.")
    ordered = validation_results.sort_values(
        [primary_metric, "validation_roc_auc", "validation_f1"],
        ascending=False,
    )
    return ordered.groupby("model", as_index=False, sort=True).head(1).reset_index(
        drop=True
    )


def refit_and_evaluate_on_test(
    best_candidates: pd.DataFrame,
    fitted_candidates: Mapping[str, Mapping[str, Any]],
    feature_sets: Mapping[str, np.ndarray],
    y_train: Sequence[int],
    y_validation: Sequence[int],
    y_test: Sequence[int],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Refit each locked model on train+validation, then evaluate test once."""
    final_models: Dict[str, Dict[str, Any]] = {}
    test_rows = []

    for row in best_candidates.to_dict(orient="records"):
        candidate_id = row["candidate_id"]
        candidate = fitted_candidates[candidate_id]
        model_name = candidate["model_name"]
        representation = candidate["representation"]
        threshold = float(candidate["threshold"])

        X_train_validation = np.concatenate(
            [
                feature_sets[f"X_train_{representation}"],
                feature_sets[f"X_validation_{representation}"],
            ],
            axis=0,
        )
        y_train_validation = np.concatenate(
            [np.asarray(y_train), np.asarray(y_validation)], axis=0
        )
        final_model = clone(candidate["model"])
        final_model.fit(X_train_validation, y_train_validation)
        test_probability = predict_positive_probability(
            final_model, feature_sets[f"X_test_{representation}"]
        )
        metrics = binary_classification_metrics(
            y_test, test_probability, threshold=threshold
        )
        test_rows.append(
            {
                "model": model_name,
                "candidate_id": candidate_id,
                "representation": representation,
                "parameters": json.dumps(
                    candidate["parameters"], sort_keys=True, ensure_ascii=False
                ),
                **{f"test_{key}": value for key, value in metrics.items()},
            }
        )
        final_models[model_name] = {
            "model": final_model,
            "representation": representation,
            "parameters": candidate["parameters"],
            "threshold": threshold,
            "validation_probabilities": candidate["validation_probabilities"],
            "test_probabilities": test_probability,
        }

    # Keep the report in a stable algorithm order. Sorting or selecting by a test
    # metric here would turn the held-out test set into another selection set.
    results = pd.DataFrame(test_rows).sort_values("model")
    return results.reset_index(drop=True), final_models


def feature_importance_table(
    final_models: Mapping[str, Mapping[str, Any]],
    feature_names_by_representation: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Create coefficient/impurity importance tables for both final models."""
    frames = []
    for model_name, bundle in final_models.items():
        model = bundle["model"]
        representation = bundle["representation"]
        names = list(feature_names_by_representation[representation])
        if hasattr(model, "coef_"):
            signed = np.asarray(model.coef_).ravel()
            importance = np.abs(signed)
        elif hasattr(model, "feature_importances_"):
            signed = np.asarray(model.feature_importances_)
            importance = signed.copy()
        else:
            continue
        if len(names) != len(importance):
            raise ValueError(
                f"{model_name}: {len(names)} names for {len(importance)} importances."
            )
        frame = pd.DataFrame(
            {
                "model": model_name,
                "representation": representation,
                "feature": names,
                "signed_effect": signed,
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)
        frame["rank"] = np.arange(1, len(frame) + 1)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_model_comparison(
    final_models: Mapping[str, Mapping[str, Any]],
    y_test: Sequence[int],
    output_dir: str | Path,
) -> None:
    """Save ROC, precision-recall, and confusion-matrix comparison figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for model_name, bundle in final_models.items():
        test_probabilities = bundle["test_probabilities"]
        fpr, tpr, _ = roc_curve(y_test, test_probabilities)
        roc_auc = roc_auc_score(y_test, test_probabilities)
        axes[0].plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")

        precision, recall, _ = precision_recall_curve(y_test, test_probabilities)
        pr_auc = average_precision_score(y_test, test_probabilities)
        axes[1].plot(recall, precision, label=f"{model_name} (AP={pr_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", label="Random")
    axes[0].set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC curves on locked test set",
    )
    axes[1].axhline(
        np.mean(y_test), linestyle="--", color="grey", label="Positive prevalence"
    )
    axes[1].set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision–Recall curves on locked test set",
    )
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "model_roc_pr_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    model_items = list(final_models.items())
    fig, axes = plt.subplots(1, len(model_items), figsize=(6 * len(model_items), 4.8))
    axes = np.atleast_1d(axes)
    for axis, (model_name, bundle) in zip(axes, model_items):
        predictions = (
            bundle["test_probabilities"] >= float(bundle["threshold"])
        ).astype(int)
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        image = axis.imshow(matrix, cmap="Blues")
        for (row_index, col_index), value in np.ndenumerate(matrix):
            axis.text(col_index, row_index, str(value), ha="center", va="center")
        axis.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["0", "1"],
            yticklabels=["0", "1"],
            xlabel="Predicted",
            ylabel="Actual",
            title=f"{model_name}\nthreshold={bundle['threshold']:.3f}",
        )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "model_confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_model_artifacts(
    output_dir: str | Path,
    final_models: Mapping[str, Mapping[str, Any]],
    preprocessors: Mapping[str, Any],
    test_results: pd.DataFrame,
) -> Dict[str, str]:
    """Persist inference-ready model bundles and a human-readable manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: Dict[str, str] = {}
    manifest: Dict[str, Any] = {"models": {}}

    result_lookup = test_results.set_index("model").to_dict(orient="index")
    for model_name, bundle in final_models.items():
        representation = bundle["representation"]
        artifact = {
            "model": bundle["model"],
            "preprocessor": preprocessors[representation],
            "representation": representation,
            "threshold": float(bundle["threshold"]),
            "positive_class": 1,
            "dropped_columns": ["enrollee_id", "city"],
        }
        path = output_dir / f"{model_name}.joblib"
        joblib.dump(artifact, path)
        saved_paths[model_name] = str(path)
        manifest["models"][model_name] = {
            "artifact": path.name,
            "representation": representation,
            "threshold": float(bundle["threshold"]),
            "parameters": bundle["parameters"],
            "test_metrics": {
                key.removeprefix("test_"): value
                for key, value in result_lookup[model_name].items()
                if key.startswith("test_") and isinstance(value, (int, float))
            },
        }

    manifest_path = output_dir / "model_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    saved_paths["manifest"] = str(manifest_path)
    return saved_paths


def predict_from_raw_dataframe(
    artifact: Mapping[str, Any], raw_features: pd.DataFrame
) -> pd.DataFrame:
    """Run inference from raw feature rows with a saved model bundle."""
    X = raw_features.drop(columns=["enrollee_id", "city", "target"], errors="ignore")
    transformed = artifact["preprocessor"].transform(X)
    probabilities = predict_positive_probability(artifact["model"], transformed)
    predictions = (probabilities >= float(artifact["threshold"])).astype(int)
    return pd.DataFrame(
        {"probability_target_1": probabilities, "prediction": predictions},
        index=raw_features.index,
    )
