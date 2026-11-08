"""Post-selection audits for the locked tabular classifiers.

The functions in this module never choose a model from test performance.  They
support uncertainty, calibration, subgroup, and reproducibility analyses after
the model configurations and decision thresholds have already been locked.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for a file."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_calibration_error(
    y_true: Sequence[int], probabilities: Sequence[float], n_bins: int = 10
) -> float:
    """Compute equal-width expected calibration error (ECE)."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if labels.shape != scores.shape:
        raise ValueError("y_true and probabilities must have matching shapes.")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("probabilities must lie in [0, 1].")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(scores, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        ece += mask.mean() * abs(labels[mask].mean() - scores[mask].mean())
    return float(ece)


def calibration_table(
    y_true: Sequence[int],
    probabilities_by_model: Mapping[str, Sequence[float]],
    n_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return model-level calibration metrics and reliability-curve bins."""
    labels = np.asarray(y_true, dtype=int)
    metric_rows: list[dict] = []
    bin_rows: list[dict] = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    for model, probabilities in probabilities_by_model.items():
        scores = np.asarray(probabilities, dtype=float)
        if scores.shape != labels.shape:
            raise ValueError(f"{model}: prediction length does not match y_true.")
        metric_rows.append(
            {
                "model": model,
                "brier_score": float(brier_score_loss(labels, scores)),
                "ece_10_bin": expected_calibration_error(labels, scores, n_bins),
            }
        )
        bin_ids = np.clip(
            np.digitize(scores, edges[1:-1], right=True), 0, n_bins - 1
        )
        for bin_id in range(n_bins):
            mask = bin_ids == bin_id
            if not mask.any():
                continue
            bin_rows.append(
                {
                    "model": model,
                    "bin": bin_id + 1,
                    "lower": edges[bin_id],
                    "upper": edges[bin_id + 1],
                    "count": int(mask.sum()),
                    "mean_probability": float(scores[mask].mean()),
                    "observed_rate": float(labels[mask].mean()),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(bin_rows)


def _safe_ranking_metric(metric, labels: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    return float(metric(labels, scores))


def bootstrap_metric_intervals(
    y_true: Sequence[int],
    probabilities_by_model: Mapping[str, Sequence[float]],
    thresholds: Mapping[str, float],
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap model metrics and paired differences against Random Forest.

    All models are resampled with identical row indices, making the difference
    intervals paired.  This analysis quantifies uncertainty; it must not be used
    to re-select a model after the test set has been opened.
    """
    if "random_forest" not in probabilities_by_model:
        raise KeyError("Paired comparison requires a 'random_forest' baseline.")
    labels = np.asarray(y_true, dtype=int)
    predictions = {
        model: np.asarray(values, dtype=float)
        for model, values in probabilities_by_model.items()
    }
    if any(values.shape != labels.shape for values in predictions.values()):
        raise ValueError("All prediction arrays must match y_true.")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100 for stable intervals.")

    rng = np.random.default_rng(random_state)
    metric_names = ("f1", "pr_auc", "roc_auc")
    samples = {
        model: {metric: [] for metric in metric_names} for model in predictions
    }
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        for model, scores in predictions.items():
            sampled_scores = scores[indices]
            sampled_predictions = (
                sampled_scores >= float(thresholds[model])
            ).astype(int)
            samples[model]["f1"].append(
                float(f1_score(sampled_labels, sampled_predictions, zero_division=0))
            )
            samples[model]["pr_auc"].append(
                _safe_ranking_metric(
                    average_precision_score, sampled_labels, sampled_scores
                )
            )
            samples[model]["roc_auc"].append(
                _safe_ranking_metric(roc_auc_score, sampled_labels, sampled_scores)
            )

    alpha = (1.0 - confidence) / 2.0
    interval_rows: list[dict] = []
    difference_rows: list[dict] = []
    for model, model_samples in samples.items():
        for metric, values in model_samples.items():
            array = np.asarray(values, dtype=float)
            interval_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "bootstrap_mean": float(np.nanmean(array)),
                    "ci_lower": float(np.nanquantile(array, alpha)),
                    "ci_upper": float(np.nanquantile(array, 1.0 - alpha)),
                    "confidence": confidence,
                    "n_bootstrap": n_bootstrap,
                }
            )
            if model == "random_forest":
                continue
            baseline = np.asarray(samples["random_forest"][metric], dtype=float)
            difference = array - baseline
            lower = float(np.nanquantile(difference, alpha))
            upper = float(np.nanquantile(difference, 1.0 - alpha))
            difference_rows.append(
                {
                    "model": model,
                    "baseline": "random_forest",
                    "metric": metric,
                    "mean_difference": float(np.nanmean(difference)),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "excludes_zero": bool(lower > 0 or upper < 0),
                    "confidence": confidence,
                    "n_bootstrap": n_bootstrap,
                }
            )
    return pd.DataFrame(interval_rows), pd.DataFrame(difference_rows)


def subgroup_classification_audit(
    y_true: Sequence[int],
    probabilities_by_model: Mapping[str, Sequence[float]],
    thresholds: Mapping[str, float],
    groups: Sequence[object],
    minimum_group_size: int = 30,
) -> pd.DataFrame:
    """Compute descriptive error metrics for each sufficiently large subgroup."""
    labels = np.asarray(y_true, dtype=int)
    group_values = pd.Series(groups).fillna("Missing").astype(str).to_numpy()
    rows: list[dict] = []
    for model, probabilities in probabilities_by_model.items():
        scores = np.asarray(probabilities, dtype=float)
        predictions = (scores >= float(thresholds[model])).astype(int)
        for group in sorted(np.unique(group_values)):
            mask = group_values == group
            if int(mask.sum()) < minimum_group_size:
                continue
            group_labels = labels[mask]
            group_predictions = predictions[mask]
            tn, fp, fn, tp = confusion_matrix(
                group_labels, group_predictions, labels=[0, 1]
            ).ravel()
            rows.append(
                {
                    "model": model,
                    "group": group,
                    "n": int(mask.sum()),
                    "positive_rate": float(group_labels.mean()),
                    "predicted_positive_rate": float(group_predictions.mean()),
                    "precision": float(
                        precision_score(
                            group_labels, group_predictions, zero_division=0
                        )
                    ),
                    "recall_tpr": float(
                        recall_score(group_labels, group_predictions, zero_division=0)
                    ),
                    "false_positive_rate": float(fp / (fp + tn))
                    if fp + tn
                    else float("nan"),
                    "false_negative_rate": float(fn / (fn + tp))
                    if fn + tp
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)

