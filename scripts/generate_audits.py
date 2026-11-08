"""Generate reproducible post-selection audit tables and figures."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.audit import (  # noqa: E402
    bootstrap_metric_intervals,
    calibration_table,
    file_sha256,
    subgroup_classification_audit,
)
from modules.deep_learning import (  # noqa: E402
    load_deep_model_artifact,
    predict_deep_from_raw_dataframe,
)
from modules.modeling import (  # noqa: E402
    binary_classification_metrics,
    build_model,
    predict_positive_probability,
    select_f1_threshold,
)
from modules.preprocessing import (  # noqa: E402
    TabularPreprocessor,
    split_train_validation_test,
)


REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
DATA_PATH = ROOT / "data" / "aug_train.csv"
MODELS = ROOT / "models"
RANDOM_STATE = 42


def locked_predictions(df: pd.DataFrame):
    train, validation, test, y_train, y_validation, y_test = (
        split_train_validation_test(df, random_state=RANDOM_STATE, drop_unnecessary=False)
    )
    probabilities: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    for model_name in ("logistic_regression", "random_forest"):
        artifact = joblib.load(MODELS / f"{model_name}.joblib")
        raw = test.drop(columns=["enrollee_id", "city"], errors="ignore")
        transformed = artifact["preprocessor"].transform(raw)
        probabilities[model_name] = predict_positive_probability(
            artifact["model"], transformed
        )
        thresholds[model_name] = float(artifact["threshold"])
    for model_name in ("resnet", "tabm"):
        model, preprocessor, checkpoint = load_deep_model_artifact(
            MODELS / f"{model_name}.pt",
            MODELS / "deep_tabular_preprocessor.joblib",
            device="cpu",
        )
        output = predict_deep_from_raw_dataframe(
            model, preprocessor, checkpoint, test, device="cpu"
        )
        probabilities[model_name] = output["probability_target_1"].to_numpy()
        thresholds[model_name] = float(checkpoint["threshold"])
    return train, validation, test, y_train, y_validation, y_test, probabilities, thresholds


def save_calibration(y_test, probabilities):
    metrics, bins = calibration_table(y_test, probabilities)
    metrics.to_csv(REPORTS / "model_calibration_metrics.csv", index=False)
    bins.to_csv(REPORTS / "model_calibration_bins.csv", index=False)
    fig, axis = plt.subplots(figsize=(7.2, 6.2))
    axis.plot([0, 1], [0, 1], "--", color="black", label="Perfect calibration")
    for model, rows in bins.groupby("model"):
        axis.plot(
            rows["mean_probability"], rows["observed_rate"], marker="o", label=model
        )
    axis.set(
        xlabel="Mean predicted probability",
        ylabel="Observed positive rate",
        title="Calibration on the locked test set (descriptive only)",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "model_calibration_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_bootstrap(y_test, probabilities, thresholds):
    intervals, differences = bootstrap_metric_intervals(
        y_test, probabilities, thresholds, n_bootstrap=2000, random_state=RANDOM_STATE
    )
    intervals.to_csv(REPORTS / "model_bootstrap_intervals.csv", index=False)
    differences.to_csv(REPORTS / "model_paired_differences.csv", index=False)
    f1_rows = intervals[intervals["metric"] == "f1"].copy()
    x = np.arange(len(f1_rows))
    errors = np.vstack(
        [
            f1_rows["bootstrap_mean"] - f1_rows["ci_lower"],
            f1_rows["ci_upper"] - f1_rows["bootstrap_mean"],
        ]
    )
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.errorbar(x, f1_rows["bootstrap_mean"], yerr=errors, fmt="o", capsize=5)
    axis.set_xticks(x, f1_rows["model"], rotation=15)
    axis.set(ylabel="F1 with 95% bootstrap CI", title="Uncertainty on locked test predictions")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "model_bootstrap_f1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_fairness(y_test, test, probabilities, thresholds):
    audit = subgroup_classification_audit(
        y_test,
        probabilities,
        thresholds,
        test["gender"].fillna("Missing"),
        minimum_group_size=30,
    )
    audit.to_csv(REPORTS / "gender_subgroup_audit.csv", index=False)
    plot = audit.pivot(index="group", columns="model", values="recall_tpr")
    axis = plot.plot(kind="bar", figsize=(10, 5.2), ylim=(0, 1), rot=0)
    axis.set(
        ylabel="Recall / true-positive rate",
        xlabel="Self-reported gender group",
        title="Gender subgroup audit on locked test predictions",
    )
    axis.legend(title="Model")
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(FIGURES / "gender_subgroup_recall.png", dpi=300, bbox_inches="tight")
    plt.close(axis.figure)


def save_permutation_importance(validation, y_validation):
    raw = validation.drop(columns=["enrollee_id", "city"], errors="ignore")
    frames = []
    for model_name in ("logistic_regression", "random_forest"):
        artifact = joblib.load(MODELS / f"{model_name}.joblib")
        transformed = artifact["preprocessor"].transform(raw)
        result = permutation_importance(
            artifact["model"],
            transformed,
            y_validation,
            scoring="average_precision",
            n_repeats=20,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        names = artifact["preprocessor"].get_feature_names_out()
        frame = pd.DataFrame(
            {
                "model": model_name,
                "feature": names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
        frame["rank"] = np.arange(1, len(frame) + 1)
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(
        REPORTS / "model_permutation_importance.csv", index=False
    )


def save_ablation(train, validation, y_train, y_validation):
    parameters = {
        "n_estimators": 250,
        "max_depth": 8,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
    }
    rows = []
    variants = {
        "full": (True, True),
        "without_engineered_features": (True, False),
        "without_missing_indicators": (False, True),
        "without_engineered_or_missing": (False, False),
    }
    train_raw = train.drop(columns=["enrollee_id", "city"], errors="ignore")
    validation_raw = validation.drop(columns=["enrollee_id", "city"], errors="ignore")
    for name, (missing, engineered) in variants.items():
        preprocessor = TabularPreprocessor(
            scaling="standard",
            add_missing_indicators=missing,
            add_engineered_features=engineered,
            random_state=RANDOM_STATE,
        )
        X_train = preprocessor.fit_transform(train_raw)
        X_validation = preprocessor.transform(validation_raw)
        model = build_model("random_forest", parameters, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        scores = predict_positive_probability(model, X_validation)
        threshold, _ = select_f1_threshold(y_validation, scores)
        metrics = binary_classification_metrics(y_validation, scores, threshold)
        rows.append(
            {
                "variant": name,
                "feature_count": X_train.shape[1],
                "validation_threshold": threshold,
                "validation_f1": metrics["f1"],
                "validation_pr_auc": metrics["pr_auc"],
                "validation_roc_auc": metrics["roc_auc"],
            }
        )
    pd.DataFrame(rows).to_csv(REPORTS / "feature_ablation_results.csv", index=False)


def save_repeated_cv(train, validation, y_train, y_validation):
    X = pd.concat([train, validation]).drop(
        columns=["enrollee_id", "city"], errors="ignore"
    )
    y = pd.concat([y_train, y_validation]).loc[X.index]
    specifications = {
        "logistic_regression": {"C": 10.0, "class_weight": "balanced"},
        "random_forest": {
            "n_estimators": 250,
            "max_depth": 8,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced",
        },
    }
    splitter = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=3, random_state=RANDOM_STATE
    )
    rows = []
    for split_number, (fit_indices, score_indices) in enumerate(splitter.split(X, y), 1):
        for model_name, parameters in specifications.items():
            preprocessor = TabularPreprocessor(random_state=RANDOM_STATE)
            X_fit = preprocessor.fit_transform(X.iloc[fit_indices])
            X_score = preprocessor.transform(X.iloc[score_indices])
            model = build_model(model_name, parameters, random_state=RANDOM_STATE, n_jobs=-1)
            model.fit(X_fit, y.iloc[fit_indices])
            scores = predict_positive_probability(model, X_score)
            rows.append(
                {
                    "model": model_name,
                    "repeat": (split_number - 1) // 5 + 1,
                    "fold": (split_number - 1) % 5 + 1,
                    "pr_auc": average_precision_score(y.iloc[score_indices], scores),
                    "roc_auc": roc_auc_score(y.iloc[score_indices], scores),
                    "f1_at_0_5": f1_score(
                        y.iloc[score_indices], scores >= 0.5, zero_division=0
                    ),
                }
            )
    detailed = pd.DataFrame(rows)
    detailed.to_csv(REPORTS / "repeated_cv_results.csv", index=False)
    summary = detailed.groupby("model")[["pr_auc", "roc_auc", "f1_at_0_5"]].agg(
        ["mean", "std"]
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(REPORTS / "repeated_cv_summary.csv", index=False)


def save_provenance():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unavailable"
    artifact_paths = [
        DATA_PATH,
        MODELS / "logistic_regression.joblib",
        MODELS / "random_forest.joblib",
        MODELS / "resnet.pt",
        MODELS / "tabm.pt",
    ]
    provenance = {
        "git_commit_at_generation": commit,
        "random_state": RANDOM_STATE,
        "selection_metric": "validation_pr_auc",
        "dataset_original_source": "https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in artifact_paths},
        "audit_policy": "Test audits are descriptive and were not used for model selection.",
    }
    (REPORTS / "audit_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    train, validation, test, y_train, y_validation, y_test, probabilities, thresholds = (
        locked_predictions(df)
    )
    pd.DataFrame({"row_index": test.index, "target": y_test, **probabilities}).to_csv(
        REPORTS / "locked_test_predictions.csv", index=False
    )
    save_calibration(y_test, probabilities)
    save_bootstrap(y_test, probabilities, thresholds)
    save_fairness(y_test, test, probabilities, thresholds)
    save_permutation_importance(validation, y_validation)
    save_ablation(train, validation, y_train, y_validation)
    save_repeated_cv(train, validation, y_train, y_validation)
    save_provenance()
    print("Generated calibration, uncertainty, subgroup, ablation, CV, and provenance audits.")


if __name__ == "__main__":
    main()
