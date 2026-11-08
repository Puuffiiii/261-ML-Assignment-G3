import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import make_classification

from modules.modeling import (
    binary_classification_metrics,
    refit_and_evaluate_on_test,
    run_validation_experiments,
    save_model_artifacts,
    select_best_candidates,
    select_f1_threshold,
)


class IdentityPreprocessor:
    def transform(self, X):
        return np.asarray(X)


class ModelingPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = make_classification(
            n_samples=240,
            n_features=8,
            n_informative=5,
            weights=[0.7, 0.3],
            random_state=7,
        )
        cls.features = {
            "X_train_std": X[:140],
            "X_validation_std": X[140:190],
            "X_test_std": X[190:],
        }
        cls.y_train = y[:140]
        cls.y_validation = y[140:190]
        cls.y_test = y[190:]
        cls.config = {
            "random_state": 7,
            "n_jobs": 1,
            "selection_metric": "validation_pr_auc",
            "models": {
                "logistic_regression": {
                    "representations": ["std"],
                    "parameter_grid": [
                        {"C": 0.1, "class_weight": "balanced"},
                        {"C": 1.0, "class_weight": "balanced"},
                    ],
                },
                "random_forest": {
                    "representations": ["std"],
                    "parameter_grid": [
                        {
                            "n_estimators": 20,
                            "max_depth": 4,
                            "class_weight": "balanced",
                        }
                    ],
                },
            },
        }

    def test_threshold_and_metrics_contract(self):
        probabilities = np.array([0.05, 0.2, 0.45, 0.8, 0.95])
        labels = np.array([0, 0, 1, 1, 1])
        threshold, best_f1 = select_f1_threshold(labels, probabilities)
        metrics = binary_classification_metrics(labels, probabilities, threshold)

        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)
        self.assertAlmostEqual(best_f1, metrics["f1"])
        self.assertEqual(metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"], 5)

    def test_train_select_refit_and_persist(self):
        validation, candidates = run_validation_experiments(
            self.features, self.y_train, self.y_validation, self.config
        )
        self.assertEqual(len(validation), 3)
        self.assertTrue(validation["validation_pr_auc"].between(0, 1).all())

        best = select_best_candidates(validation, "validation_pr_auc")
        self.assertEqual(set(best["model"]), {"logistic_regression", "random_forest"})
        test_results, final_models = refit_and_evaluate_on_test(
            best,
            candidates,
            self.features,
            self.y_train,
            self.y_validation,
            self.y_test,
        )
        self.assertEqual(len(test_results), 2)
        self.assertTrue(test_results["test_roc_auc"].between(0, 1).all())

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = save_model_artifacts(
                temp_dir,
                final_models,
                {"std": IdentityPreprocessor()},
                test_results,
            )
            self.assertEqual(set(paths), {"logistic_regression", "random_forest", "manifest"})
            for path in paths.values():
                self.assertTrue(Path(path).exists())
            artifact = joblib.load(paths["logistic_regression"])
            self.assertIn("model", artifact)
            self.assertIn("preprocessor", artifact)
            self.assertIn("threshold", artifact)


if __name__ == "__main__":
    unittest.main()
