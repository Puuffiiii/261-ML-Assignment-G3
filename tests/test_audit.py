import tempfile
import unittest
from pathlib import Path

import numpy as np

from modules.audit import (
    bootstrap_metric_intervals,
    calibration_table,
    expected_calibration_error,
    file_sha256,
    subgroup_classification_audit,
)


class AuditTest(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([0, 0, 0, 1, 1, 1, 0, 1] * 20)
        base = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.3, 0.7] * 20)
        self.probabilities = {
            "random_forest": base,
            "resnet": np.clip(base + 0.02, 0, 1),
        }
        self.thresholds = {"random_forest": 0.5, "resnet": 0.5}

    def test_calibration_contract(self):
        self.assertGreaterEqual(
            expected_calibration_error(self.labels, self.probabilities["random_forest"]), 0
        )
        metrics, bins = calibration_table(self.labels, self.probabilities, n_bins=5)
        self.assertEqual(set(metrics["model"]), set(self.probabilities))
        self.assertTrue(metrics["brier_score"].between(0, 1).all())
        self.assertGreater(len(bins), 0)

    def test_bootstrap_and_subgroup_contracts(self):
        intervals, differences = bootstrap_metric_intervals(
            self.labels,
            self.probabilities,
            self.thresholds,
            n_bootstrap=100,
            random_state=7,
        )
        self.assertEqual(set(intervals["metric"]), {"f1", "pr_auc", "roc_auc"})
        self.assertEqual(set(differences["model"]), {"resnet"})
        groups = np.array(["A", "B"] * (len(self.labels) // 2))
        audit = subgroup_classification_audit(
            self.labels, self.probabilities, self.thresholds, groups, minimum_group_size=10
        )
        self.assertEqual(set(audit["group"]), {"A", "B"})

    def test_file_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.txt"
            path.write_text("abc", encoding="utf-8")
            self.assertEqual(
                file_sha256(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )


if __name__ == "__main__":
    unittest.main()
