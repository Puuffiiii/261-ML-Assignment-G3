import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from modules.deep_learning import (
    DEEP_NUMERICAL_FEATURES,
    DeepTabularPreprocessor,
    build_deep_model,
    load_deep_model_artifact,
    predict_deep_from_raw_dataframe,
    save_deep_model_artifacts,
)


class DeepLearningPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv("data/aug_train.csv").head(40)

    def test_preprocessor_handles_unseen_categories(self):
        preprocessor = DeepTabularPreprocessor().fit(self.raw.iloc[:30])
        unseen = self.raw.iloc[30:32].copy()
        unseen.loc[unseen.index[0], "city"] = "city_never_seen_in_train"
        categorical, numerical = preprocessor.transform(unseen)

        self.assertEqual(categorical.shape, (2, 10))
        self.assertEqual(numerical.shape, (2, len(DEEP_NUMERICAL_FEATURES)))
        self.assertEqual(categorical[0, 0], 0)
        self.assertTrue(np.isfinite(numerical).all())

    def test_resnet_and_tabm_forward_contracts(self):
        preprocessor = DeepTabularPreprocessor().fit(self.raw.iloc[:30])
        categorical, numerical = preprocessor.transform(self.raw.iloc[30:35])
        categorical_tensor = torch.from_numpy(categorical)
        numerical_tensor = torch.from_numpy(numerical)

        resnet_config = {
            "embedding_dim": 4,
            "d_block": 16,
            "n_blocks": 1,
            "d_hidden_multiplier": 1.5,
            "dropout": 0.0,
            "residual_dropout": 0.0,
        }
        tabm_config = {
            **resnet_config,
            "member_count": 4,
        }
        resnet = build_deep_model(
            "resnet", preprocessor.cardinalities_, len(DEEP_NUMERICAL_FEATURES), resnet_config
        )
        tabm = build_deep_model(
            "tabm", preprocessor.cardinalities_, len(DEEP_NUMERICAL_FEATURES), tabm_config
        )

        self.assertEqual(tuple(resnet(categorical_tensor, numerical_tensor).shape), (5,))
        self.assertEqual(tuple(tabm(categorical_tensor, numerical_tensor).shape), (5, 4))

    def test_checkpoint_round_trip_and_raw_inference(self):
        preprocessor = DeepTabularPreprocessor().fit(self.raw.iloc[:30])
        configs = {
            "resnet": {
                "embedding_dim": 4,
                "d_block": 16,
                "n_blocks": 1,
                "d_hidden_multiplier": 1.5,
                "dropout": 0.0,
                "residual_dropout": 0.0,
            },
            "tabm": {
                "embedding_dim": 4,
                "member_count": 4,
                "d_block": 16,
                "n_blocks": 1,
                "d_hidden_multiplier": 1.5,
                "dropout": 0.0,
                "residual_dropout": 0.0,
            },
        }
        final_models = {}
        for algorithm, config in configs.items():
            model = build_deep_model(
                algorithm,
                preprocessor.cardinalities_,
                len(DEEP_NUMERICAL_FEATURES),
                config,
            )
            final_models[algorithm] = {
                "state_dict": model.state_dict(),
                "config": config,
                "threshold": 0.4,
                "best_epoch": 1,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "validation_metrics": {"pr_auc": 0.5},
                "test_metrics": {"pr_auc": 0.5},
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = save_deep_model_artifacts(temp_dir, preprocessor, final_models)
            for algorithm in ("resnet", "tabm"):
                model, loaded_preprocessor, checkpoint = load_deep_model_artifact(
                    paths[algorithm], paths["preprocessor"]
                )
                predictions = predict_deep_from_raw_dataframe(
                    model,
                    loaded_preprocessor,
                    checkpoint,
                    self.raw.iloc[30:35],
                )
                self.assertEqual(len(predictions), 5)
                self.assertTrue(predictions["probability_target_1"].between(0, 1).all())
            self.assertTrue(Path(paths["manifest"]).exists())


if __name__ == "__main__":
    unittest.main()
