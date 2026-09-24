"""Small deterministic tests for leakage-safe modeling utilities."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.modeling.common import (
    FeatureSet,
    assert_features_safe,
    assert_split_isolation,
    make_linear_pipeline,
    regression_metrics,
)
from src.modeling.hdb import BASE_FEATURES, MRT_FEATURES


class ModelingSafetyTests(unittest.TestCase):
    def test_split_isolation(self) -> None:
        frame = pd.DataFrame({
            "transaction_date": ["2022-12-01", "2023-01-01", "2025-01-01"],
            "data_split": ["train", "validation", "test"],
        })
        assert_split_isolation(frame, {"train": 1, "validation": 1, "test": 1})
        with self.assertRaises(ValueError):
            assert_split_isolation(frame.iloc[[1, 0, 2]].assign(
                data_split=["train", "validation", "test"]
            ), {"train": 1, "validation": 1, "test": 1})

    def test_feature_leakage_prevention(self) -> None:
        with self.assertRaises(ValueError):
            assert_features_safe(
                FeatureSet(numeric=("resale_price",), categorical=()),
                "resale_price",
                {"latitude"},
            )

    def test_preprocessing_is_fitted_on_train_only(self) -> None:
        features = FeatureSet(numeric=("value",), categorical=("kind",))
        train = pd.DataFrame({"value": [1.0, 3.0, np.nan], "kind": ["a", "a", "b"]})
        target = pd.Series([10.0, 20.0, 15.0])
        pipeline = make_linear_pipeline(features).fit(train, target)
        statistic = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps[
            "imputer"
        ].statistics_[0]
        self.assertEqual(statistic, 2.0)
        prediction = pipeline.predict(pd.DataFrame({"value": [np.nan], "kind": ["unseen"]}))
        self.assertEqual(prediction.shape, (1,))

    def test_metric_calculation(self) -> None:
        metrics = regression_metrics(pd.Series([100.0, 200.0]), np.array([110.0, 180.0]))
        self.assertAlmostEqual(metrics["mae"], 15.0)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(250.0))
        self.assertEqual(metrics["rows"], 2)

    def test_mrt_ablation_feature_sets(self) -> None:
        self.assertNotIn("nearest_mrt_distance_m", BASE_FEATURES.columns)
        self.assertNotIn("mrt_distance_missing", BASE_FEATURES.columns)
        self.assertIn("nearest_mrt_distance_m", MRT_FEATURES.columns)
        self.assertIn("mrt_distance_missing", MRT_FEATURES.columns)


if __name__ == "__main__":
    unittest.main()
