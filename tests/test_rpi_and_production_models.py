"""RPI inference-safety and frozen production artifact tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.modeling.rpi import (
    OUTPUT_PATH,
    add_hdb_rpi_features,
    load_rpi,
    parse_rpi_text,
    rpi_for_valuation_month,
)


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"


class RPIBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rpi = load_rpi(OUTPUT_PATH)

    def test_parser_reconstructs_complete_validated_sequence(self) -> None:
        lines = []
        for row in self.rpi.sort_values(["year", "quarter"], ascending=False).itertuples():
            change = "" if pd.isna(row.qoq_percent) else f" {row.qoq_percent:.1f}%"
            lines.append(f"{row.quarter}Q {row.rpi:.1f}{change}")
        parsed = parse_rpi_text("\n".join(lines))
        self.assertEqual(len(parsed), 146)
        self.assertEqual(parsed.iloc[0]["rpi"], 24.3)
        self.assertEqual(parsed.iloc[-1]["rpi"], 202.8)

    def test_lagged_rpi_never_uses_same_or_future_quarter(self) -> None:
        frame = pd.DataFrame({
            "transaction_year": [2025], "transaction_quarter": [1], "transaction_month": [2],
        })
        enriched = add_hdb_rpi_features(frame, self.rpi)
        self.assertEqual(enriched.loc[0, "contemporaneous_rpi"], 201.0)
        self.assertEqual(enriched.loc[0, "lagged_rpi"], 197.9)  # 2024-Q4

    def test_future_valuation_uses_latest_official_older_quarter(self) -> None:
        result = rpi_for_valuation_month("2027-01", self.rpi)
        self.assertEqual(result["rpi_source_quarter"], "2026-Q2")
        self.assertEqual(result["lagged_rpi"], 202.8)
        self.assertTrue(result["is_latest_available_fallback"])


class ProductionArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads((MODELS / "inference_contract.json").read_text())
        cls.rpi = load_rpi()

    def test_contract_feature_order_matches_model_metadata(self) -> None:
        for category in ("hdb", "ec", "landed"):
            metadata = json.loads((MODELS / f"{category}_model_metadata.json").read_text())
            contract_features = self.contract["categories"][category]["model_features_in_order"]
            self.assertEqual(contract_features, metadata["features_in_order"])

    def test_landed_contract_rejects_multi_unit_input(self) -> None:
        fields = self.contract["categories"]["landed"]["inputs"]
        units = next(item for item in fields if item["name"] == "number_of_units")
        self.assertEqual(units["maximum"], 1)

    def test_serialized_models_reload_handle_unseen_categories_and_predict_finite_sgd(self) -> None:
        for category in ("hdb", "ec", "landed"):
            metadata = json.loads((MODELS / f"{category}_model_metadata.json").read_text())
            features = metadata["features_in_order"]
            path = PROCESSED / f"{category}_enriched.csv"
            if category == "hdb":
                row = pd.read_csv(path, skiprows=range(1, 900_000), nrows=1, low_memory=False)
                row = add_hdb_rpi_features(row, self.rpi)
            else:
                row = pd.read_csv(path, nrows=1, low_memory=False)
                row["months_since_1990"] = (
                    (row["transaction_year"].astype(int) - 1990) * 12
                    + row["transaction_month"].astype(int) - 1
                )
            categorical = metadata["categorical_features"][0]
            row[categorical] = "__UNSEEN_AT_INFERENCE__"
            model = joblib.load(MODELS / f"{category}_model.joblib")
            prediction = np.asarray(model.predict(row[features]), dtype=float)
            self.assertEqual(prediction.shape, (1,))
            self.assertTrue(np.isfinite(prediction).all())
            self.assertGreater(prediction[0], 0)


if __name__ == "__main__":
    unittest.main()
