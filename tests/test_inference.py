"""Deterministic, network-free tests for the production inference boundary."""

from __future__ import annotations

import math
import unittest

from src.inference.predictor import get_predictor
from src.inference.reference_data import load_hdb_locations, load_ui_reference, lookup_lagged_rpi
from src.inference.validation import InputValidationError


class ProductionInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = get_predictor()
        cls.refs = load_ui_reference()

    def assert_valid_result(self, result, expected_category: str) -> None:
        self.assertEqual(result.category, expected_category)
        self.assertTrue(math.isfinite(result.predicted_price))
        self.assertGreater(result.predicted_price, 0)
        self.assertLessEqual(result.range_lower, result.predicted_price)
        self.assertGreaterEqual(result.range_upper, result.predicted_price)
        self.assertEqual(result.range_percentile, "P80")
        self.assertGreater(result.model_test_mae, 0)

    def hdb_request(self, block: str, street: str) -> dict:
        return {
            "valuation_month": "2026-09",
            "town": "ANG MO KIO",
            "flat_type": "4 ROOM",
            "floor_area_sqm": 90,
            "storey_range": "10 TO 12",
            "flat_model": "MODEL A",
            "lease_commence_year": 1985,
            "block": block,
            "street_name": street,
        }

    def test_hdb_eligible_local_mrt_and_lagged_rpi(self) -> None:
        locations = load_hdb_locations()
        eligible = locations.loc[locations["mrt_distance_model_eligible"]].iloc[0]
        result = self.predictor.predict_hdb(self.hdb_request(str(eligible["block"]), str(eligible["street_name"])))
        self.assert_valid_result(result, "hdb")
        self.assertTrue(result.references["mrt_used_by_model"])
        self.assertFalse(result.derived["mrt_distance_missing"])
        self.assertEqual(result.references["rpi_source_quarter"], "2026-Q2")
        self.assertEqual(result.model_name, "IndexedTargetRegressor")

    def test_hdb_unknown_address_uses_missing_mrt_path(self) -> None:
        result = self.predictor.predict_hdb(self.hdb_request("NOT-A-BLOCK", "NOT-A-STREET"))
        self.assert_valid_result(result, "hdb")
        self.assertFalse(result.references["mrt_used_by_model"])
        self.assertTrue(result.derived["mrt_distance_missing"])

    def test_future_hdb_date_uses_latest_older_rpi_without_fabrication(self) -> None:
        value = lookup_lagged_rpi("2030-01")
        self.assertEqual(value["rpi_source_quarter"], "2026-Q2")
        self.assertTrue(value["is_latest_available_fallback"])

    def test_ec_known_and_unseen_project_are_prediction_safe(self) -> None:
        options = self.refs["ec"]
        base = {
            "valuation_month": "2026-09",
            "project_name": options["projects"][0],
            "area_sqm": 100,
            "floor_level": options["floor_levels"][0],
            "sale_type": options["sale_types"][0],
            "area_type": options["area_types"][0],
            "property_type": options["property_types"][0],
            "tenure_category": "LEASEHOLD",
            "lease_duration_years": 99,
            "lease_commence_year": 2020,
            "postal_district": options["postal_districts"][0],
            "market_segment": options["market_segments"][0],
        }
        known = self.predictor.predict_ec(base)
        unseen = self.predictor.predict_ec({**base, "project_name": "UNSEEN TEST PROJECT"})
        self.assert_valid_result(known, "ec")
        self.assert_valid_result(unseen, "ec")
        self.assertFalse(known.references["mrt_used_by_model"])
        self.assertTrue(any("not present" in item for item in unseen.warnings))

    def test_landed_forces_one_unit(self) -> None:
        options = self.refs["landed"]
        base = {
            "valuation_month": "2026-09",
            "area_sqm": 250,
            "sale_type": options["sale_types"][0],
            "area_type": options["area_types"][0],
            "property_type": options["property_types"][0],
            "tenure_category": "FREEHOLD",
            "postal_district": options["postal_districts"][0],
            "market_segment": options["market_segments"][0],
        }
        result = self.predictor.predict_landed(base)
        self.assert_valid_result(result, "landed")
        self.assertEqual(result.derived["number_of_units"], 1)
        with self.assertRaises(InputValidationError):
            self.predictor.predict_landed({**base, "number_of_units": 2})


if __name__ == "__main__":
    unittest.main()
