"""Deterministic location/MRT tests with no live API dependency."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.enrich_location_data import build_location_features, merge_location_features
from src.location.geo import haversine_m, nearest_mrt_exits
from src.location.geocode_properties import (
    GeocodeCache,
    build_query_fallbacks,
    choose_result,
    geocode_record,
    process_records,
)
from src.location.validate_matches import validate_cached_matches


class FakeClient:
    def __init__(self, payloads: list[dict]):
        self.payloads = payloads
        self.calls: list[str] = []

    def search(self, query: str) -> dict:
        self.calls.append(query)
        return self.payloads.pop(0)


class GeographicTests(unittest.TestCase):
    def test_haversine_zero_and_known_scale(self) -> None:
        self.assertAlmostEqual(float(haversine_m(1.30, 103.80, 1.30, 103.80)), 0.0, places=6)
        # 0.001 degree latitude is approximately 111.2 metres.
        self.assertAlmostEqual(float(haversine_m(1.30, 103.80, 1.301, 103.80)), 111.2, delta=0.3)

    def test_nearest_mrt_exit_selection(self) -> None:
        properties = pd.DataFrame({"location_key": ["A"], "latitude": [1.3001], "longitude": [103.8001]})
        exits = pd.DataFrame({
            "station_name": ["NEAR", "FAR"], "exit_code": ["A", "B"],
            "latitude": [1.3000, 1.4000], "longitude": [103.8000, 103.9000],
        })
        result = nearest_mrt_exits(properties, exits)
        self.assertEqual(result.loc[0, "nearest_mrt_station"], "NEAR")
        self.assertLess(result.loc[0, "nearest_mrt_distance_m"], 20)


class GeocodingStrategyTests(unittest.TestCase):
    def test_query_fallbacks(self) -> None:
        hdb = {"category": "hdb", "block": "123", "street_name": "ANG MO KIO AVE 3"}
        self.assertEqual([q.query for q in build_query_fallbacks(hdb)], [
            "123 ANG MO KIO AVE 3", "ANG MO KIO AVE 3",
        ])
        landed = {
            "category": "landed", "project_name": "LANDED HOUSING DEVELOPMENT",
            "street_name": "TOH TUCK CLOSE",
        }
        self.assertEqual([q.kind for q in build_query_fallbacks(landed)], ["project_street", "street"])

    def test_geocode_quality_exact_hdb(self) -> None:
        record = {
            "category": "hdb", "location_key": "123|ANG MO KIO AVE 3", "block": "123",
            "project_name": None, "street_name": "ANG MO KIO AVE 3", "postal_district": None,
        }
        candidate = build_query_fallbacks(record)[0]
        api_result = {
            "LATITUDE": "1.3700", "LONGITUDE": "103.8500", "BLK_NO": "123",
            "ROAD_NAME": "ANG MO KIO AVENUE 3", "ADDRESS": "123 ANG MO KIO AVENUE 3",
        }
        chosen = choose_result(record, candidate, [api_result])
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen[1], "exact_building")

    def test_fallback_behavior_with_mock_api(self) -> None:
        record = {
            "category": "ec", "location_key": "EXAMPLE|TEST ROAD|01", "block": None,
            "project_name": "EXAMPLE RESIDENCES", "street_name": "TEST ROAD", "postal_district": "01",
        }
        client = FakeClient([
            {"found": 0, "results": []},
            {"found": 1, "results": [{
                "LATITUDE": "1.2900", "LONGITUDE": "103.8500", "BLK_NO": "1",
                "ROAD_NAME": "TEST ROAD", "BUILDING": "EXAMPLE RESIDENCES",
                "ADDRESS": "1 TEST ROAD EXAMPLE RESIDENCES", "POSTAL": "018001",
            }]},
        ])
        result = geocode_record(record, client)
        self.assertEqual(result["geocode_status"], "success")
        self.assertEqual(result["match_quality"], "project")
        self.assertEqual(result["query_rank"], 2)
        self.assertEqual(len(client.calls), 2)

    def test_cache_reuses_success(self) -> None:
        record = {
            "category": "hdb", "location_key": "123|TEST ROAD", "block": "123",
            "project_name": None, "street_name": "TEST ROAD", "postal_district": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            cache = GeocodeCache(Path(directory) / "cache.sqlite")
            try:
                cache.seed([record])
                completed = cache.get("hdb", "123|TEST ROAD")
                completed.update({
                    "latitude": 1.3, "longitude": 103.8, "geocode_status": "success",
                    "match_quality": "exact_building", "source": "mock",
                })
                cache.update(completed)
                client = FakeClient([])
                counts = process_records(cache, [record], client)
                self.assertEqual(counts["reused_success"], 1)
                self.assertEqual(client.calls, [])
            finally:
                cache.close()

    def test_merge_preserves_rows(self) -> None:
        transactions = pd.DataFrame({"location_key": ["A", "A", "B"], "target": [1, 2, 3]})
        features = pd.DataFrame({"location_key": ["A", "B"], "latitude": [1.3, None]})
        merged = merge_location_features(transactions, features)
        self.assertEqual(len(merged), len(transactions))
        self.assertEqual(merged["target"].tolist(), [1, 2, 3])


class MatchValidationTests(unittest.TestCase):
    @staticmethod
    def _row(**overrides: object) -> dict:
        row = {
            "category": "ec", "geocode_status": "success", "match_quality": "street",
            "latitude": 1.35, "longitude": 103.90, "block": None,
            "project_name": "EXAMPLE RESIDENCES", "street_name": "TEST ROAD",
            "matched_block": "1", "matched_road": "TEST ROAD",
            "matched_building": "EXAMPLE RESIDENCES",
            "matched_address": "1 TEST ROAD EXAMPLE RESIDENCES",
        }
        row.update(overrides)
        return row

    def test_ec_exact_project_and_street_evidence_upgrades_quality(self) -> None:
        result = validate_cached_matches(pd.DataFrame([self._row()]))
        self.assertEqual(result.loc[0, "validated_match_quality"], "project")
        self.assertEqual(result.loc[0, "match_validation_status"], "accepted")

    def test_landed_street_result_is_never_upgraded(self) -> None:
        result = validate_cached_matches(pd.DataFrame([
            self._row(category="landed", project_name="GREEN GARDENS")
        ]))
        self.assertEqual(result.loc[0, "validated_match_quality"], "street")

    def test_landed_unrelated_project_result_is_rejected(self) -> None:
        result = validate_cached_matches(pd.DataFrame([
            self._row(
                category="landed", match_quality="project", project_name="GREEN GARDENS",
                matched_road="OTHER ROAD", matched_building="GREEN GARDENS PLAYGROUND",
                matched_address="2 OTHER ROAD GREEN GARDENS PLAYGROUND",
            )
        ]))
        self.assertEqual(result.loc[0, "validated_match_quality"], "unresolved")
        self.assertTrue(bool(result.loc[0, "suspicious_match"]))

    def test_hdb_commonwealth_abbreviation_is_consistent(self) -> None:
        result = validate_cached_matches(pd.DataFrame([
            self._row(
                category="hdb", match_quality="exact_building", block="88",
                project_name=None, street_name="C'WEALTH CLOSE", matched_block="88",
                matched_road="COMMONWEALTH CLOSE", matched_building="NIL",
                matched_address="88 COMMONWEALTH CLOSE SINGAPORE 140088",
            )
        ]))
        self.assertEqual(result.loc[0, "validated_match_quality"], "exact_building")

    def test_ineligible_distance_is_reference_only(self) -> None:
        cache = pd.DataFrame([{
            **self._row(
                project_name="PRIVE", matched_building="NIL",
                street_name="PUNGGOL FIELD", matched_road="PUNGGOL FIELD",
                matched_address="PUNGGOL FIELD",
            ),
            "location_key": "PRIVE|PUNGGOL FIELD|19", "postal_code": None,
            "query_used": "PUNGGOL FIELD", "query_rank": 3,
        }])
        exits = pd.DataFrame({
            "station_name": ["PUNGGOL"], "exit_code": ["B"],
            "latitude": [1.4052], "longitude": [103.9023],
        })
        features = build_location_features("ec", cache, exits)
        self.assertFalse(bool(features.loc[0, "mrt_distance_model_eligible"]))
        self.assertTrue(pd.isna(features.loc[0, "nearest_mrt_distance_m"]))
        self.assertTrue(pd.notna(features.loc[0, "nearest_mrt_distance_reference_m"]))


if __name__ == "__main__":
    unittest.main()
