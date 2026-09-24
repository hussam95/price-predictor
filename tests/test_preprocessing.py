"""Focused unit tests for deterministic canonical transformations."""

from __future__ import annotations

import unittest

import pandas as pd

from src.preprocess_data import build_feature_manifest, candidate_features, validate_manifest
from src.preprocessing.common import (
    parse_hdb_month,
    parse_numeric,
    parse_range_midpoint,
    parse_tenure,
)
from src.preprocessing.hdb import derive_hdb_property_age


class CommonTransformationTests(unittest.TestCase):
    def test_hdb_storey_midpoint(self) -> None:
        self.assertEqual(parse_range_midpoint("01 TO 03"), (1.0, 3.0, 2.0))
        self.assertEqual(parse_range_midpoint("36 TO 40"), (36.0, 40.0, 38.0))
        self.assertTrue(pd.isna(parse_range_midpoint("-")[2]))

    def test_hdb_date_parsing(self) -> None:
        parsed = parse_hdb_month(pd.Series(["1990-01", "2026-02", "bad"]))
        self.assertEqual(parsed.iloc[0], pd.Timestamp("1990-01-01"))
        self.assertEqual(parsed.iloc[1], pd.Timestamp("2026-02-01"))
        self.assertTrue(pd.isna(parsed.iloc[2]))

    def test_hdb_property_age(self) -> None:
        age = derive_hdb_property_age(
            pd.Series([2020, 2025], dtype="Int64"),
            pd.Series([1980, 2015], dtype="Int64"),
        )
        self.assertEqual(age.tolist(), [40.0, 10.0])

    def test_ura_numeric_parsing(self) -> None:
        parsed = parse_numeric(pd.Series(["1,718,000", "$2,500.50", "-", " 79 "]))
        self.assertEqual(parsed.iloc[0], 1718000.0)
        self.assertEqual(parsed.iloc[1], 2500.5)
        self.assertTrue(pd.isna(parsed.iloc[2]))
        self.assertEqual(parsed.iloc[3], 79.0)

    def test_ura_floor_midpoint(self) -> None:
        self.assertEqual(parse_range_midpoint("06 to 10"), (6.0, 10.0, 8.0))
        self.assertEqual(parse_range_midpoint("31 to 35"), (31.0, 35.0, 33.0))

    def test_tenure_parsing_real_formats(self) -> None:
        freehold = parse_tenure("Freehold")
        self.assertEqual(freehold[0], "FREEHOLD")
        self.assertTrue(pd.isna(freehold[1]))
        self.assertTrue(pd.isna(freehold[2]))
        self.assertEqual(parse_tenure("99 yrs lease commencing from 2014"), ("LEASEHOLD", 99.0, 2014.0))
        self.assertEqual(parse_tenure("999 yrs lease commencing from 1879"), ("LEASEHOLD", 999.0, 1879.0))
        self.assertEqual(parse_tenure("956 yrs lease commencing from 1928"), ("LEASEHOLD", 956.0, 1928.0))
        generic_lease = parse_tenure("99 years leasehold")
        self.assertEqual(generic_lease[:2], ("LEASEHOLD", 99.0))
        self.assertTrue(pd.isna(generic_lease[2]))


class LeakageManifestTests(unittest.TestCase):
    def test_manifest_validation(self) -> None:
        manifest = build_feature_manifest()
        validate_manifest(manifest)
        for category in ("hdb", "ec", "landed"):
            candidates = candidate_features(manifest[category])
            self.assertNotIn(manifest[category]["target"], candidates)
            joined = " ".join(candidates).lower()
            self.assertNotIn("unit_price", joined)
            self.assertNotIn("nett_price", joined)

    def test_manifest_rejects_target_as_candidate(self) -> None:
        manifest = build_feature_manifest()
        manifest["ec"]["safe_numeric_candidate_features"].append("transaction_price")
        with self.assertRaises(ValueError):
            validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
