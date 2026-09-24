"""Canonical preprocessing for all five HDB resale transaction files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import (
    add_temporal_features,
    assign_chronological_split,
    make_location_key,
    normalize_text,
    parse_hdb_month,
    parse_numeric,
    parse_range_series,
)


EXPECTED_BASE_COLUMNS = [
    "month", "town", "flat_type", "block", "street_name", "storey_range",
    "floor_area_sqm", "flat_model", "lease_commence_date", "resale_price",
]


def _parse_supplied_remaining_lease(series: pd.Series) -> pd.Series:
    """Standardize source-provided values only; never fill historically absent values."""
    if pd.api.types.is_numeric_dtype(series):
        return (parse_numeric(series) * 12).astype("Float64")
    parts = series.astype("string").str.extract(
        r"(?i)^\s*(?P<years>\d+)\s+years?(?:\s+(?P<months>\d+)\s+months?)?\s*$"
    )
    years = pd.to_numeric(parts["years"], errors="coerce")
    months = pd.to_numeric(parts["months"], errors="coerce").fillna(0)
    return (years * 12 + months).astype("Float64")


def derive_hdb_property_age(transaction_year: pd.Series, lease_commence_year: pd.Series) -> pd.Series:
    age = (transaction_year.astype("Float64") - lease_commence_year.astype("Float64")).astype("Float64")
    return age.mask(age.lt(0))


def preprocess_hdb(raw_directory: Path) -> pd.DataFrame:
    paths = sorted(raw_directory.glob("*.csv"))
    if len(paths) != 5:
        raise ValueError(f"Expected 5 HDB CSV files, found {len(paths)} in {raw_directory}")

    parts: list[pd.DataFrame] = []
    for path in paths:
        source = pd.read_csv(path)
        expected = EXPECTED_BASE_COLUMNS.copy()
        if "remaining_lease" in source:
            expected.insert(expected.index("resale_price"), "remaining_lease")
        if list(source.columns) != expected:
            raise ValueError(f"Unexpected HDB schema in {path.name}: {list(source.columns)}")
        source.insert(0, "source_row_number", pd.Series(range(2, len(source) + 2), dtype="Int64"))
        source.insert(0, "source_file", path.name)
        parts.append(source)

    raw = pd.concat(parts, ignore_index=True, sort=False)
    dates = parse_hdb_month(raw["month"])
    result = pd.DataFrame(index=raw.index)
    result["source_file"] = raw["source_file"].astype("string")
    result["source_row_number"] = raw["source_row_number"].astype("Int64")
    result = add_temporal_features(result, dates)
    result["town"] = normalize_text(raw["town"], uppercase=True)
    flat_type = normalize_text(raw["flat_type"], uppercase=True)
    result["flat_type"] = flat_type.replace({"MULTI-GENERATION": "MULTI GENERATION"})
    result["block"] = normalize_text(raw["block"], uppercase=True)
    result["street_name"] = normalize_text(raw["street_name"], uppercase=True)
    result["storey_range"] = normalize_text(raw["storey_range"], uppercase=True)
    result = pd.concat([result, parse_range_series(result["storey_range"], "storey")], axis=1)
    result["floor_area_sqm"] = parse_numeric(raw["floor_area_sqm"]).astype("Float64")
    result["flat_model"] = normalize_text(raw["flat_model"], uppercase=True)
    result["lease_commence_year"] = parse_numeric(raw["lease_commence_date"]).astype("Int64")
    raw_property_age = result["transaction_year"].astype("Float64") - result["lease_commence_year"].astype("Float64")
    result["property_age_at_transaction"] = derive_hdb_property_age(result["transaction_year"], result["lease_commence_year"])
    # This is a consistent 99-year arithmetic proxy, not the source remaining_lease field.
    result["approx_remaining_lease_years"] = (99 - result["property_age_at_transaction"]).astype("Float64")
    result["source_remaining_lease"] = raw["remaining_lease"].astype("string")
    result["source_remaining_lease_months"] = _parse_supplied_remaining_lease(raw["remaining_lease"])
    result["location_key"] = make_location_key(result["block"], result["street_name"])
    result["resale_price"] = parse_numeric(raw["resale_price"]).astype("Float64")

    result["quality_invalid_transaction_date"] = dates.isna()
    result["quality_invalid_target"] = result["resale_price"].isna() | result["resale_price"].le(0)
    result["quality_invalid_area"] = result["floor_area_sqm"].isna() | result["floor_area_sqm"].le(0)
    result["quality_negative_property_age"] = raw_property_age.lt(0).fillna(False)
    result["quality_unparsed_storey"] = result["storey_midpoint"].isna()
    result["quality_any"] = result[[
        "quality_invalid_transaction_date", "quality_invalid_target", "quality_invalid_area",
        "quality_negative_property_age", "quality_unparsed_storey",
    ]].any(axis=1)
    result["data_split"] = assign_chronological_split(dates, "2022-12-31", "2024-12-31")
    return result
