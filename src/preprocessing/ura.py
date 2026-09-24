"""Canonical preprocessing for audited URA EC and landed transaction files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import (
    add_temporal_features,
    assign_chronological_split,
    derive_lease_age,
    make_location_key,
    normalize_text,
    parse_numeric,
    parse_range_series,
    parse_tenure_series,
    parse_ura_month,
)


EXPECTED_COLUMNS = [
    "Project Name", "Transacted Price ($)", "Area (SQFT)", "Unit Price ($ PSF)",
    "Sale Date", "Street Name", "Type of Sale", "Type of Area", "Area (SQM)",
    "Unit Price ($ PSM)", "Nett Price($)", "Property Type", "Number of Units",
    "Tenure", "Postal District", "Market Segment", "Floor Level",
]


def _canonicalize_ura(raw: pd.DataFrame, *, category: str) -> pd.DataFrame:
    if list(raw.columns) != EXPECTED_COLUMNS + ["source_file", "source_row_number"]:
        raise ValueError(f"Unexpected URA schema for {category}: {list(raw.columns)}")

    dates = parse_ura_month(raw["Sale Date"])
    result = pd.DataFrame(index=raw.index)
    result["source_file"] = raw["source_file"].astype("string")
    result["source_row_number"] = raw["source_row_number"].astype("Int64")
    result = add_temporal_features(result, dates)
    result["project_name"] = normalize_text(raw["Project Name"], uppercase=True)
    result["street_name"] = normalize_text(raw["Street Name"], uppercase=True)
    result["sale_type"] = normalize_text(raw["Type of Sale"], uppercase=True)
    result["area_type"] = normalize_text(raw["Type of Area"], uppercase=True)
    result["area_sqm"] = parse_numeric(raw["Area (SQM)"]).astype("Float64")
    result["property_type"] = normalize_text(raw["Property Type"], uppercase=True)
    result["number_of_units"] = parse_numeric(raw["Number of Units"]).astype("Int64")
    result["tenure"] = normalize_text(raw["Tenure"])
    tenure = parse_tenure_series(result["tenure"])
    result = pd.concat([result, tenure], axis=1)
    result["postal_district"] = parse_numeric(raw["Postal District"]).astype("Int64")
    result["market_segment"] = normalize_text(raw["Market Segment"], uppercase=True)
    result["floor_level"] = normalize_text(raw["Floor Level"], uppercase=True).replace("-", pd.NA)
    floor = parse_range_series(result["floor_level"], "floor")
    result = pd.concat([result, floor], axis=1)
    lease_age = derive_lease_age(
        result["transaction_year"], result["lease_commence_year"], result["lease_duration_years"]
    )
    result = pd.concat([result, lease_age], axis=1)
    district_key = result["postal_district"].astype("string").str.zfill(2)
    result["location_key"] = make_location_key(result["project_name"], result["street_name"], district_key)
    result["transaction_price"] = parse_numeric(raw["Transacted Price ($)"]).astype("Float64")
    result["is_multi_unit_transaction"] = result["number_of_units"].gt(1).fillna(False)

    result["quality_invalid_transaction_date"] = dates.isna()
    result["quality_invalid_target"] = result["transaction_price"].isna() | result["transaction_price"].le(0)
    result["quality_invalid_area"] = result["area_sqm"].isna() | result["area_sqm"].le(0)
    result["quality_unparsed_tenure"] = result["tenure_category"].eq("UNKNOWN")
    result["quality_leasehold_missing_commencement"] = (
        result["tenure_category"].eq("LEASEHOLD") & result["lease_commence_year"].isna()
    )
    result["quality_unparsed_floor"] = result["floor_level"].notna() & result["floor_midpoint"].isna()
    result["quality_any"] = result[[
        "quality_invalid_transaction_date", "quality_invalid_target", "quality_invalid_area",
        "quality_lease_starts_after_transaction", "quality_unparsed_tenure",
        "quality_leasehold_missing_commencement", "quality_unparsed_floor",
    ]].any(axis=1)
    result["data_split"] = assign_chronological_split(dates, "2023-12-31", "2024-12-31")
    return result


def _read_ura(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    if list(data.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected URA schema in {path.name}: {list(data.columns)}")
    data["source_file"] = path.name
    data["source_row_number"] = pd.Series(range(2, len(data) + 2), dtype="Int64")
    return data


def preprocess_ec(raw_directory: Path) -> pd.DataFrame:
    paths = sorted(raw_directory.glob("*.csv"))
    if len(paths) != 2:
        raise ValueError(f"Expected 2 EC CSV files, found {len(paths)} in {raw_directory}")
    raw = pd.concat([_read_ura(path) for path in paths], ignore_index=True)
    return _canonicalize_ura(raw, category="ec")


def preprocess_landed(raw_file: Path) -> pd.DataFrame:
    raw = _read_ura(raw_file)
    result = _canonicalize_ura(raw, category="landed")
    prices = result["transaction_price"]
    q1, q3 = prices.quantile([0.25, 0.75])
    upper_fence = q3 + 1.5 * (q3 - q1)
    # Target-derived diagnostic: explicitly excluded from modeling in the manifest.
    result["diagnostic_price_outlier_iqr"] = prices.gt(upper_fence).fillna(False)
    return result
