"""Generate compact local UI references and pre-aggregated market summaries.

This is an offline, deterministic read of processed datasets. It performs no
geocoding, network access, or model training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if pd.isna(value):
        return None
    return value


def sorted_strings(series: pd.Series) -> list[str]:
    return sorted(series.dropna().astype(str).unique().tolist())


def numeric_range(frame: pd.DataFrame, column: str) -> dict[str, float]:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return {"min": float(values.min()), "max": float(values.max()), "median": float(values.median())}


def monthly_prices(frame: pd.DataFrame, target: str) -> list[dict[str, Any]]:
    grouped = frame.groupby("transaction_date", observed=True).agg(
        median_price=(target, "median"), transactions=(target, "size")
    ).reset_index()
    return native(grouped.to_dict("records"))


def category_summary(frame: pd.DataFrame, column: str, target: str, limit: int | None = None) -> list[dict[str, Any]]:
    grouped = frame.groupby(column, observed=True).agg(
        transactions=(target, "size"), median_price=(target, "median")
    ).reset_index().sort_values("transactions", ascending=False)
    if limit:
        grouped = grouped.head(limit)
    return native(grouped.to_dict("records"))


def main() -> None:
    contract = json.loads((MODELS / "inference_contract.json").read_text(encoding="utf-8"))

    hdb_columns = [
        "transaction_date", "data_split", "town", "flat_type", "flat_model", "block",
        "street_name", "storey_range", "floor_area_sqm", "storey_midpoint",
        "lease_commence_year", "location_key", "resale_price", "geocode_quality",
        "matched_address", "nearest_mrt_station", "nearest_mrt_distance_reference_m",
        "mrt_distance_model_eligible", "nearest_mrt_distance_m",
    ]
    hdb = pd.read_csv(PROCESSED / "hdb_enriched.csv", usecols=hdb_columns, low_memory=False)
    hdb["block"] = hdb["block"].astype("string")
    hdb_locations = hdb[[
        "location_key", "block", "street_name", "geocode_quality", "matched_address",
        "nearest_mrt_station", "nearest_mrt_distance_reference_m",
        "mrt_distance_model_eligible", "nearest_mrt_distance_m",
    ]].drop_duplicates("location_key")
    hdb_locations.to_csv(PROCESSED / "hdb_location_lookup.csv", index=False)

    hdb_fit = hdb.loc[hdb["data_split"].isin(["train", "validation"])]
    town_streets = {
        town: sorted_strings(group["street_name"])
        for town, group in hdb.groupby("town", observed=True)
    }
    street_blocks = {
        f"{town}|{street}": sorted_strings(group["block"])
        for (town, street), group in hdb.groupby(["town", "street_name"], observed=True)
    }

    ec_columns = [
        "transaction_date", "data_split", "project_name", "street_name", "sale_type",
        "area_type", "area_sqm", "property_type", "tenure", "tenure_category",
        "lease_duration_years", "lease_commence_year", "postal_district", "market_segment",
        "floor_level", "floor_midpoint", "transaction_price", "nearest_mrt_station",
        "nearest_mrt_distance_reference_m", "geocode_quality",
    ]
    ec = pd.read_csv(PROCESSED / "ec_enriched.csv", usecols=ec_columns, low_memory=False)
    ec["postal_district"] = ec["postal_district"].astype("Int64").astype("string").str.zfill(2)
    ec_fit = ec.loc[ec["data_split"].isin(["train", "validation"])]
    ec_context: dict[str, Any] = {}
    for project, group in ec.groupby("project_name", observed=True):
        first = group.iloc[0]
        ec_context[" ".join(str(project).strip().upper().split())] = native({
            "streets": sorted_strings(group["street_name"]),
            "postal_districts": sorted_strings(group["postal_district"]),
            "property_types": sorted_strings(group["property_type"]),
            "tenure_categories": sorted_strings(group["tenure_category"]),
            "lease_durations": sorted(pd.to_numeric(group["lease_duration_years"], errors="coerce").dropna().unique().tolist()),
            "lease_commence_years": sorted(pd.to_numeric(group["lease_commence_year"], errors="coerce").dropna().astype(int).unique().tolist()),
            "nearest_mrt_station": first["nearest_mrt_station"],
            "nearest_mrt_distance_m": first["nearest_mrt_distance_reference_m"],
            "location_quality": first["geocode_quality"],
        })

    landed_columns = [
        "transaction_date", "data_split", "sale_type", "area_type", "area_sqm",
        "property_type", "tenure", "tenure_category", "lease_duration_years",
        "lease_commence_year", "postal_district", "market_segment", "transaction_price",
    ]
    landed = pd.read_csv(PROCESSED / "landed_enriched.csv", usecols=landed_columns, low_memory=False)
    landed["postal_district"] = landed["postal_district"].astype("Int64").astype("string").str.zfill(2)
    landed_fit = landed.loc[landed["data_split"].isin(["train", "validation"])]

    references = {
        "version": 1,
        "hdb": {
            "towns": sorted_strings(hdb["town"]),
            "flat_types": sorted_strings(hdb["flat_type"]),
            "flat_models": sorted_strings(hdb["flat_model"]),
            "storey_ranges": sorted_strings(hdb["storey_range"]),
            "town_streets": town_streets,
            "street_blocks": street_blocks,
            "training_ranges": {
                column: numeric_range(hdb_fit, column)
                for column in ("floor_area_sqm", "storey_midpoint", "lease_commence_year")
            },
        },
        "ec": {
            "projects": sorted_strings(ec["project_name"]),
            "property_types": sorted_strings(ec["property_type"]),
            "sale_types": sorted_strings(ec["sale_type"]),
            "area_types": sorted_strings(ec["area_type"]),
            "postal_districts": sorted_strings(ec["postal_district"]),
            "market_segments": sorted_strings(ec["market_segment"]),
            "tenure_categories": sorted_strings(ec["tenure_category"]),
            "floor_levels": sorted_strings(ec["floor_level"]),
            "project_context": ec_context,
            "training_ranges": {"area_sqm": numeric_range(ec_fit, "area_sqm")},
        },
        "landed": {
            "property_types": sorted_strings(landed["property_type"]),
            "sale_types": sorted_strings(landed["sale_type"]),
            "area_types": sorted_strings(landed["area_type"]),
            "postal_districts": sorted_strings(landed["postal_district"]),
            "market_segments": sorted_strings(landed["market_segment"]),
            "tenure_categories": sorted_strings(landed["tenure_category"]),
            "training_ranges": {"area_sqm": numeric_range(landed_fit, "area_sqm")},
        },
        "contract_valid_categories": {
            category: {
                item["name"]: item["valid_values"]
                for item in section["inputs"] if "valid_values" in item
            }
            for category, section in contract["categories"].items()
        },
    }
    (PROCESSED / "ui_reference_data.json").write_text(
        json.dumps(native(references), indent=2) + "\n", encoding="utf-8"
    )

    rpi = pd.read_csv(PROCESSED / "hdb_rpi.csv")
    market = {
        "version": 1,
        "hdb": {
            "monthly": monthly_prices(hdb, "resale_price"),
            "towns": category_summary(hdb, "town", "resale_price"),
            "flat_types": category_summary(hdb, "flat_type", "resale_price"),
            "rpi": native(rpi[["year", "quarter", "rpi"]].to_dict("records")),
        },
        "ec": {
            "monthly": monthly_prices(ec, "transaction_price"),
            "projects": category_summary(ec, "project_name", "transaction_price", limit=25),
            "sale_types": category_summary(ec, "sale_type", "transaction_price"),
            "market_segments": category_summary(ec, "market_segment", "transaction_price"),
        },
        "landed": {
            "monthly": monthly_prices(landed, "transaction_price"),
            "property_types": category_summary(landed, "property_type", "transaction_price"),
            "districts": category_summary(landed, "postal_district", "transaction_price"),
            "market_segments": category_summary(landed, "market_segment", "transaction_price"),
        },
    }
    (PROCESSED / "market_summary.json").write_text(
        json.dumps(native(market), indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote UI references, {len(hdb_locations):,} HDB locations, and market summaries."
    )


if __name__ == "__main__":
    main()
