"""Compact local reference-data access; no network operations."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from src.modeling.rpi import load_rpi, rpi_for_valuation_month


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def normalize_key_part(value: Any) -> str:
    return " ".join(str(value).strip().upper().split())


def hdb_location_key(block: Any, street_name: Any) -> str:
    return f"{normalize_key_part(block)}|{normalize_key_part(street_name)}"


@lru_cache(maxsize=1)
def load_ui_reference() -> dict[str, Any]:
    path = PROCESSED / "ui_reference_data.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m src.inference.build_ui_artifacts"
        )
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_market_summary() -> dict[str, Any]:
    path = PROCESSED / "market_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m src.inference.build_ui_artifacts"
        )
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_hdb_locations() -> pd.DataFrame:
    path = PROCESSED / "hdb_location_lookup.csv"
    frame = pd.read_csv(path, dtype={"block": "string", "street_name": "string"})
    if frame["location_key"].duplicated().any():
        raise ValueError("HDB location lookup contains duplicate location keys")
    return frame.set_index("location_key", drop=False)


def lookup_hdb_location(block: Any, street_name: Any) -> dict[str, Any] | None:
    key = hdb_location_key(block, street_name)
    locations = load_hdb_locations()
    if key not in locations.index:
        return None
    row = locations.loc[key]
    eligible_value = row["mrt_distance_model_eligible"]
    eligible = (
        bool(eligible_value)
        if isinstance(eligible_value, bool)
        else str(eligible_value).strip().lower() in {"true", "1", "yes"}
    )
    return {
        "location_key": key,
        "eligible": eligible,
        "geocode_quality": None if pd.isna(row["geocode_quality"]) else str(row["geocode_quality"]),
        "matched_address": None if pd.isna(row["matched_address"]) else str(row["matched_address"]),
        "nearest_mrt_station": None if pd.isna(row["nearest_mrt_station"]) else str(row["nearest_mrt_station"]),
        "nearest_mrt_distance_m": None if pd.isna(row["nearest_mrt_distance_m"]) else float(row["nearest_mrt_distance_m"]),
        "reference_distance_m": None if pd.isna(row["nearest_mrt_distance_reference_m"]) else float(row["nearest_mrt_distance_reference_m"]),
    }


@lru_cache(maxsize=1)
def load_calibration() -> dict[str, Any]:
    return json.loads((MODELS / "error_calibration.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_contract() -> dict[str, Any]:
    return json.loads((MODELS / "inference_contract.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_rpi_reference() -> pd.DataFrame:
    return load_rpi(PROCESSED / "hdb_rpi.csv")


def lookup_lagged_rpi(valuation_month: Any) -> dict[str, object]:
    return rpi_for_valuation_month(valuation_month, load_rpi_reference())


def lookup_project_context(category: str, project_name: str) -> dict[str, Any] | None:
    contexts = load_ui_reference().get(category, {}).get("project_context", {})
    return contexts.get(normalize_key_part(project_name))
