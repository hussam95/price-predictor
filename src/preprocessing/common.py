"""Shared deterministic preprocessing helpers."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype


def normalize_text(series: pd.Series, *, uppercase: bool = False) -> pd.Series:
    """Trim text and collapse internal whitespace without semantic rewrites."""
    result = series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    if uppercase:
        result = result.str.upper()
    return result


def parse_numeric(series: pd.Series) -> pd.Series:
    """Parse comma/currency-formatted values; source '-' placeholders become missing."""
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .replace({"-": pd.NA, "": pd.NA})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def parse_hdb_month(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%Y-%m", errors="coerce")


def parse_ura_month(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%y-%b", errors="coerce")


def add_temporal_features(data: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    result = data.copy()
    result["transaction_date"] = dates
    result["transaction_year"] = dates.dt.year.astype("Int64")
    result["transaction_month"] = dates.dt.month.astype("Int64")
    result["transaction_quarter"] = dates.dt.quarter.astype("Int64")
    return result


_RANGE_RE = re.compile(r"^\s*(\d+)\s+TO\s+(\d+)\s*$", re.IGNORECASE)


def parse_range_midpoint(value: Any) -> tuple[float, float, float]:
    """Return lower, upper, and midpoint for strings such as '06 to 10'."""
    if value is None or pd.isna(value):
        return np.nan, np.nan, np.nan
    match = _RANGE_RE.match(str(value))
    if not match:
        return np.nan, np.nan, np.nan
    lower, upper = int(match.group(1)), int(match.group(2))
    return float(lower), float(upper), (lower + upper) / 2.0


def parse_range_series(series: pd.Series, prefix: str) -> pd.DataFrame:
    parsed = pd.DataFrame(
        [parse_range_midpoint(value) for value in series],
        columns=[f"{prefix}_lower", f"{prefix}_upper", f"{prefix}_midpoint"],
        index=series.index,
    )
    for column in parsed:
        parsed[column] = parsed[column].astype("Float64")
    return parsed


_LEASE_START_RE = re.compile(
    r"^\s*(\d+)\s+(?:YRS?|YEARS?)\s+LEASE\s+COMMENCING\s+FROM\s+(\d{4})\s*$",
    re.IGNORECASE,
)
_LEASEHOLD_RE = re.compile(r"^\s*(\d+)\s+(?:YRS?|YEARS?)\s+LEASEHOLD\s*$", re.IGNORECASE)


def parse_tenure(value: Any) -> tuple[str, float, float]:
    """Return tenure category, finite duration, and commencement year."""
    if value is None or pd.isna(value) or not str(value).strip() or str(value).strip() == "-":
        return "UNKNOWN", np.nan, np.nan
    text = re.sub(r"\s+", " ", str(value).strip())
    if text.upper() == "FREEHOLD":
        return "FREEHOLD", np.nan, np.nan
    match = _LEASE_START_RE.match(text)
    if match:
        return "LEASEHOLD", float(match.group(1)), float(match.group(2))
    match = _LEASEHOLD_RE.match(text)
    if match:
        return "LEASEHOLD", float(match.group(1)), np.nan
    return "UNKNOWN", np.nan, np.nan


def parse_tenure_series(series: pd.Series) -> pd.DataFrame:
    parsed = pd.DataFrame(
        [parse_tenure(value) for value in series],
        columns=["tenure_category", "lease_duration_years", "lease_commence_year"],
        index=series.index,
    )
    parsed["tenure_category"] = parsed["tenure_category"].astype("string")
    parsed["lease_duration_years"] = parsed["lease_duration_years"].astype("Float64")
    parsed["lease_commence_year"] = parsed["lease_commence_year"].astype("Float64")
    return parsed


def derive_lease_age(
    transaction_year: pd.Series,
    lease_commence_year: pd.Series,
    lease_duration_years: pd.Series,
) -> pd.DataFrame:
    raw_age = transaction_year.astype("Float64") - lease_commence_year.astype("Float64")
    starts_after_transaction = raw_age.lt(0).fillna(False)
    meaningful_age = raw_age.mask(starts_after_transaction)
    remaining = (lease_duration_years.astype("Float64") - meaningful_age).where(
        lease_duration_years.notna() & meaningful_age.notna()
    )
    return pd.DataFrame({
        "property_age_at_transaction": meaningful_age.astype("Float64"),
        "approx_remaining_lease_years": remaining.astype("Float64"),
        "quality_lease_starts_after_transaction": starts_after_transaction.astype(bool),
    })


def make_location_key(*parts: pd.Series) -> pd.Series:
    normalized = [normalize_text(part, uppercase=True).fillna("<MISSING>") for part in parts]
    result = normalized[0]
    for part in normalized[1:]:
        result = result + "|" + part
    return result.astype("string")


def assign_chronological_split(
    dates: pd.Series,
    train_end: str,
    validation_end: str,
) -> pd.Series:
    train_end_ts = pd.Timestamp(train_end)
    validation_end_ts = pd.Timestamp(validation_end)
    result = pd.Series("test", index=dates.index, dtype="string")
    result.loc[dates <= train_end_ts] = "train"
    result.loc[(dates > train_end_ts) & (dates <= validation_end_ts)] = "validation"
    result.loc[dates.isna()] = "unassigned"
    return result


def quality_flag_summary(data: pd.DataFrame) -> dict[str, int]:
    return {
        column: int(data[column].fillna(False).astype(bool).sum())
        for column in data.columns
        if is_bool_dtype(data[column].dtype)
        and (column.startswith("quality_") or column.startswith("diagnostic_") or column.startswith("is_"))
    }
