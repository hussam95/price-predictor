"""Input normalization and validation for local prediction requests."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd

from src.preprocessing.common import parse_range_midpoint


class InputValidationError(ValueError):
    """Raised when a prediction request contains impossible or missing input."""


def valuation_timestamp(value: str | date | datetime | pd.Timestamp) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise InputValidationError("Valuation month must be a valid date or YYYY-MM value.") from error
    if pd.isna(timestamp):
        raise InputValidationError("Valuation month is required.")
    return timestamp.to_period("M").start_time


def required_text(inputs: dict[str, Any], name: str) -> str:
    value = str(inputs.get(name, "")).strip().upper()
    if not value:
        raise InputValidationError(f"{name.replace('_', ' ').title()} is required.")
    return " ".join(value.split())


def positive_number(inputs: dict[str, Any], name: str) -> float:
    try:
        value = float(inputs.get(name))
    except (TypeError, ValueError) as error:
        raise InputValidationError(f"{name.replace('_', ' ').title()} must be numeric.") from error
    if value <= 0:
        raise InputValidationError(f"{name.replace('_', ' ').title()} must be greater than zero.")
    return value


def optional_number(inputs: dict[str, Any], name: str) -> float | None:
    value = inputs.get(name)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise InputValidationError(f"{name.replace('_', ' ').title()} must be numeric when supplied.") from error


def storey_midpoint(inputs: dict[str, Any]) -> float:
    if inputs.get("storey_midpoint") is not None:
        return positive_number(inputs, "storey_midpoint")
    value = inputs.get("storey_range")
    _, _, midpoint = parse_range_midpoint(value)
    if pd.isna(midpoint):
        raise InputValidationError("Storey range must use a known format such as '10 TO 12'.")
    return float(midpoint)


def lease_values(inputs: dict[str, Any], valuation_year: int) -> tuple[str, float | None, float | None]:
    category = required_text(inputs, "tenure_category")
    if category == "FREEHOLD":
        return category, None, None
    duration = optional_number(inputs, "lease_duration_years")
    commencement = optional_number(inputs, "lease_commence_year")
    if category == "LEASEHOLD":
        if duration is None or duration <= 0:
            raise InputValidationError("Lease duration must be positive for leasehold property.")
        if commencement is None:
            raise InputValidationError("Lease commencement year is required for leasehold property.")
        if int(commencement) > valuation_year:
            raise InputValidationError("Lease commencement year cannot be after the valuation year.")
        return category, duration, float(valuation_year - int(commencement))
    return category, duration, None if commencement is None else float(valuation_year - int(commencement))


def range_warning(name: str, value: float, bounds: dict[str, float]) -> str | None:
    minimum, maximum = float(bounds["min"]), float(bounds["max"])
    if value < minimum or value > maximum:
        return (
            f"{name} ({value:g}) is outside the observed production-training range "
            f"({minimum:g}–{maximum:g}); the estimate may be less reliable."
        )
    return None

