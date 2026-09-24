"""Single local inference adapter for all frozen property models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.preprocessing.common import parse_range_midpoint

from .reference_data import (
    MODELS,
    load_calibration,
    load_contract,
    load_ui_reference,
    lookup_hdb_location,
    lookup_lagged_rpi,
    lookup_project_context,
)
from .validation import (
    InputValidationError,
    lease_values,
    optional_number,
    positive_number,
    range_warning,
    required_text,
    storey_midpoint,
    valuation_timestamp,
)


class InferenceError(RuntimeError):
    """User-actionable local inference failure."""


@dataclass(frozen=True)
class PredictionResult:
    category: str
    predicted_price: float
    range_lower: float
    range_upper: float
    range_percentile: str
    historical_error_amount: float
    calibration_band: str
    valuation_month: str
    model_name: str
    model_test_mae: float
    derived: dict[str, Any]
    references: dict[str, Any]
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LoadedModel:
    estimator: object
    metadata: dict[str, Any]


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise InferenceError(f"Missing {label}: {path}") from error
    except json.JSONDecodeError as error:
        raise InferenceError(f"Invalid {label}: {path}") from error


@lru_cache(maxsize=1)
def load_production_models() -> dict[str, LoadedModel]:
    loaded: dict[str, LoadedModel] = {}
    for category in ("hdb", "ec", "landed"):
        model_path = MODELS / f"{category}_model.joblib"
        metadata_path = MODELS / f"{category}_model_metadata.json"
        metadata = _load_json(metadata_path, f"{category} model metadata")
        try:
            estimator = joblib.load(model_path)
        except FileNotFoundError as error:
            raise InferenceError(f"Missing frozen model: {model_path}") from error
        except Exception as error:
            raise InferenceError(f"Could not load frozen {category} model: {error}") from error
        loaded[category] = LoadedModel(estimator, metadata)
    return loaded


def _months_since_1990(timestamp: pd.Timestamp) -> int:
    return (timestamp.year - 1990) * 12 + timestamp.month - 1


def _category_warning(category: str, column: str, value: str) -> str | None:
    contract = load_contract()["categories"][category]
    item = next((field for field in contract["inputs"] if field["name"] == column), None)
    if item and item.get("valid_values") and value not in item["valid_values"]:
        return (
            f"{column.replace('_', ' ').title()} is not present in production training data. "
            "The fitted unknown-category behavior will be used."
        )
    return None


def _calibrated_range(category: str, prediction: float, percentile: str = "p80") -> tuple[float, float, float, str]:
    config = load_calibration()[category]
    edges = config["band_edges_sgd"]
    labels = config["band_labels"]
    selected = "global"
    error = float(config["global"][f"{percentile}_absolute_error"])
    for index, label in enumerate(labels):
        lower = -np.inf if edges[index] is None else float(edges[index])
        upper = np.inf if edges[index + 1] is None else float(edges[index + 1])
        if lower <= prediction < upper and label in config["by_predicted_price_band"]:
            selected = label
            error = float(config["by_predicted_price_band"][label][percentile])
            break
    return max(0.0, prediction - error), prediction + error, error, selected


class PropertyPredictor:
    """Construct contract-exact feature rows and invoke frozen pipelines."""

    def __init__(self) -> None:
        self.contract = load_contract()
        self.models = load_production_models()
        self.references = load_ui_reference()

    def predict(self, category: str, inputs: dict[str, Any]) -> PredictionResult:
        normalized = category.strip().lower()
        if normalized == "hdb":
            return self.predict_hdb(inputs)
        if normalized == "ec":
            return self.predict_ec(inputs)
        if normalized == "landed":
            return self.predict_landed(inputs)
        raise InputValidationError("Category must be hdb, ec, or landed.")

    def _predict_row(
        self,
        category: str,
        row: dict[str, Any],
        valuation: pd.Timestamp,
        derived: dict[str, Any],
        references: dict[str, Any],
        warnings: list[str],
        limitations: list[str],
    ) -> PredictionResult:
        loaded = self.models[category]
        features = self.contract["categories"][category]["model_features_in_order"]
        missing = [feature for feature in features if feature not in row]
        if missing:
            raise InferenceError(f"Internal inference row is missing model features: {missing}")
        frame = pd.DataFrame([{feature: row[feature] for feature in features}])
        try:
            prediction = float(np.asarray(loaded.estimator.predict(frame), dtype=float)[0])
        except Exception as error:
            raise InferenceError(f"Frozen {category.upper()} model could not produce a prediction: {error}") from error
        if not np.isfinite(prediction) or prediction <= 0:
            raise InferenceError("Model returned an invalid non-positive price.")
        lower, upper, amount, band = _calibrated_range(category, prediction, "p80")
        metrics = loaded.metadata["final_test_metrics"]
        return PredictionResult(
            category=category,
            predicted_price=prediction,
            range_lower=lower,
            range_upper=upper,
            range_percentile="P80",
            historical_error_amount=amount,
            calibration_band=band,
            valuation_month=valuation.strftime("%Y-%m"),
            model_name=str(loaded.metadata["model_type"]),
            model_test_mae=float(metrics["mae"]),
            derived=derived,
            references=references,
            warnings=tuple(warnings),
            limitations=tuple(limitations),
        )

    def predict_hdb(self, inputs: dict[str, Any]) -> PredictionResult:
        valuation = valuation_timestamp(inputs.get("valuation_month"))
        area = positive_number(inputs, "floor_area_sqm")
        storey = storey_midpoint(inputs)
        try:
            lease_year = int(inputs.get("lease_commence_year"))
        except (TypeError, ValueError) as error:
            raise InputValidationError("Lease commencement year must be an integer.") from error
        if lease_year > valuation.year:
            raise InputValidationError("Lease commencement year cannot be after the valuation year.")
        town = required_text(inputs, "town")
        flat_type = required_text(inputs, "flat_type")
        flat_model = required_text(inputs, "flat_model")
        block = required_text(inputs, "block")
        street = required_text(inputs, "street_name")
        rpi = lookup_lagged_rpi(valuation)
        location = lookup_hdb_location(block, street)
        mrt_distance = None
        references: dict[str, Any] = {
            "address": f"{block} {street}",
            "rpi_source_quarter": rpi["rpi_source_quarter"],
            "lagged_rpi": rpi["lagged_rpi"],
            "rpi_fallback_used": rpi["is_latest_available_fallback"],
            "mrt_used_by_model": False,
        }
        warnings: list[str] = []
        if location:
            references.update({
                "matched_address": location["matched_address"],
                "nearest_mrt_station": location["nearest_mrt_station"],
                "nearest_mrt_distance_m": location["reference_distance_m"],
                "geocode_quality": location["geocode_quality"],
            })
            if location["eligible"] and location["nearest_mrt_distance_m"] is not None:
                mrt_distance = location["nearest_mrt_distance_m"]
                references["mrt_used_by_model"] = True
            else:
                warnings.append("The local address match is not ML-eligible; MRT distance was treated as missing by the model.")
        else:
            warnings.append("Address was not found in the local cache; MRT distance was treated as missing. No live lookup was made.")
        if rpi["is_latest_available_fallback"]:
            warnings.append(
                f"The immediately prior quarter is unavailable locally; latest older official RPI {rpi['rpi_source_quarter']} was used."
            )
        bounds = self.references["hdb"]["training_ranges"]
        for name, value in (("Floor area", area), ("Storey midpoint", storey), ("Lease commencement year", lease_year)):
            warning = range_warning(name, float(value), bounds[{"Floor area": "floor_area_sqm", "Storey midpoint": "storey_midpoint", "Lease commencement year": "lease_commence_year"}[name]])
            if warning:
                warnings.append(warning)
        for column, value in (("town", town), ("flat_type", flat_type), ("flat_model", flat_model)):
            warning = _category_warning("hdb", column, value)
            if warning:
                warnings.append(warning)
        row = {
            "floor_area_sqm": area,
            "storey_midpoint": storey,
            "lease_commence_year": lease_year,
            "property_age_at_transaction": valuation.year - lease_year,
            "months_since_1990": _months_since_1990(valuation),
            "lagged_rpi": float(rpi["lagged_rpi"]),
            "nearest_mrt_distance_m": np.nan if mrt_distance is None else mrt_distance,
            "mrt_distance_missing": mrt_distance is None,
            "town": town,
            "flat_type": flat_type,
            "flat_model": flat_model,
        }
        return self._predict_row(
            "hdb", row, valuation,
            derived={
                "property_age_at_transaction": row["property_age_at_transaction"],
                "months_since_1990": row["months_since_1990"],
                "lagged_rpi": row["lagged_rpi"],
                "mrt_distance_missing": row["mrt_distance_missing"],
            },
            references=references,
            warnings=warnings,
            limitations=["Estimate is not a professional valuation."],
        )

    def _predict_ura(self, category: str, inputs: dict[str, Any]) -> PredictionResult:
        valuation = valuation_timestamp(inputs.get("valuation_month"))
        area = positive_number(inputs, "area_sqm")
        tenure, duration, age = lease_values(inputs, valuation.year)
        property_type = required_text(inputs, "property_type")
        sale_type = required_text(inputs, "sale_type")
        area_type = required_text(inputs, "area_type")
        district_text = required_text(inputs, "postal_district")
        try:
            district = int(district_text)
        except ValueError as error:
            raise InputValidationError("Postal district must be a number.") from error
        if district < 1 or district > 28:
            raise InputValidationError("Postal district must be between 1 and 28.")
        segment = required_text(inputs, "market_segment")
        warnings: list[str] = []
        bounds = self.references[category]["training_ranges"]
        warning = range_warning("Area", area, bounds["area_sqm"])
        if warning:
            warnings.append(warning)
        categorical = {
            "sale_type": sale_type, "area_type": area_type, "property_type": property_type,
            "tenure_category": tenure, "postal_district": district, "market_segment": segment,
        }
        for column, value in categorical.items():
            warning = _category_warning(category, column, str(value))
            if warning:
                warnings.append(warning)
        row: dict[str, Any] = {
            "area_sqm": area,
            "number_of_units": 1,
            "lease_duration_years": np.nan if duration is None else duration,
            "property_age_at_transaction": np.nan if age is None else age,
            "months_since_1990": _months_since_1990(valuation),
            **categorical,
        }
        references: dict[str, Any] = {"mrt_used_by_model": False}
        limitations = ["Estimate is not a professional valuation."]
        if category == "ec":
            project = required_text(inputs, "project_name")
            row["project_name"] = project
            floor = optional_number(inputs, "floor_midpoint")
            if floor is None and inputs.get("floor_level"):
                _, _, parsed = parse_range_midpoint(inputs["floor_level"])
                floor = None if pd.isna(parsed) else float(parsed)
            row["floor_midpoint"] = np.nan if floor is None else floor
            warning = _category_warning("ec", "project_name", project)
            if warning:
                warnings.append(warning)
            context = lookup_project_context("ec", project)
            if context:
                references.update(context)
                references["mrt_used_by_model"] = False
        else:
            row["is_multi_unit_transaction"] = False
            limitations.append(
                "Landed estimates are less precise because the market is heterogeneous; multi-unit transactions are outside scope."
            )
        return self._predict_row(
            category, row, valuation,
            derived={
                "months_since_1990": row["months_since_1990"],
                "property_age_at_transaction": None if age is None else age,
                "number_of_units": 1,
                "is_multi_unit_transaction": False,
            },
            references=references,
            warnings=warnings,
            limitations=limitations,
        )

    def predict_ec(self, inputs: dict[str, Any]) -> PredictionResult:
        return self._predict_ura("ec", inputs)

    def predict_landed(self, inputs: dict[str, Any]) -> PredictionResult:
        if inputs.get("number_of_units") not in (None, 1, "1"):
            raise InputValidationError("Landed prediction supports exactly one individual property.")
        return self._predict_ura("landed", {**inputs, "number_of_units": 1})


@lru_cache(maxsize=1)
def get_predictor() -> PropertyPredictor:
    return PropertyPredictor()


def predict_property(category: str, inputs: dict[str, Any]) -> PredictionResult:
    return get_predictor().predict(category, inputs)


def predict_hdb(**inputs: Any) -> PredictionResult:
    return get_predictor().predict_hdb(inputs)


def predict_ec(**inputs: Any) -> PredictionResult:
    return get_predictor().predict_ec(inputs)


def predict_landed(**inputs: Any) -> PredictionResult:
    return get_predictor().predict_landed(inputs)
