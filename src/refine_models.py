"""Limited validation-driven refinement and final production model freeze.

Configuration selection uses training and validation only. Test predictions are
made only after each category's configuration is fixed and refit on train plus
validation.

Run:
    .venv/bin/python -m src.refine_models
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import TransformedTargetRegressor

from .modeling.common import (
    RANDOM_SEED,
    FeatureSet,
    IndexedTargetRegressor,
    absolute_error_percentiles,
    make_hist_pipeline,
    make_linear_pipeline,
    regression_metrics,
)
from .modeling.rpi import add_hdb_rpi_features, load_rpi, rpi_for_valuation_month


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "modeling"
MODEL_DIR = ROOT / "models"
BASELINE_RESULTS = REPORT_DIR / "selected_model_metrics.csv"

ORIGINAL_TEST = {
    "hdb": {"mae": 120_335.929073246, "rmse": 138_261.04358661606, "r2": 0.5412509542697682},
    "ec": {"mae": 130_501.52401630791, "rmse": 160_181.95197852905, "r2": 0.7685807136286751},
    "landed": {"mae": 1_205_641.970790837, "rmse": 2_944_451.685461361, "r2": 0.7087647205900913},
}

PRICE_BANDS = {
    "hdb": ([-np.inf, 400_000, 600_000, 800_000, np.inf], ["under_400k", "400k_600k", "600k_800k", "800k_plus"]),
    "ec": ([-np.inf, 1_200_000, 1_600_000, 2_000_000, np.inf], ["under_1_2m", "1_2m_1_6m", "1_6m_2m", "2m_plus"]),
    "landed": ([-np.inf, 4_000_000, 7_000_000, 12_000_000, np.inf], ["under_4m", "4m_7m", "7m_12m", "12m_plus"]),
}


@dataclass(frozen=True)
class Experiment:
    name: str
    features: FeatureSet
    factory: Callable[[], object]
    selectable: bool = True
    notes: str = ""
    requires_lagged_rpi: bool = False


def fit_predict(experiment: Experiment, train: pd.DataFrame, validation: pd.DataFrame, target: str) -> tuple[object, np.ndarray, float, int]:
    fitting = train
    if experiment.requires_lagged_rpi:
        fitting = fitting.loc[fitting["lagged_rpi"].notna()]
    started = time.perf_counter()
    model = experiment.factory()
    model.fit(fitting[experiment.features.columns], fitting[target])
    seconds = time.perf_counter() - started
    prediction = np.asarray(model.predict(validation[experiment.features.columns]), dtype=float)
    return model, prediction, seconds, len(fitting)


def metric_row(category: str, experiment: str, split: str, y: pd.Series, prediction: np.ndarray, **extra: object) -> dict[str, object]:
    residual = prediction - y.to_numpy(dtype=float)
    return {
        "category": category,
        "experiment": experiment,
        "evaluation_split": split,
        **regression_metrics(y, prediction),
        "mean_residual_prediction_minus_actual": float(residual.mean()),
        **extra,
    }


def add_month_index(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["months_since_1990"] = (
        (result["transaction_year"].astype(int) - 1990) * 12
        + result["transaction_month"].astype(int) - 1
    )
    return result


def hdb_experiments() -> list[Experiment]:
    physical = ("floor_area_sqm", "storey_midpoint", "lease_commence_year", "property_age_at_transaction")
    categories = ("town", "flat_type", "flat_model")
    current = FeatureSet((*physical, "transaction_year", "transaction_month", "nearest_mrt_distance_m", "mrt_distance_missing"), categories)
    lag_mrt = FeatureSet((*physical, "transaction_year", "transaction_month", "lagged_rpi", "nearest_mrt_distance_m", "mrt_distance_missing"), categories)
    lag_no_mrt = FeatureSet((*physical, "transaction_year", "transaction_month", "lagged_rpi"), categories)
    month_lag_mrt = FeatureSet((*physical, "months_since_1990", "lagged_rpi", "nearest_mrt_distance_m", "mrt_distance_missing"), categories)
    month_lag_no_mrt = FeatureSet((*physical, "months_since_1990", "lagged_rpi"), categories)
    contemporaneous = FeatureSet((*physical, "transaction_year", "transaction_month", "contemporaneous_rpi", "nearest_mrt_distance_m", "mrt_distance_missing"), categories)

    def hist(features: FeatureSet) -> object:
        return make_hist_pipeline(features, max_iter=180, min_samples_leaf=40)

    return [
        Experiment("A_current_baseline", current, lambda: hist(current), notes="Milestone 4 reference"),
        Experiment("B_lagged_rpi", lag_mrt, lambda: hist(lag_mrt), requires_lagged_rpi=True),
        Experiment("C_lagged_rpi_no_mrt", lag_no_mrt, lambda: hist(lag_no_mrt), requires_lagged_rpi=True),
        Experiment("D_month_index_lagged_rpi", month_lag_mrt, lambda: hist(month_lag_mrt), requires_lagged_rpi=True),
        Experiment(
            "E_rpi_scaled_target_with_mrt", month_lag_mrt,
            lambda: IndexedTargetRegressor(hist(month_lag_mrt), "lagged_rpi"),
            notes="Target modeled in constant lagged-RPI units", requires_lagged_rpi=True,
        ),
        Experiment(
            "F_rpi_scaled_target_no_mrt", month_lag_no_mrt,
            lambda: IndexedTargetRegressor(hist(month_lag_no_mrt), "lagged_rpi"),
            notes="Target modeled in constant lagged-RPI units", requires_lagged_rpi=True,
        ),
        Experiment(
            "diagnostic_contemporaneous_rpi", contemporaneous, lambda: hist(contemporaneous),
            selectable=False, notes="Historical diagnostic only; prohibited for production",
        ),
    ]


def ec_experiments() -> list[Experiment]:
    numeric_base = ("area_sqm", "number_of_units", "floor_midpoint", "lease_duration_years", "property_age_at_transaction")
    categories = ("project_name", "sale_type", "area_type", "property_type", "tenure_category", "postal_district", "market_segment")
    current = FeatureSet((*numeric_base, "transaction_year", "transaction_month"), categories)
    time_index = FeatureSet((*numeric_base, "months_since_1990"), categories)
    return [
        Experiment("A_current_ridge", current, lambda: make_linear_pipeline(current)),
        Experiment("B_time_index_ridge", time_index, lambda: make_linear_pipeline(time_index)),
        Experiment(
            "C_time_index_ridge_rare_project", time_index,
            lambda: make_linear_pipeline(time_index, rare_columns=("project_name",), rare_min_count=20),
            notes="Project grouping fitted from training counts only",
        ),
        Experiment(
            "D_time_index_hist_gradient_boosting", time_index,
            lambda: make_hist_pipeline(time_index, max_iter=220, min_samples_leaf=20),
        ),
    ]


def landed_experiments() -> list[Experiment]:
    numeric_base = ("area_sqm", "number_of_units", "lease_duration_years", "property_age_at_transaction", "is_multi_unit_transaction")
    categories = ("sale_type", "area_type", "property_type", "tenure_category", "postal_district", "market_segment")
    current = FeatureSet((*numeric_base, "transaction_year", "transaction_month"), categories)
    time_index = FeatureSet((*numeric_base, "months_since_1990"), categories)
    return [
        Experiment("A_current_hist_gradient_boosting", current, lambda: make_hist_pipeline(current, max_iter=250, min_samples_leaf=20)),
        Experiment("B_time_index_hist_gradient_boosting", time_index, lambda: make_hist_pipeline(time_index, max_iter=250, min_samples_leaf=20)),
        Experiment(
            "C_time_index_log_target", time_index,
            lambda: TransformedTargetRegressor(
                regressor=make_hist_pipeline(time_index, max_iter=250, min_samples_leaf=20),
                func=np.log1p,
                inverse_func=np.expm1,
                check_inverse=False,
            ),
            notes="Evaluated in original SGD after inverse transformation",
        ),
    ]


def category_robustness(category: str, train: pd.DataFrame, validation: pd.DataFrame, columns: tuple[str, ...], rare_threshold: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column in columns:
        counts = train[column].astype("string").value_counts()
        validation_values = validation[column].astype("string")
        unseen = ~validation_values.isin(counts.index)
        rare_values = set(counts.loc[counts < rare_threshold].index)
        rows.append({
            "category": category,
            "column": column,
            "rare_threshold": rare_threshold,
            "train_unique": len(counts),
            "validation_unique": validation_values.nunique(),
            "unseen_validation_categories": validation_values.loc[unseen].nunique(),
            "unseen_validation_rows": int(unseen.sum()),
            "unseen_validation_percent": float(100 * unseen.mean()),
            "rare_training_categories": len(rare_values),
            "validation_rows_in_rare_training_categories": int(validation_values.isin(rare_values).sum()),
        })
    return rows


def hdb_drift_tables(train: pd.DataFrame, validation: pd.DataFrame, train_prediction: np.ndarray, validation_prediction: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces: list[pd.DataFrame] = []
    for split, frame, prediction in (
        ("train", train, train_prediction), ("validation", validation, validation_prediction)
    ):
        work = frame[["transaction_year", "transaction_quarter", "resale_price", "town", "flat_type"]].copy()
        work["prediction"] = prediction
        work["residual"] = work["prediction"] - work["resale_price"]
        work["absolute_error"] = work["residual"].abs()
        for level, keys in (("year", ["transaction_year"]), ("quarter", ["transaction_year", "transaction_quarter"])):
            grouped = work.groupby(keys, observed=True).agg(
                rows=("resale_price", "size"),
                actual_mean=("resale_price", "mean"),
                actual_median=("resale_price", "median"),
                predicted_mean=("prediction", "mean"),
                predicted_median=("prediction", "median"),
                mean_residual_prediction_minus_actual=("residual", "mean"),
                mae=("absolute_error", "mean"),
            ).reset_index()
            grouped.insert(0, "period_level", level)
            grouped.insert(0, "split", split)
            pieces.append(grouped)
    period_table = pd.concat(pieces, ignore_index=True)

    validation_work = validation[["resale_price", "town", "flat_type"]].copy()
    validation_work["prediction"] = validation_prediction
    validation_work["residual"] = validation_prediction - validation_work["resale_price"]
    validation_work["absolute_error"] = validation_work["residual"].abs()
    group_pieces: list[pd.DataFrame] = []
    total_absolute_error = validation_work["absolute_error"].sum()
    for dimension in ("town", "flat_type"):
        grouped = validation_work.groupby(dimension, observed=True).agg(
            rows=("resale_price", "size"),
            actual_mean=("resale_price", "mean"),
            predicted_mean=("prediction", "mean"),
            mean_residual_prediction_minus_actual=("residual", "mean"),
            mae=("absolute_error", "mean"),
            absolute_error_total=("absolute_error", "sum"),
        ).reset_index().rename(columns={dimension: "group"})
        grouped["row_share_percent"] = 100 * grouped["rows"] / len(validation_work)
        grouped["absolute_error_share_percent"] = 100 * grouped["absolute_error_total"] / total_absolute_error
        grouped.insert(0, "dimension", dimension)
        group_pieces.append(grouped)
    return period_table, pd.concat(group_pieces, ignore_index=True)


def calibration(category: str, actual: pd.Series, prediction: np.ndarray) -> dict[str, object]:
    errors = np.abs(actual.to_numpy(dtype=float) - prediction)
    bins, labels = PRICE_BANDS[category]
    bands = pd.cut(prediction, bins=bins, labels=labels, include_lowest=True)
    band_rows: dict[str, object] = {}
    for label in labels:
        mask = np.asarray(bands == label)
        if not mask.any():
            continue
        values = errors[mask]
        band_rows[label] = {
            "rows": int(mask.sum()),
            "p50": float(np.quantile(values, 0.50)),
            "p80": float(np.quantile(values, 0.80)),
            "p90": float(np.quantile(values, 0.90)),
            "p95": float(np.quantile(values, 0.95)),
        }
    global_values = absolute_error_percentiles(actual, prediction)
    return {
        "strategy": "Select a broad band using the model prediction, then apply that band's historical held-out absolute-error percentile; fall back to global percentiles if a band is unavailable.",
        "interpretation": "Historical prediction error range, not a formal confidence interval.",
        "band_edges_sgd": [None if not np.isfinite(value) else value for value in bins],
        "band_labels": labels,
        "global": global_values,
        "by_predicted_price_band": band_rows,
    }


def field(name: str, data_type: str, required: bool, **extra: object) -> dict[str, object]:
    return {"name": name, "type": data_type, "required": required, **extra}


def build_inference_contract(
    selected: dict[str, Experiment],
    combined_training: dict[str, pd.DataFrame],
    calibration_data: dict[str, object],
) -> dict[str, object]:
    categories: dict[str, object] = {}
    for category in ("hdb", "ec", "landed"):
        experiment = selected[category]
        frame = combined_training[category]
        categorical_values = {
            column: sorted(frame[column].dropna().astype(str).unique().tolist())
            for column in experiment.features.categorical
        }
        common = {
            "model_artifact": f"models/{category}_model.joblib",
            "model_features_in_order": experiment.features.columns,
            "prediction_output": {"name": "predicted_price_sgd", "type": "float", "minimum": 0},
            "unknown_category_behavior": (
                "Accepted without crashing. Ridge ignores unseen one-hot categories; "
                "HistGradientBoosting maps unseen ordinal categories to missing/unknown."
            ),
            "numeric_missing_behavior": "Training-fitted median imputation unless a field is explicitly required.",
            "historical_error_calibration": calibration_data[category],
        }
        if category == "hdb":
            inputs = [
                field("valuation_month", "YYYY-MM", True, derivation="Supplies calendar time and RPI lookup quarter."),
                field("town", "string", True, valid_values=categorical_values["town"]),
                field("flat_type", "string", True, valid_values=categorical_values["flat_type"]),
                field("flat_model", "string", True, valid_values=categorical_values["flat_model"]),
                field("floor_area_sqm", "float", True, minimum=1),
                field("storey_midpoint", "float", True, derivation="Midpoint of selected HDB storey range."),
                field("lease_commence_year", "integer", True),
            ]
            if "nearest_mrt_distance_m" in experiment.features.columns:
                inputs.append(field(
                    "nearest_mrt_distance_m", "float", False, minimum=0,
                    missing_behavior="Leave null; derive mrt_distance_missing=true. Never substitute zero.",
                ))
            common.update({
                "inputs": inputs,
                "derived_fields": {
                    "property_age_at_transaction": "valuation year - lease_commence_year",
                    "months_since_1990": "(valuation year - 1990) * 12 + valuation month - 1",
                    "mrt_distance_missing": "nearest_mrt_distance_m is null",
                    "lagged_rpi": "Latest official RPI whose quarter is strictly before the valuation quarter.",
                },
                "rpi_policy": {
                    "source": "data/processed/hdb_rpi.csv",
                    "same_quarter_prohibited": True,
                    "future_behavior": (
                        "Use the latest locally available official quarter strictly before the valuation quarter; "
                        "if the immediately prior quarter is unavailable, keep the latest older value and disclose its source quarter. Never extrapolate or fabricate RPI."
                    ),
                    "latest_available_quarter": "2026-Q2",
                },
            })
        else:
            inputs = [
                field("valuation_month", "YYYY-MM", True),
                field("area_sqm", "float", True, minimum=1),
                field(
                    "number_of_units", "integer", True, minimum=1,
                    maximum=1 if category == "landed" else None,
                    validation=(
                        "Must equal 1; multi-unit/portfolio transactions are outside model scope."
                        if category == "landed" else "Positive integer."
                    ),
                ),
                field("property_age_at_transaction", "float", False, missing_behavior="Median-imputed; normally derive from lease commencement year."),
                field("lease_duration_years", "float", False, missing_behavior="May be null for freehold."),
            ]
            if "floor_midpoint" in experiment.features.columns:
                inputs.append(field("floor_midpoint", "float", False, missing_behavior="Training median when unavailable."))
            for column in experiment.features.categorical:
                inputs.append(field(column, "string", True, valid_values=categorical_values[column]))
            common.update({
                "inputs": inputs,
                "derived_fields": {
                    "months_since_1990": "(valuation year - 1990) * 12 + valuation month - 1",
                    "is_multi_unit_transaction": "number_of_units > 1",
                },
                "exclusions": ["MRT distance"] if category == "landed" else ["MRT distance (validation showed deterioration)"],
            })
        categories[category] = common
    return {"version": 1, "generated_for": "Milestone 5 production freeze", "categories": categories}


def metadata_for(
    category: str,
    experiment: Experiment,
    final_metric: dict[str, object],
    error_data: dict[str, object],
    fit_rows: int,
) -> dict[str, object]:
    return {
        "status": "production_frozen",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "category": category,
        "target": "resale_price" if category == "hdb" else "transaction_price",
        "experiment": experiment.name,
        "model_type": experiment.factory().__class__.__name__,
        "features_in_order": experiment.features.columns,
        "numeric_features": list(experiment.features.numeric),
        "categorical_features": list(experiment.features.categorical),
        "preprocessing": "Training-fitted numeric median imputation; training-fitted categorical encoding; category-specific estimator.",
        "fit_rows": fit_rows,
        "training_period": "1990-01 through 2024-12" if category == "hdb" else "2021-01 through 2024-12",
        "validation_selection_period": "2023-01 through 2024-12" if category == "hdb" else "2024-01 through 2024-12",
        "test_period": "2025-01 through 2026-02" if category == "hdb" else "2025-01 through 2026-09",
        "final_test_metrics": final_metric,
        "empirical_error_calibration": error_data,
        "mrt_usage": "nearest_mrt_distance_m" in experiment.features.columns,
        "rpi_usage": "lagged_rpi" in experiment.features.columns,
        "inference_requirements": "See models/inference_contract.json",
        "exclusions_and_limitations": [
            "Not valuation advice", "No private-condo or rental prediction", "No coordinates as predictors",
            "Landed excludes multi-unit transactions" if category == "landed" else "Category coverage follows supplied data",
        ],
        "versions": {
            "python": platform.python_version(), "sklearn": sklearn.__version__,
            "pandas": pd.__version__, "numpy": np.__version__,
        },
        "random_seed": RANDOM_SEED,
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rpi = load_rpi()
    refinement_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    robustness_rows: list[dict[str, object]] = []
    selected: dict[str, Experiment] = {}
    frozen_models: dict[str, object] = {}
    combined_training: dict[str, pd.DataFrame] = {}
    final_predictions: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    fit_rows: dict[str, int] = {}

    # HDB: test data is not touched until validation selection has completed.
    hdb = pd.read_csv(PROCESSED / "hdb_enriched.csv", low_memory=False)
    hdb = add_hdb_rpi_features(hdb, rpi)
    hdb_train = hdb.loc[hdb["data_split"].eq("train")]
    hdb_validation = hdb.loc[hdb["data_split"].eq("validation")]
    hdb_exps = hdb_experiments()
    hdb_models: dict[str, object] = {}
    hdb_predictions: dict[str, np.ndarray] = {}
    for experiment in hdb_exps:
        model, prediction, seconds, rows = fit_predict(experiment, hdb_train, hdb_validation, "resale_price")
        hdb_models[experiment.name] = model
        hdb_predictions[experiment.name] = prediction
        refinement_rows.append(metric_row(
            "hdb", experiment.name, "validation", hdb_validation["resale_price"], prediction,
            selectable=experiment.selectable, training_rows=rows, fit_seconds=seconds, notes=experiment.notes,
        ))
    hdb_selectable = [row for row in refinement_rows if row["category"] == "hdb" and row["selectable"]]
    hdb_winner_name = str(min(hdb_selectable, key=lambda row: float(row["mae"]))["experiment"])
    selected["hdb"] = next(experiment for experiment in hdb_exps if experiment.name == hdb_winner_name)
    baseline_model = hdb_models["A_current_baseline"]
    train_baseline_prediction = np.asarray(
        baseline_model.predict(hdb_train[next(exp for exp in hdb_exps if exp.name == "A_current_baseline").features.columns])
    )
    drift_periods, drift_groups = hdb_drift_tables(
        hdb_train, hdb_validation, train_baseline_prediction, hdb_predictions["A_current_baseline"]
    )
    drift_periods.to_csv(REPORT_DIR / "hdb_temporal_drift.csv", index=False)
    drift_groups.to_csv(REPORT_DIR / "hdb_drift_groups.csv", index=False)
    robustness_rows.extend(category_robustness("hdb", hdb_train, hdb_validation, ("town", "flat_type", "flat_model"), 100))

    # EC: one time representation, one training-only rare-project policy, one nonlinear comparator.
    ec = add_month_index(pd.read_csv(PROCESSED / "ec_enriched.csv", low_memory=False))
    ec_train = ec.loc[ec["data_split"].eq("train")]
    ec_validation = ec.loc[ec["data_split"].eq("validation")]
    ec_exps = ec_experiments()
    for experiment in ec_exps:
        _, prediction, seconds, rows = fit_predict(experiment, ec_train, ec_validation, "transaction_price")
        refinement_rows.append(metric_row(
            "ec", experiment.name, "validation", ec_validation["transaction_price"], prediction,
            selectable=True, training_rows=rows, fit_seconds=seconds, notes=experiment.notes,
        ))
    ec_rows = [row for row in refinement_rows if row["category"] == "ec"]
    ec_winner_name = str(min(ec_rows, key=lambda row: float(row["mae"]))["experiment"])
    selected["ec"] = next(experiment for experiment in ec_exps if experiment.name == ec_winner_name)
    robustness_rows.extend(category_robustness(
        "ec", ec_train, ec_validation,
        ("project_name", "sale_type", "property_type", "postal_district", "market_segment"), 20,
    ))

    # Landed: all selection/evaluation rows are single-unit; luxury transactions remain.
    landed = add_month_index(pd.read_csv(PROCESSED / "landed_enriched.csv", low_memory=False))
    landed_train = landed.loc[landed["data_split"].eq("train") & ~landed["is_multi_unit_transaction"].astype(bool)]
    landed_validation = landed.loc[landed["data_split"].eq("validation") & ~landed["is_multi_unit_transaction"].astype(bool)]
    landed_exps = landed_experiments()
    for experiment in landed_exps:
        _, prediction, seconds, rows = fit_predict(experiment, landed_train, landed_validation, "transaction_price")
        refinement_rows.append(metric_row(
            "landed", experiment.name, "validation", landed_validation["transaction_price"], prediction,
            selectable=True, training_rows=rows, fit_seconds=seconds, notes=experiment.notes,
        ))
    landed_rows = [row for row in refinement_rows if row["category"] == "landed"]
    landed_winner_name = str(min(landed_rows, key=lambda row: float(row["mae"]))["experiment"])
    selected["landed"] = next(experiment for experiment in landed_exps if experiment.name == landed_winner_name)
    robustness_rows.extend(category_robustness(
        "landed", landed_train, landed_validation, ("property_type", "postal_district", "market_segment"), 20,
    ))

    # Freeze choices, refit on train+validation, then access TEST exactly once per category.
    category_frames = {"hdb": hdb, "ec": ec, "landed": landed}
    targets = {"hdb": "resale_price", "ec": "transaction_price", "landed": "transaction_price"}
    for category in ("hdb", "ec", "landed"):
        frame = category_frames[category]
        experiment = selected[category]
        fit_mask = frame["data_split"].isin(["train", "validation"])
        test_mask = frame["data_split"].eq("test")
        if category == "landed":
            individual = ~frame["is_multi_unit_transaction"].astype(bool)
            fit_mask &= individual
            test_mask &= individual
        if experiment.requires_lagged_rpi:
            fit_mask &= frame["lagged_rpi"].notna()
        fitting = frame.loc[fit_mask]
        test = frame.loc[test_mask]
        model = experiment.factory()
        model.fit(fitting[experiment.features.columns], fitting[targets[category]])
        prediction = np.asarray(model.predict(test[experiment.features.columns]), dtype=float)
        if not np.isfinite(prediction).all():
            raise ValueError(f"{category}: non-finite final predictions")
        frozen_models[category] = model
        combined_training[category] = fitting
        final_predictions[category] = (test, prediction)
        fit_rows[category] = len(fitting)
        final_rows.append(metric_row(
            category, experiment.name, "test_once_after_freeze", test[targets[category]], prediction,
            training_rows=len(fitting),
            mae_change_vs_milestone4=float(regression_metrics(test[targets[category]], prediction)["mae"]) - ORIGINAL_TEST[category]["mae"],
            rmse_change_vs_milestone4=float(regression_metrics(test[targets[category]], prediction)["rmse"]) - ORIGINAL_TEST[category]["rmse"],
            r2_change_vs_milestone4=float(regression_metrics(test[targets[category]], prediction)["r2"]) - ORIGINAL_TEST[category]["r2"],
        ))
        joblib.dump(model, MODEL_DIR / f"{category}_model.joblib", compress=3)

    calibration_data = {
        category: calibration(category, test[targets[category]], prediction)
        for category, (test, prediction) in final_predictions.items()
    }
    (MODEL_DIR / "error_calibration.json").write_text(
        json.dumps(calibration_data, indent=2) + "\n", encoding="utf-8"
    )
    contract = build_inference_contract(selected, combined_training, calibration_data)
    (MODEL_DIR / "inference_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    for category in ("hdb", "ec", "landed"):
        final_metric = next(row for row in final_rows if row["category"] == category)
        metadata = metadata_for(category, selected[category], final_metric, calibration_data[category], fit_rows[category])
        (MODEL_DIR / f"{category}_model_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    pd.DataFrame(refinement_rows).to_csv(REPORT_DIR / "refinement_results.csv", index=False)
    pd.DataFrame(final_rows).to_csv(REPORT_DIR / "final_model_results.csv", index=False)
    pd.DataFrame(robustness_rows).to_csv(REPORT_DIR / "category_robustness.csv", index=False)

    # Machine-readable audit of deterministic future-date RPI behavior.
    rpi_policy_examples = {
        month: rpi_for_valuation_month(month, rpi)
        for month in ("2026-03-01", "2026-07-01", "2027-01-01")
    }
    (REPORT_DIR / "rpi_policy_examples.json").write_text(
        json.dumps(rpi_policy_examples, indent=2) + "\n", encoding="utf-8"
    )
    write_summary(
        pd.DataFrame(refinement_rows), pd.DataFrame(final_rows), drift_periods,
        drift_groups, pd.DataFrame(robustness_rows), selected, calibration_data,
        rpi_policy_examples,
    )
    print("Frozen production models and refinement reports written.")


def dollars(value: float) -> str:
    return f"${value:,.0f}"


def write_summary(
    refinement: pd.DataFrame,
    final: pd.DataFrame,
    drift: pd.DataFrame,
    drift_groups: pd.DataFrame,
    robustness: pd.DataFrame,
    selected: dict[str, Experiment],
    calibration_data: dict[str, object],
    rpi_examples: dict[str, object],
) -> None:
    hdb_year = drift.loc[(drift["split"] == "validation") & (drift["period_level"] == "year")]
    hdb_2023 = hdb_year.loc[hdb_year["transaction_year"] == 2023].iloc[0]
    hdb_2024 = hdb_year.loc[hdb_year["transaction_year"] == 2024].iloc[0]
    town = drift_groups.loc[drift_groups["dimension"].eq("town")].sort_values("absolute_error_share_percent", ascending=False).iloc[0]
    flat = drift_groups.loc[drift_groups["dimension"].eq("flat_type")].sort_values("absolute_error_share_percent", ascending=False).iloc[0]
    lines = [
        "# Model refinement and production freeze", "",
        "Configuration selection used TRAIN fitting and VALIDATION metrics only. Existing TEST results were not used for selection. After each configuration was fixed, it was refit on TRAIN+VALIDATION and evaluated once on TEST; no post-test changes were made.", "",
        "## HDB drift diagnosis", "",
        f"- The baseline mean prediction bias was {dollars(hdb_2023['mean_residual_prediction_minus_actual'])} in 2023 and {dollars(hdb_2024['mean_residual_prediction_minus_actual'])} in 2024 (prediction minus actual).",
        f"- Actual median rose from {dollars(hdb_2023['actual_median'])} in 2023 to {dollars(hdb_2024['actual_median'])} in 2024, while predicted median moved from {dollars(hdb_2023['predicted_median'])} to {dollars(hdb_2024['predicted_median'])}. This is predominantly a market-level upward shift that a bounded tree time split cannot extrapolate.",
        f"- {town['group']} contributed the largest town share of validation absolute error ({town['absolute_error_share_percent']:.1f}%); {flat['group']} contributed the largest flat-type share ({flat['absolute_error_share_percent']:.1f}%). Full year/quarter and group tables are provided separately.", "",
        "## Validation experiments", "",
        "| Category | Experiment | Selectable | MAE | RMSE | R² | Bias |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in refinement.itertuples():
        lines.append(
            f"| {row.category.upper()} | {row.experiment} | {row.selectable} | {dollars(row.mae)} | "
            f"{dollars(row.rmse)} | {row.r2:.3f} | {dollars(row.mean_residual_prediction_minus_actual)} |"
        )
    lines += ["", "## Final frozen models and one-time test", ""]
    for category in ("hdb", "ec", "landed"):
        row = final.loc[final["category"].eq(category)].iloc[0]
        exp = selected[category]
        lines += [
            f"### {category.upper()}", "",
            f"- Frozen configuration: **{exp.name}**.",
            f"- Test: MAE {dollars(row['mae'])}, RMSE {dollars(row['rmse'])}, R² {row['r2']:.3f}, "
            f"MAE {row['mae_percent_of_median']:.2f}% of median, bias {dollars(row['mean_residual_prediction_minus_actual'])}.",
            f"- Versus Milestone 4: MAE {dollars(row['mae_change_vs_milestone4'])}, "
            f"RMSE {dollars(row['rmse_change_vs_milestone4'])}, R² {row['r2_change_vs_milestone4']:+.3f}.", "",
        ]
    hdb_final = selected["hdb"]
    lines += [
        "## RPI and MRT policy", "",
        "- `hdb_rpi.csv` contains 146 validated official quarters from 1990-Q1 through 2026-Q2. Contemporaneous RPI was diagnostic-only and could not be selected.",
        f"- Production HDB uses lagged RPI: **{'yes' if 'lagged_rpi' in hdb_final.features.columns else 'no'}**. It uses the latest official quarter strictly before the valuation quarter.",
        "- For a future month beyond available data, the app uses the latest older official RPI and displays its source quarter; it never extrapolates or fabricates RPI.",
        f"- Deterministic examples: `{json.dumps(rpi_examples, sort_keys=True)}`.",
        f"- Final HDB MRT usage: **{'retained' if 'nearest_mrt_distance_m' in hdb_final.features.columns else 'excluded'}**, selected using the matched validation configurations after temporal/RPI refinement.", "",
        "## Rare/unseen categories", "",
    ]
    for row in robustness.itertuples():
        if row.unseen_validation_rows or row.validation_rows_in_rare_training_categories:
            lines.append(
                f"- {row.category.upper()} `{row.column}`: {row.unseen_validation_rows:,} unseen validation rows; "
                f"{row.validation_rows_in_rare_training_categories:,} rows in train-defined rare categories."
            )
    lines += [
        "- All frozen preprocessors handle unseen categories without crashing. The EC rare-project experiment groups categories using TRAIN frequencies only; whether it was selected is shown above.", "",
        "## Historical prediction error ranges", "",
        "Ranges use absolute errors from the single final chronological test evaluation. The app should select a fixed broad band using the model prediction, then use that band's P80/P90/P95 value; fall back to the global value when needed. These are historical prediction error ranges, not confidence intervals.", "",
    ]
    for category in ("hdb", "ec", "landed"):
        global_errors = calibration_data[category]["global"]
        lines.append(
            f"- **{category.upper()} global:** P50 {dollars(global_errors['median_absolute_error'])}, "
            f"P80 {dollars(global_errors['p80_absolute_error'])}, P90 {dollars(global_errors['p90_absolute_error'])}, "
            f"P95 {dollars(global_errors['p95_absolute_error'])}. Predicted-price bands are stored in `models/error_calibration.json`."
        )
    lines += [
        "", "## Limitations", "",
        "- RPI availability is quarter-level and publication timing is represented conservatively by requiring a strictly prior quarter; the local file must be deliberately updated to use newer official releases.",
        "- HDB remains exposed to structural shifts not captured by RPI. EC unseen projects rely on non-project features or train-derived rare handling. Landed remains heterogeneous and coarse-location constrained.",
        "- Multi-unit landed sales are excluded; legitimate luxury single-property sales remain. Models are demonstration tools, not professional valuations.",
    ]
    (REPORT_DIR / "refinement_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
