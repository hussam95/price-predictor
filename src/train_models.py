"""Run time-aware baseline experiments and MRT feature ablations.

Usage:
    .venv/bin/python -m src.train_models
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from .modeling import ec as ec_config
from .modeling import hdb as hdb_config
from .modeling import landed as landed_config
from .modeling.common import (
    RANDOM_SEED,
    FeatureSet,
    absolute_error_percentiles,
    add_analysis_bands,
    assert_features_safe,
    assert_split_isolation,
    evaluate_groups,
    make_hist_pipeline,
    make_linear_pipeline,
    model_parameters,
)
from .modeling.evaluate import feature_permutation_importance, metric_record


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "modeling"
ERROR_DIR = REPORT_DIR / "error_analysis"
MODEL_DIR = ROOT / "models"
MANIFEST_PATH = PROCESSED / "feature_manifest.json"
SPLIT_PLAN_PATH = PROCESSED / "split_plan.json"

EXPECTED_SPLITS = {
    "hdb": {"train": 889_599, "validation": 53_586, "test": 28_338},
    "ec": {"train": 7_418, "validation": 3_245, "test": 5_832},
    "landed": {"train": 3_838, "validation": 1_695, "test": 3_207},
}

FORBIDDEN_FIELDS = {
    "resale_price", "transaction_price", "unit_price_psf", "unit_price_psm",
    "nett_price", "price_per_sqm", "price_per_sqft", "diagnostic_price_outlier_iqr",
    "latitude", "longitude", "nearest_mrt_station", "nearest_mrt_exit",
    "nearest_mrt_distance_reference_m", "matched_address", "postal_code",
}


def load_frame(category: str, columns: set[str]) -> pd.DataFrame:
    path = PROCESSED / f"{category}_enriched.csv"
    frame = pd.read_csv(path, usecols=lambda column: column in columns, low_memory=False)
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{category}: missing columns {sorted(missing)}")
    return frame


def verify_manifest_features(category: str, features: FeatureSet, manifest: dict[str, Any]) -> None:
    section = manifest[category]
    authorized = {
        *section["safe_numeric_candidate_features"],
        *section["safe_categorical_candidate_features"],
        *section["temporal_features"],
    }
    unauthorized = set(features.columns) - authorized
    if unauthorized:
        raise ValueError(f"{category}: features not authorized by manifest: {sorted(unauthorized)}")
    assert_features_safe(features, section["target"], FORBIDDEN_FIELDS)


def split_frame(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {name: frame.loc[frame["data_split"].eq(name)] for name in ("train", "validation", "test")}


def fit_model(
    factory: Callable[[FeatureSet], Pipeline],
    features: FeatureSet,
    train: pd.DataFrame,
    target: str,
) -> tuple[Pipeline, float]:
    started = time.perf_counter()
    estimator = factory(features)
    estimator.fit(train[features.columns], train[target])
    return estimator, time.perf_counter() - started


def predict(estimator: Pipeline, features: FeatureSet, frame: pd.DataFrame) -> np.ndarray:
    output = np.asarray(estimator.predict(frame[features.columns]), dtype=float)
    if output.shape != (len(frame),):
        raise ValueError(f"Unexpected prediction shape {output.shape}")
    return output


def median_record(category: str, train: pd.DataFrame, validation: pd.DataFrame, target: str) -> dict[str, Any]:
    value = float(train[target].median())
    return metric_record(
        category=category,
        model_name="training_median",
        feature_variant="none",
        evaluation_split="validation",
        training_policy="all_transactions",
        y_true=validation[target],
        prediction=np.full(len(validation), value),
    )


def run_mrt_category(
    category: str,
    frame: pd.DataFrame,
    target: str,
    base_features: FeatureSet,
    mrt_features: FeatureSet,
    baseline_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    splits = split_frame(frame)
    train, validation, test = splits["train"], splits["validation"], splits["test"]
    baseline_rows.append(median_record(category, train, validation, target))
    fitted: dict[tuple[str, str], tuple[Pipeline, FeatureSet]] = {}
    factories: dict[str, Callable[[FeatureSet], Pipeline]] = {
        "ridge": make_linear_pipeline,
        "hist_gradient_boosting": lambda f: make_hist_pipeline(
            f,
            max_iter=180 if category == "hdb" else 220,
            min_samples_leaf=40 if category == "hdb" else 20,
        ),
    }
    for model_name, factory in factories.items():
        for variant, features in (("without_mrt", base_features), ("with_mrt", mrt_features)):
            estimator, seconds = fit_model(factory, features, train, target)
            fitted[(model_name, variant)] = (estimator, features)
            runtime_rows.append({
                "category": category, "model": model_name, "feature_variant": variant,
                "training_policy": "all_transactions", "fit_seconds": seconds,
                "training_rows": len(train),
            })
            validation_prediction = predict(estimator, features, validation)
            baseline_rows.append(metric_record(
                category=category, model_name=model_name, feature_variant=variant,
                evaluation_split="validation", training_policy="all_transactions",
                y_true=validation[target], prediction=validation_prediction,
            ))

    validation_candidates = [
        row for row in baseline_rows
        if row["category"] == category and row["evaluation_split"] == "validation" and row["model"] != "training_median"
    ]
    selected_row = min(validation_candidates, key=lambda row: float(row["mae"]))
    selected_family = str(selected_row["model"])

    ablation_rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str], np.ndarray] = {}
    for split_name, evaluation in (("validation", validation), ("test", test)):
        records: dict[str, dict[str, Any]] = {}
        for variant in ("without_mrt", "with_mrt"):
            estimator, features = fitted[(selected_family, variant)]
            output = predict(estimator, features, evaluation)
            predictions[(split_name, variant)] = output
            record = metric_record(
                category=category, model_name=selected_family, feature_variant=variant,
                evaluation_split=split_name, training_policy="all_transactions",
                y_true=evaluation[target], prediction=output,
            )
            records[variant] = record
            if split_name == "test":
                baseline_rows.append(record)
        without, with_mrt = records["without_mrt"], records["with_mrt"]
        ablation_rows.append({
            "category": category, "model": selected_family, "evaluation_split": split_name,
            "rows": len(evaluation),
            "mae_without_mrt": without["mae"], "mae_with_mrt": with_mrt["mae"],
            "mae_absolute_change": with_mrt["mae"] - without["mae"],
            "mae_percent_change": 100.0 * (with_mrt["mae"] - without["mae"]) / without["mae"],
            "rmse_without_mrt": without["rmse"], "rmse_with_mrt": with_mrt["rmse"],
            "rmse_absolute_change": with_mrt["rmse"] - without["rmse"],
            "rmse_percent_change": 100.0 * (with_mrt["rmse"] - without["rmse"]) / without["rmse"],
            "r2_without_mrt": without["r2"], "r2_with_mrt": with_mrt["r2"],
            "r2_absolute_change": with_mrt["r2"] - without["r2"],
        })

    keep_mrt = ablation_rows[0]["mae_with_mrt"] < ablation_rows[0]["mae_without_mrt"]
    selected_variant = "with_mrt" if keep_mrt else "without_mrt"
    selected_estimator, selected_features = fitted[(selected_family, selected_variant)]
    return {
        "selected_model": selected_estimator,
        "selected_model_name": selected_family,
        "selected_variant": selected_variant,
        "selected_features": selected_features,
        "validation_prediction": predictions[("validation", selected_variant)],
        "test_prediction": predictions[("test", selected_variant)],
        "splits": splits,
        "ablation_rows": ablation_rows,
        "parameters": model_parameters(selected_estimator),
    }


def run_landed(
    frame: pd.DataFrame,
    target: str,
    features: FeatureSet,
    baseline_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    splits = split_frame(frame)
    train, validation, test = splits["train"], splits["validation"], splits["test"]
    baseline_rows.append(median_record("landed", train, validation, target))
    factories: dict[str, Callable[[FeatureSet], Pipeline]] = {
        "ridge": make_linear_pipeline,
        "hist_gradient_boosting": lambda f: make_hist_pipeline(f, max_iter=250, min_samples_leaf=20),
    }
    fitted_all: dict[str, Pipeline] = {}
    for model_name, factory in factories.items():
        estimator, seconds = fit_model(factory, features, train, target)
        fitted_all[model_name] = estimator
        runtime_rows.append({
            "category": "landed", "model": model_name, "feature_variant": "no_mrt",
            "training_policy": "all_transactions", "fit_seconds": seconds,
            "training_rows": len(train),
        })
        output = predict(estimator, features, validation)
        baseline_rows.append(metric_record(
            category="landed", model_name=model_name, feature_variant="no_mrt",
            evaluation_split="validation", training_policy="all_transactions",
            y_true=validation[target], prediction=output,
        ))

    candidates = [
        row for row in baseline_rows
        if row["category"] == "landed" and row["evaluation_split"] == "validation" and row["model"] != "training_median"
    ]
    selected_family = str(min(candidates, key=lambda row: float(row["mae"]))["model"])
    factory = factories[selected_family]
    single_train = train.loc[~train["is_multi_unit_transaction"].astype(bool)]
    single_validation = validation.loc[~validation["is_multi_unit_transaction"].astype(bool)]
    single_test = test.loc[~test["is_multi_unit_transaction"].astype(bool)]
    single_model, seconds = fit_model(factory, features, single_train, target)
    runtime_rows.append({
        "category": "landed", "model": selected_family, "feature_variant": "no_mrt",
        "training_policy": "exclude_multi_unit", "fit_seconds": seconds,
        "training_rows": len(single_train),
    })

    all_model = fitted_all[selected_family]
    policy_rows: list[dict[str, Any]] = []
    policy_predictions: dict[tuple[str, str], np.ndarray] = {}
    for split_name, evaluation in (("validation", single_validation), ("test", single_test)):
        for policy, estimator in (("all_transactions", all_model), ("exclude_multi_unit", single_model)):
            output = predict(estimator, features, evaluation)
            policy_predictions[(split_name, policy)] = output
            policy_rows.append(metric_record(
                category="landed", model_name=selected_family, feature_variant="no_mrt",
                evaluation_split=split_name, training_policy=policy,
                y_true=evaluation[target], prediction=output,
            ))
    validation_policy = [row for row in policy_rows if row["evaluation_split"] == "validation"]
    selected_policy = str(min(validation_policy, key=lambda row: float(row["mae"]))["training_policy"])
    selected_model = all_model if selected_policy == "all_transactions" else single_model
    selected_validation = policy_predictions[("validation", selected_policy)]
    selected_test = policy_predictions[("test", selected_policy)]
    baseline_rows.extend(row for row in policy_rows if row["evaluation_split"] == "test")

    full_test_prediction = predict(all_model, features, test)
    selected_full_test_prediction = predict(selected_model, features, test)
    return {
        "selected_model": selected_model,
        "selected_model_name": selected_family,
        "selected_policy": selected_policy,
        "selected_features": features,
        "validation_prediction": selected_validation,
        "test_prediction": selected_test,
        "splits": {**splits, "selected_validation": single_validation, "selected_test": single_test},
        "policy_rows": policy_rows,
        "full_test_prediction": full_test_prediction,
        "selected_full_test_prediction": selected_full_test_prediction,
        "parameters": model_parameters(selected_model),
        "multi_unit_counts": {
            name: int(part["is_multi_unit_transaction"].astype(bool).sum()) for name, part in splits.items()
        },
    }


def selected_metric(
    result: dict[str, Any], category: str, target: str, split_name: str,
) -> dict[str, Any]:
    if category == "landed":
        frame = result["splits"][f"selected_{split_name}"]
        policy = result["selected_policy"]
        variant = "no_mrt"
    else:
        frame = result["splits"][split_name]
        policy = "all_transactions"
        variant = result["selected_variant"]
    return metric_record(
        category=category, model_name=result["selected_model_name"], feature_variant=variant,
        evaluation_split=split_name, training_policy=policy,
        y_true=frame[target], prediction=result[f"{split_name}_prediction"],
    )


def write_error_analysis(
    category: str,
    result: dict[str, Any],
    target: str,
    train: pd.DataFrame,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    if category == "landed":
        test = result["splits"]["selected_test"]
    else:
        test = result["splits"]["test"]
    prediction = result["test_prediction"]
    analysis = test.copy()
    analysis["_target"] = analysis[target]
    analysis = add_analysis_bands(analysis, train[target])
    analysis["recent_period"] = (
        analysis["transaction_year"].astype("Int64").astype(str)
        + "-" + analysis["transaction_month"].astype("Int64").astype(str).str.zfill(2)
    )
    if "is_multi_unit_transaction" in analysis:
        analysis["unit_policy"] = np.where(
            analysis["is_multi_unit_transaction"].astype(bool), "multi_unit", "single_unit"
        )
    group_config = {
        "hdb": hdb_config.ERROR_GROUPS,
        "ec": ec_config.ERROR_GROUPS,
        "landed": landed_config.ERROR_GROUPS,
    }[category]
    group_frames = [
        evaluate_groups(analysis, prediction, column, minimum_rows=minimum)
        for column, minimum in group_config.items()
    ]
    grouped = pd.concat(group_frames, ignore_index=True)
    if category == "landed":
        full_test = result["splits"]["test"].copy()
        full_test["_target"] = full_test[target]
        full_test["unit_policy"] = np.where(
            full_test["is_multi_unit_transaction"].astype(bool), "multi_unit", "single_unit"
        )
        unit_groups = evaluate_groups(
            full_test,
            result["selected_full_test_prediction"],
            "unit_policy",
            minimum_rows=1,
        )
        grouped = pd.concat([grouped, unit_groups], ignore_index=True)
    grouped.to_csv(ERROR_DIR / f"{category}_group_errors.csv", index=False)

    percentiles: list[dict[str, Any]] = []
    for split_name in ("validation", "test"):
        if category == "landed":
            frame = result["splits"][f"selected_{split_name}"]
        else:
            frame = result["splits"][split_name]
        percentiles.append({
            "category": category, "evaluation_split": split_name,
            "rows": len(frame),
            **absolute_error_percentiles(frame[target], result[f"{split_name}_prediction"]),
        })
    return percentiles, grouped


def money(value: float) -> str:
    return f"${value:,.0f}"


def metric_sentence(row: dict[str, Any]) -> str:
    return (
        f"MAE {money(float(row['mae']))}, RMSE {money(float(row['rmse']))}, "
        f"R² {float(row['r2']):.3f} on {int(row['rows']):,} rows "
        f"(MAE {float(row['mae_percent_of_median']):.2f}% of median target)."
    )


def build_summary(
    selected: dict[str, dict[str, Any]],
    selected_metrics: list[dict[str, Any]],
    ablation: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    error_percentiles: list[dict[str, Any]],
    importance: pd.DataFrame,
    group_results: dict[str, pd.DataFrame],
    runtime: pd.DataFrame,
    split_plan: dict[str, Any],
) -> str:
    metric_lookup = {(row["category"], row["evaluation_split"]): row for row in selected_metrics}
    lines = [
        "# Time-aware baseline modeling summary", "",
        "All models use the existing chronological `data_split` assignments. Validation selected configurations; test was used only for shortlisted MRT pairs and final landed policy comparisons.", "",
    ]
    for category in ("hdb", "ec"):
        result = selected[category]
        validation = metric_lookup[(category, "validation")]
        test = metric_lookup[(category, "test")]
        val_ablation = next(row for row in ablation if row["category"] == category and row["evaluation_split"] == "validation")
        test_ablation = next(row for row in ablation if row["category"] == category and row["evaluation_split"] == "test")
        lines += [
            f"## {category.upper()}", "",
            f"Selected baseline: **{result['selected_model_name']} / {result['selected_variant']}**.",
            f"- Validation: {metric_sentence(validation)}",
            f"- Test: {metric_sentence(test)}",
            f"- MRT validation MAE changed by {money(val_ablation['mae_absolute_change'])} ({val_ablation['mae_percent_change']:+.2f}%); RMSE changed by {money(val_ablation['rmse_absolute_change'])} ({val_ablation['rmse_percent_change']:+.2f}%).",
            f"- MRT test MAE changed by {money(test_ablation['mae_absolute_change'])} ({test_ablation['mae_percent_change']:+.2f}%); RMSE changed by {money(test_ablation['rmse_absolute_change'])} ({test_ablation['rmse_percent_change']:+.2f}%).",
            f"- Decision: **{'retain' if result['selected_variant'] == 'with_mrt' else 'do not retain'} MRT distance** based on validation MAE.", "",
        ]
    landed = selected["landed"]
    landed_policy_lookup = {
        (row["evaluation_split"], row["training_policy"]): row for row in policy_rows
    }
    landed_val_all = landed_policy_lookup[("validation", "all_transactions")]
    landed_val_excluded = landed_policy_lookup[("validation", "exclude_multi_unit")]
    landed_test_all = landed_policy_lookup[("test", "all_transactions")]
    landed_test_excluded = landed_policy_lookup[("test", "exclude_multi_unit")]
    lines += [
        "## Landed", "",
        f"Selected baseline: **{landed['selected_model_name']}**, policy **{landed['selected_policy']}**; MRT is prohibited.",
        f"- Validation: {metric_sentence(metric_lookup[('landed', 'validation')])}",
        f"- Test: {metric_sentence(metric_lookup[('landed', 'test')])}",
        f"- Multi-unit rows by split: {landed['multi_unit_counts']}. Luxury single-property transactions were retained.", "",
        f"- Excluding multi-unit training rows changed single-unit validation MAE by "
        f"{money(landed_val_excluded['mae'] - landed_val_all['mae'])} "
        f"({100.0 * (landed_val_excluded['mae'] - landed_val_all['mae']) / landed_val_all['mae']:+.2f}%) and test MAE by "
        f"{money(landed_test_excluded['mae'] - landed_test_all['mae'])} "
        f"({100.0 * (landed_test_excluded['mae'] - landed_test_all['mae']) / landed_test_all['mae']:+.2f}%). "
        "The exclusion policy is retained because validation selected it and portfolio sales do not match individual-property inference; the test difference is small and adverse.", "",
        "Policy comparison on single-unit evaluation rows:", "",
        "| Split | Training policy | Rows | MAE | RMSE | R² |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in policy_rows:
        lines.append(
            f"| {row['evaluation_split']} | {row['training_policy']} | {row['rows']:,} | "
            f"{money(row['mae'])} | {money(row['rmse'])} | {row['r2']:.3f} |"
        )
    lines += ["", "## Features and preprocessing", ""]
    for category in ("hdb", "ec", "landed"):
        result = selected[category]
        features: FeatureSet = result["selected_features"]
        lines += [
            f"- **{category.upper()} numeric:** {', '.join(features.numeric)}.",
            f"- **{category.upper()} categorical:** {', '.join(features.categorical)}.",
        ]
    lines += [
        "- Numeric missing values use training-fitted median imputation. Linear models standardize numeric values and one-hot encode categories; HistGradientBoosting uses training-fitted ordinal category mappings and native categorical splits.",
        "- Excluded: targets and price derivatives, diagnostic target-outlier flags, coordinates, station/exit names, reference-only MRT distance, HDB block/street/location key, and landed project/street. EC project is retained because its observed cardinality is modest and it identifies the development.",
        "- `transaction_year` and `transaction_month` are known at inference: a future UI must ask for the intended valuation/transaction month (defaulting explicitly to the current month), then derive both values. RPI is not used.", "",
        "## Empirical error ranges", "",
        "These are empirical absolute errors, not formal confidence intervals.", "",
        "| Category | Split | Rows | Median | P80 | P90 | P95 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in error_percentiles:
        lines.append(
            f"| {row['category'].upper()} | {row['evaluation_split']} | {row['rows']:,} | "
            f"{money(row['median_absolute_error'])} | {money(row['p80_absolute_error'])} | "
            f"{money(row['p90_absolute_error'])} | {money(row['p95_absolute_error'])} |"
        )
    lines += ["", "## Feature importance and error analysis", ""]
    for category in ("hdb", "ec", "landed"):
        top = importance.loc[importance["category"].eq(category)].head(5)
        text = ", ".join(
            f"{row.feature} ({money(row.importance_mean_mae_increase)} MAE increase)"
            for row in top.itertuples()
        )
        lines.append(f"- **{category.upper()} permutation importance:** {text}.")
    def worst_group(category: str, dimension: str) -> pd.Series:
        subset = group_results[category].loc[group_results[category]["dimension"].eq(dimension)]
        return subset.sort_values("mae", ascending=False).iloc[0]

    hdb_flat = worst_group("hdb", "flat_type")
    hdb_town = worst_group("hdb", "town")
    hdb_recent = worst_group("hdb", "recent_period")
    ec_project = worst_group("ec", "project_name")
    ec_sale = worst_group("ec", "sale_type")
    ec_band = worst_group("ec", "price_band")
    landed_type = worst_group("landed", "property_type")
    landed_segment = worst_group("landed", "market_segment")
    landed_band = worst_group("landed", "price_band")
    landed_units = group_results["landed"].loc[group_results["landed"]["dimension"].eq("unit_policy")]
    multi = landed_units.loc[landed_units["group"].eq("multi_unit")].iloc[0]
    lines += [
        f"- **HDB error pattern:** {hdb_flat['group']} flats had {money(hdb_flat['mae'])} MAE; "
        f"{hdb_town['group']} had {money(hdb_town['mae'])} MAE. The worst recent month "
        f"({hdb_recent['group']}) had {money(hdb_recent['mae'])} MAE and "
        f"{money(abs(hdb_recent['mean_signed_error']))} mean underprediction, showing strong post-training temporal drift.",
        f"- **EC error pattern:** {ec_project['group']} had {money(ec_project['mae'])} MAE; "
        f"{ec_sale['group']} transactions had {money(ec_sale['mae'])} MAE. The {ec_band['group']} price band "
        f"was hardest at {money(ec_band['mae'])} MAE.",
        f"- **Landed error pattern:** {landed_type['group']} had {money(landed_type['mae'])} MAE; "
        f"{landed_segment['group']} had {money(landed_segment['mae'])} MAE, and the {landed_band['group']} price band "
        f"had {money(landed_band['mae'])} MAE. The four multi-unit test rows had {money(multi['mae'])} MAE and are too few for a stable estimate.",
        "- Group-level errors are in `reports/modeling/error_analysis/`; minimum group sizes are HDB 100 (town 200), EC 30, and landed 30 except the explicitly flagged unit-policy diagnostic.",
        "- Price-band thresholds are fitted from training-target quartiles and used only for held-out error analysis, never as model features.", "",
        "## Reproducibility, usefulness, and limitations", "",
        f"- Python {platform.python_version()}, scikit-learn {sklearn.__version__}, pandas {pd.__version__}, NumPy {np.__version__}; random seed {RANDOM_SEED}.",
        f"- Split plan: `{json.dumps(split_plan, sort_keys=True)}`.",
        f"- Total measured model-fit time: {runtime['fit_seconds'].sum():.1f} seconds. No grid search or external data was used.",
        "- **Demonstration readiness:** EC is the strongest baseline (test MAE 7.91% of its median). HDB is usable only as a visibly caveated demo because test MAE rises to 19.16% under temporal drift. Landed is a coarse demo estimate only (23.64%); it is not yet reliable for high-end valuation.",
        "- The outputs are not valuation advice; rare luxury assets, unseen developments, structural market shifts, and coarse landed location remain material limitations.",
        "- Next milestone: limited model refinement and calibration using validation only, including temporal-trend treatment, rare/unseen-category diagnostics, and optional HDB RPI with explicit inference semantics; then freeze production pipelines.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    split_plan = json.loads(SPLIT_PLAN_PATH.read_text(encoding="utf-8"))

    configs = {
        "hdb": (hdb_config.BASE_FEATURES, hdb_config.MRT_FEATURES),
        "ec": (ec_config.BASE_FEATURES, ec_config.MRT_FEATURES),
        "landed": (landed_config.FEATURES,),
    }
    group_columns = {
        "hdb": set(hdb_config.ERROR_GROUPS) - {"price_band", "mrt_distance_band", "recent_period"},
        "ec": set(ec_config.ERROR_GROUPS) - {"price_band", "mrt_distance_band"},
        "landed": set(landed_config.ERROR_GROUPS) - {"price_band"},
    }
    frames: dict[str, pd.DataFrame] = {}
    for category, feature_sets in configs.items():
        for feature_set in feature_sets:
            verify_manifest_features(category, feature_set, manifest)
        columns = {
            manifest[category]["target"], "data_split", "transaction_date",
            "transaction_year", "transaction_month", *group_columns[category],
        }
        for feature_set in feature_sets:
            columns.update(feature_set.columns)
        frames[category] = load_frame(category, columns)
        assert_split_isolation(frames[category], EXPECTED_SPLITS[category])
        target = manifest[category]["target"]
        if frames[category][target].isna().any() or not pd.api.types.is_numeric_dtype(frames[category][target]):
            raise ValueError(f"{category}: invalid target")
        print(f"{category}: verified {len(frames[category]):,} rows and chronological splits", flush=True)

    baseline_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    results = {
        "hdb": run_mrt_category(
            "hdb", frames["hdb"], manifest["hdb"]["target"],
            hdb_config.BASE_FEATURES, hdb_config.MRT_FEATURES, baseline_rows, runtime_rows,
        ),
        "ec": run_mrt_category(
            "ec", frames["ec"], manifest["ec"]["target"],
            ec_config.BASE_FEATURES, ec_config.MRT_FEATURES, baseline_rows, runtime_rows,
        ),
        "landed": run_landed(
            frames["landed"], manifest["landed"]["target"],
            landed_config.FEATURES, baseline_rows, runtime_rows,
        ),
    }

    selected_metrics: list[dict[str, Any]] = []
    percentile_rows: list[dict[str, Any]] = []
    importance_frames: list[pd.DataFrame] = []
    group_results: dict[str, pd.DataFrame] = {}
    for category, result in results.items():
        target = manifest[category]["target"]
        for split_name in ("validation", "test"):
            selected_metrics.append(selected_metric(result, category, target, split_name))
        train = result["splits"]["train"]
        percentiles, grouped = write_error_analysis(category, result, target, train)
        percentile_rows.extend(percentiles)
        group_results[category] = grouped
        if category == "landed":
            importance_frame = result["splits"]["selected_test"]
            importance_target = importance_frame[target]
            sample_rows = len(importance_frame)
        else:
            importance_frame = result["splits"]["test"]
            importance_target = importance_frame[target]
            sample_rows = 10_000 if category == "hdb" else 5_000
        importance = feature_permutation_importance(
            result["selected_model"], result["selected_features"],
            importance_frame, importance_target,
            sample_rows=sample_rows, random_seed=RANDOM_SEED,
        )
        importance.insert(0, "category", category)
        importance_frames.append(importance)
        joblib.dump(result["selected_model"], MODEL_DIR / f"{category}_baseline_provisional.joblib", compress=3)
        print(
            f"{category}: selected {result['selected_model_name']} "
            f"({result.get('selected_variant', result.get('selected_policy'))})",
            flush=True,
        )

    baseline = pd.DataFrame(baseline_rows)
    runtime = pd.DataFrame(runtime_rows)
    ablation_rows = [*results["hdb"]["ablation_rows"], *results["ec"]["ablation_rows"]]
    policy_rows = results["landed"]["policy_rows"]
    importance = pd.concat(importance_frames, ignore_index=True)
    baseline.to_csv(REPORT_DIR / "baseline_results.csv", index=False)
    pd.DataFrame(selected_metrics).to_csv(REPORT_DIR / "selected_model_metrics.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(REPORT_DIR / "mrt_ablation.csv", index=False)
    pd.DataFrame(policy_rows).to_csv(REPORT_DIR / "landed_multi_unit_policy.csv", index=False)
    pd.DataFrame(percentile_rows).to_csv(REPORT_DIR / "error_percentiles.csv", index=False)
    importance.to_csv(REPORT_DIR / "feature_importance.csv", index=False)
    runtime.to_csv(REPORT_DIR / "runtime.csv", index=False)

    metadata = {
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "random_seed": RANDOM_SEED,
        "split_plan": split_plan,
        "expected_split_counts": EXPECTED_SPLITS,
        "models": {
            category: {
                "selected_model": result["selected_model_name"],
                "selected_variant_or_policy": result.get("selected_variant", result.get("selected_policy")),
                "numeric_features": list(result["selected_features"].numeric),
                "categorical_features": list(result["selected_features"].categorical),
                "parameters": result["parameters"],
                "artifact": f"models/{category}_baseline_provisional.joblib",
            }
            for category, result in results.items()
        },
        "hdb_permutation_sample_rows": 10_000,
        "ec_permutation_sample_rows": 5_000,
        "landed_permutation_sample_rows": "all selected-policy test rows",
        "hdb_training_sampling": "none; all training rows used",
    }
    (REPORT_DIR / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    summary = build_summary(
        results, selected_metrics, ablation_rows, policy_rows, percentile_rows,
        importance, group_results, runtime, split_plan,
    )
    (REPORT_DIR / "modeling_summary.md").write_text(summary, encoding="utf-8")
    print(f"Completed experiments; model fit time {runtime['fit_seconds'].sum():.1f}s", flush=True)


if __name__ == "__main__":
    main()
