"""Build leakage-safe canonical property datasets from immutable raw files.

Run from the repository root with:
    python -m src.preprocess_data
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from .preprocessing import preprocess_ec, preprocess_hdb, preprocess_landed
from .preprocessing.common import quality_flag_summary


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
AUDIT_PATH = REPORTS / "data_audit.json"

SPLIT_BOUNDARIES = {
    "hdb": {"train_end": "2022-12-31", "validation_end": "2024-12-31"},
    "ec": {"train_end": "2023-12-31", "validation_end": "2024-12-31"},
    "landed": {"train_end": "2023-12-31", "validation_end": "2024-12-31"},
}


def build_feature_manifest() -> dict[str, Any]:
    common_ura_excluded = [
        "Unit Price ($ PSF)",
        "Unit Price ($ PSM)",
        "Nett Price($)",
        "any feature derived from transaction_price",
        "same-period aggregate target statistics",
    ]
    return {
        "version": 1,
        "policy": "Only safe_numeric_candidate_features, safe_categorical_candidate_features, and temporal_features may be offered to later model pipelines.",
        "hdb": {
            "target": "resale_price",
            "identifier_reference_columns": [
                "source_file", "source_row_number", "transaction_date", "block", "street_name",
                "storey_range", "source_remaining_lease", "source_remaining_lease_months",
                "location_key", "data_split",
            ],
            "safe_numeric_candidate_features": [
                "floor_area_sqm", "storey_lower", "storey_upper", "storey_midpoint",
                "lease_commence_year", "property_age_at_transaction", "approx_remaining_lease_years",
            ],
            "safe_categorical_candidate_features": ["town", "flat_type", "flat_model"],
            "temporal_features": ["transaction_year", "transaction_month", "transaction_quarter"],
            "location_fields_awaiting_enrichment": ["block", "street_name", "location_key"],
            "eda_only_fields": [],
            "leakage_excluded_fields": [
                "resale_price-derived price_per_sqm", "same-period aggregate target statistics",
                "post-sale valuations/outcome fields",
            ],
            "quality_diagnostic_fields": [
                "quality_invalid_transaction_date", "quality_invalid_target", "quality_invalid_area",
                "quality_negative_property_age", "quality_unparsed_storey", "quality_any",
            ],
            "assumptions": {
                "approx_remaining_lease_years": "99 - property_age_at_transaction; year-granularity proxy only, never a replacement for source remaining_lease.",
            },
        },
        "ec": {
            "target": "transaction_price",
            "identifier_reference_columns": [
                "source_file", "source_row_number", "transaction_date", "tenure", "floor_level",
                "location_key", "data_split",
            ],
            "safe_numeric_candidate_features": [
                "area_sqm", "number_of_units", "floor_lower", "floor_upper", "floor_midpoint",
                "lease_duration_years", "lease_commence_year", "property_age_at_transaction",
                "approx_remaining_lease_years", "is_multi_unit_transaction",
            ],
            "safe_categorical_candidate_features": [
                "project_name", "street_name", "sale_type", "area_type", "property_type",
                "tenure_category", "postal_district", "market_segment",
            ],
            "temporal_features": ["transaction_year", "transaction_month", "transaction_quarter"],
            "location_fields_awaiting_enrichment": ["project_name", "street_name", "postal_district", "location_key"],
            "eda_only_fields": [],
            "leakage_excluded_fields": common_ura_excluded,
            "quality_diagnostic_fields": [
                "quality_invalid_transaction_date", "quality_invalid_target", "quality_invalid_area",
                "quality_lease_starts_after_transaction", "quality_unparsed_tenure",
                "quality_leasehold_missing_commencement", "quality_unparsed_floor", "quality_any",
            ],
            "assumptions": {
                "approx_remaining_lease_years": "lease_duration_years - property_age_at_transaction; only populated for parseable finite leases with nonnegative age.",
            },
        },
        "landed": {
            "target": "transaction_price",
            "identifier_reference_columns": [
                "source_file", "source_row_number", "transaction_date", "tenure", "floor_level",
                "location_key", "data_split",
            ],
            "safe_numeric_candidate_features": [
                "area_sqm", "number_of_units", "lease_duration_years", "lease_commence_year",
                "property_age_at_transaction", "approx_remaining_lease_years", "is_multi_unit_transaction",
            ],
            "safe_categorical_candidate_features": [
                "project_name", "street_name", "sale_type", "area_type", "property_type",
                "tenure_category", "postal_district", "market_segment",
            ],
            "temporal_features": ["transaction_year", "transaction_month", "transaction_quarter"],
            "location_fields_awaiting_enrichment": ["project_name", "street_name", "postal_district", "location_key"],
            "eda_only_fields": ["diagnostic_price_outlier_iqr"],
            "leakage_excluded_fields": common_ura_excluded + [
                "diagnostic_price_outlier_iqr",
            ],
            "quality_diagnostic_fields": [
                "quality_invalid_transaction_date", "quality_invalid_target", "quality_invalid_area",
                "quality_lease_starts_after_transaction", "quality_unparsed_tenure",
                "quality_leasehold_missing_commencement", "quality_unparsed_floor",
                "is_multi_unit_transaction", "quality_any",
            ],
            "assumptions": {
                "approx_remaining_lease_years": "lease_duration_years - property_age_at_transaction; null for freehold, unknown commencement, and lease starts after transaction.",
                "diagnostic_price_outlier_iqr": "Target-derived 1.5xIQR diagnostic retained for EDA only and prohibited from model inputs.",
            },
        },
    }


def candidate_features(category_manifest: dict[str, Any]) -> list[str]:
    return (
        category_manifest["safe_numeric_candidate_features"]
        + category_manifest["safe_categorical_candidate_features"]
        + category_manifest["temporal_features"]
    )


def validate_manifest(manifest: dict[str, Any]) -> None:
    forbidden_tokens = ["unit price", "nett price", "price_per", "price outlier", "iqr_upper"]
    for category in ("hdb", "ec", "landed"):
        section = manifest[category]
        candidates = candidate_features(section)
        if section["target"] in candidates:
            raise ValueError(f"{category}: target appears in candidate features")
        for feature in candidates:
            normalized = feature.lower().replace("_", " ")
            if any(token in normalized for token in forbidden_tokens):
                raise ValueError(f"{category}: leakage-prone feature in candidate list: {feature}")
        overlap = set(candidates) & set(section["leakage_excluded_fields"])
        if overlap:
            raise ValueError(f"{category}: candidate/leakage overlap: {sorted(overlap)}")


def _expected_counts(audit: dict[str, Any]) -> dict[str, int]:
    return {
        "hdb": int(audit["hdb_combined"]["rows"]),
        "ec": int(audit["ec_combined"]["rows"]),
        "landed": int(audit["landed"]["rows"]),
    }


def validate_processed(
    category: str,
    data: pd.DataFrame,
    manifest: dict[str, Any],
    expected_rows: int,
) -> dict[str, Any]:
    section = manifest[category]
    target = section["target"]
    required = set(
        [target, "transaction_date", "transaction_year", "transaction_month", "transaction_quarter", "data_split"]
        + section["identifier_reference_columns"]
        + section["safe_numeric_candidate_features"]
        + section["safe_categorical_candidate_features"]
        + section["temporal_features"]
        + section["quality_diagnostic_fields"]
    )
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{category}: missing canonical fields: {missing}")
    if len(data) != expected_rows:
        raise ValueError(f"{category}: {len(data)} rows do not reconcile to audited {expected_rows}")
    if not is_numeric_dtype(data[target].dtype):
        raise TypeError(f"{category}: target is not numeric")
    if data[target].isna().any() or data[target].le(0).any():
        raise ValueError(f"{category}: target has missing/nonpositive values")
    if data["transaction_date"].isna().any():
        raise ValueError(f"{category}: unparsed transaction dates")
    if data["quality_invalid_area"].any():
        raise ValueError(f"{category}: impossible/missing areas were identified; inspect before modeling")
    if "property_age_at_transaction" in data and data["property_age_at_transaction"].lt(0).any():
        raise ValueError(f"{category}: canonical property-age feature contains negative values")
    candidates = candidate_features(section)
    leakage_overlap = set(candidates) & set(section["leakage_excluded_fields"])
    if leakage_overlap:
        raise ValueError(f"{category}: leakage fields exposed as candidates: {sorted(leakage_overlap)}")

    raw_ura_leakage = {"Unit Price ($ PSF)", "Unit Price ($ PSM)", "Nett Price($)"}
    unexpected_raw_leakage = sorted(raw_ura_leakage & set(data.columns))
    if unexpected_raw_leakage:
        raise ValueError(f"{category}: raw leakage columns present: {unexpected_raw_leakage}")

    return {
        "rows": len(data),
        "columns": len(data.columns),
        "target_numeric": True,
        "target_missing": int(data[target].isna().sum()),
        "target_nonpositive": int(data[target].le(0).sum()),
        "transaction_date_missing": int(data["transaction_date"].isna().sum()),
        "split_counts": {str(k): int(v) for k, v in data["data_split"].value_counts().items()},
        "quality_and_diagnostic_counts": quality_flag_summary(data),
        "rows_dropped": 0,
        "raw_leakage_columns_present": unexpected_raw_leakage,
        "manifest_candidate_leakage_overlap": sorted(leakage_overlap),
    }


def build_split_plan(datasets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    plan: dict[str, Any] = {"version": 1, "method": "Chronological month boundaries; preprocessing must be fit on training rows only."}
    for category, data in datasets.items():
        boundaries = SPLIT_BOUNDARIES[category]
        counts = data["data_split"].value_counts()
        plan[category] = {
            "train": {"through": boundaries["train_end"][:7], "rows": int(counts.get("train", 0))},
            "validation": {
                "from": (pd.Timestamp(boundaries["train_end"]) + pd.offsets.MonthBegin()).strftime("%Y-%m"),
                "through": boundaries["validation_end"][:7],
                "rows": int(counts.get("validation", 0)),
            },
            "test": {
                "from": (pd.Timestamp(boundaries["validation_end"]) + pd.offsets.MonthBegin()).strftime("%Y-%m"),
                "through": data["transaction_date"].max().strftime("%Y-%m"),
                "rows": int(counts.get("test", 0)),
            },
            "split_column": "data_split",
        }
    return plan


def _schema_lines(data: pd.DataFrame) -> str:
    return ", ".join(f"`{column}` ({dtype})" for column, dtype in data.dtypes.items())


def build_summary(
    datasets: dict[str, pd.DataFrame],
    validation: dict[str, Any],
    split_plan: dict[str, Any],
) -> str:
    landed = datasets["landed"]
    lines = [
        "# Canonical preprocessing summary", "",
        "## Outcome", "",
        "The pipeline combines and canonicalizes the audited HDB, EC, and landed transactions without modifying raw files or dropping rows. Private residential, rental, MRT/geocoding, modeling, and application work remain out of scope.", "",
        "Run with `.venv/bin/python -m src.preprocess_data`. Run focused tests with `.venv/bin/python -m unittest discover -s tests -v`.", "",
        "| Dataset | Processed rows | Columns | Rows dropped |",
        "| --- | ---: | ---: | ---: |",
    ]
    for category, data in datasets.items():
        lines.append(f"| {category.upper()} | {len(data):,} | {len(data.columns)} | 0 |")
    lines += ["", "## Canonical schemas", ""]
    for category, data in datasets.items():
        lines += [f"### {category.upper()}", "", _schema_lines(data), ""]

    lines += [
        "## Important transformations and assumptions", "",
        "- All transaction dates are month timestamps at the first day of the source month, with year/month/quarter columns.",
        "- Text is trimmed and internal whitespace collapsed. HDB categories/addresses are uppercased; the audited `MULTI-GENERATION`/`MULTI GENERATION` flat-type spelling is unified to `MULTI GENERATION`. Semantically distinct labels are otherwise preserved.",
        "- HDB storey bands and URA floor bands are parsed into lower/upper/midpoint values.",
        "- HDB `property_age_at_transaction` is transaction year minus lease commencement year. `approx_remaining_lease_years = 99 - property age` is an explicitly approximate year-granularity feature. Source-provided remaining lease is retained separately and historical gaps are not filled.",
        "- URA tenure is parsed into freehold/leasehold, lease duration, and commencement year. Freehold has no finite duration or remaining-lease value. Lease age/remaining lease are null where commencement is unknown or after the transaction month/year.",
        "- Landed multi-unit rows and price extremes remain present. The price IQR flag/fence are target-derived EDA diagnostics and prohibited from model inputs.",
        "- `source_file` and one-based CSV `source_row_number` (header is row 1) preserve provenance.", "",
        "## Quality and diagnostic flags", "",
    ]
    for category in ("hdb", "ec", "landed"):
        counts = validation[category]["quality_and_diagnostic_counts"]
        lines.append(f"- **{category.upper()}:** " + ", ".join(f"`{k}`={v:,}" for k, v in counts.items()))
    lines += [
        "",
        f"The landed 1.5×IQR upper fence is **${validation['landed']['diagnostics']['price_iqr_upper_fence']:,.0f}**; **{int(landed['diagnostic_price_outlier_iqr'].sum()):,}** rows are flagged. **{int(landed['is_multi_unit_transaction'].sum()):,}** landed rows contain more than one unit. No rows are deleted.",
        "",
        "The 51 HDB rows whose commencement year follows the transaction year are retained and flagged; their derived property-age/remaining-lease fields are null. No EC or landed row has this condition.", "",
        "## Leakage protection", "",
        "- Canonical EC/landed CSVs omit raw `Unit Price ($ PSF)`, `Unit Price ($ PSM)`, and `Nett Price($)` entirely.",
        "- No price-per-area or same-period aggregate target feature is created for any category.",
        "- Landed target-derived IQR diagnostics are listed as EDA-only and leakage-excluded, never as candidates.",
        "- `feature_manifest.json` is the allowlist for later modeling. Validation fails if targets or recognized price-derived fields enter candidate lists.", "",
        "## Chronological split plan", "",
        "| Dataset | Train | Validation | Test |",
        "| --- | --- | --- | --- |",
    ]
    for category in ("hdb", "ec", "landed"):
        spec = split_plan[category]
        lines.append(
            f"| {category.upper()} | through {spec['train']['through']} ({spec['train']['rows']:,}) | "
            f"{spec['validation']['from']}–{spec['validation']['through']} ({spec['validation']['rows']:,}) | "
            f"{spec['test']['from']}–{spec['test']['through']} ({spec['test']['rows']:,}) |"
        )
    lines += [
        "", "These boundaries reproduce the audit recommendations. The `data_split` column records membership without duplicating datasets.", "",
        "## MRT/location readiness", "",
        "- HDB: normalized `block`, `street_name`, and `location_key = BLOCK|STREET`.",
        "- EC/landed: normalized `project_name`, `street_name`, `postal_district`, and `location_key = PROJECT|STREET|DISTRICT`.",
        "- No coordinates, MRT data, API calls, or geocoding are included.", "",
        "## Unresolved issues", "",
        "- Identical source rows remain because transaction IDs/unit numbers are unavailable.",
        "- HDB historical remaining lease remains unavailable; the arithmetic proxy is not exact.",
        "- URA generic leasehold strings lack commencement years (seven EC rows and two landed rows), so lease-age features remain null and the rows are flagged.",
        "- Landed generic project labels, multi-unit transactions, and extreme luxury transactions need explicit modeling policy later.",
    ]
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    manifest = build_feature_manifest()
    validate_manifest(manifest)

    datasets = {
        "hdb": preprocess_hdb(RAW / "hdb"),
        "ec": preprocess_ec(RAW / "ura" / "ec"),
        "landed": preprocess_landed(
            RAW / "ura" / "landed" / "URA Private Property Transactions (By District) for Landed Properties.csv"
        ),
    }
    expected = _expected_counts(audit)
    validation = {
        category: validate_processed(category, data, manifest, expected[category])
        for category, data in datasets.items()
    }
    landed_prices = datasets["landed"]["transaction_price"]
    landed_q1, landed_q3 = landed_prices.quantile([0.25, 0.75])
    validation["landed"]["diagnostics"] = {
        "price_iqr_upper_fence": float(landed_q3 + 1.5 * (landed_q3 - landed_q1)),
    }
    split_plan = build_split_plan(datasets)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    for category, data in datasets.items():
        data.to_csv(PROCESSED / f"{category}.csv", index=False, date_format="%Y-%m-%d")
    write_json(PROCESSED / "feature_manifest.json", manifest)
    write_json(PROCESSED / "split_plan.json", split_plan)
    write_json(PROCESSED / "preprocessing_validation.json", validation)
    (REPORTS / "preprocessing_summary.md").write_text(
        build_summary(datasets, validation, split_plan), encoding="utf-8"
    )

    for category, data in datasets.items():
        print(f"{category}: {len(data):,} rows -> data/processed/{category}.csv")
    print("Validation passed; wrote manifest, split plan, validation report, and preprocessing summary.")


if __name__ == "__main__":
    main()
