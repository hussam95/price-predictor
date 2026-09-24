#!/usr/bin/env python3
"""Audit the local property datasets without changing source data.

Run from the repository root:
    .venv/bin/python src/audit_data.py

The script writes reports/data_audit.json and reports/data_audit.md. It performs
only local, read-only inspection of files under data/raw and docs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"

HDB_FILES = sorted((RAW / "hdb").glob("*.csv"))
EC_FILES = sorted((RAW / "ura" / "ec").glob("*.csv"))
LANDED_FILE = RAW / "ura" / "landed" / "URA Private Property Transactions (By District) for Landed Properties.csv"
PRIVATE_FILE = RAW / "ura" / "private" / "URA Private Residential Property Transactions (1).csv"
STREETS_FILE = RAW / "reference" / "HDB Street Names.xlsx"

URA_KEY_CATEGORIES = [
    "Project Name", "Street Name", "Property Type", "Type of Sale", "Type of Area",
    "Tenure", "Postal District", "Market Segment", "Floor Level",
]
HDB_KEY_CATEGORIES = ["town", "flat_type", "flat_model", "storey_range"]
PERCENTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def native(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    if pd.isna(value):
        return None
    return value


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.replace(",", "", regex=False), errors="coerce")


def parse_dates(data: pd.DataFrame) -> pd.Series:
    if "month" in data:
        return pd.to_datetime(data["month"], format="%Y-%m", errors="coerce")
    return pd.to_datetime(data["Sale Date"], format="%y-%b", errors="coerce")


def semantic_type(column: str, series: pd.Series) -> str:
    if column in {"month", "Sale Date"}:
        return "year-month"
    values = series.dropna().astype("string").str.strip()
    usable = values[values != "-"]
    if len(usable) and numeric(usable).notna().all():
        if column in {"Postal District", "lease_commence_date", "Number of Units"}:
            return "integer-coded numeric"
        return "numeric"
    if column in {"block", "Project Name", "Street Name", "street_name"}:
        return "identifier/text"
    return "categorical/text"


def target_summary(values: pd.Series) -> dict[str, Any]:
    q = values.quantile(PERCENTILES)
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return native({
        "count": values.count(), "min": values.min(), "mean": values.mean(),
        "median": values.median(), "max": values.max(), "std": values.std(),
        "percentiles": {f"p{int(p * 100):02d}": q.loc[p] for p in PERCENTILES},
        "iqr_outlier_rule": {
            "lower_fence": low, "upper_fence": high,
            "count_below": (values < low).sum(), "count_above": (values > high).sum(),
        },
    })


def category_counts(data: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for column in columns:
        if column not in data:
            continue
        counts = data[column].fillna("<NULL>").astype(str).value_counts(dropna=False)
        result[column] = {
            "cardinality": int(data[column].nunique(dropna=True)),
            "counts": {str(k): int(v) for k, v in counts.items()},
        }
    return result


def profile_csv(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = pd.read_csv(path)
    dates = parse_dates(data)
    target = "resale_price" if "resale_price" in data else "Transacted Price ($)"
    target_values = numeric(data[target])
    columns: dict[str, Any] = {}
    for column in data.columns:
        nulls = int(data[column].isna().sum())
        blanks = int(data[column].fillna("").astype(str).str.strip().eq("").sum())
        dashes = int(data[column].fillna("").astype(str).str.strip().eq("-").sum())
        columns[column] = {
            "pandas_dtype": str(data[column].dtype),
            "inferred_type": semantic_type(column, data[column]),
            "null_count": nulls,
            "null_percent": round(nulls / len(data) * 100, 6),
            "blank_count": blanks,
            "dash_placeholder_count": dashes,
            "cardinality": int(data[column].nunique(dropna=True)),
        }

    malformed = {
        "unparseable_dates": int(dates.isna().sum()),
        "unparseable_target_prices": int(target_values.isna().sum()),
        "nonpositive_target_prices": int((target_values <= 0).sum()),
    }
    numeric_columns = ["floor_area_sqm", "lease_commence_date", "resale_price"] if "month" in data else [
        "Transacted Price ($)", "Area (SQFT)", "Unit Price ($ PSF)", "Area (SQM)",
        "Unit Price ($ PSM)", "Number of Units", "Postal District",
    ]
    for column in numeric_columns:
        if column in data:
            source = data[column].astype("string").str.strip()
            expected = source.notna() & source.ne("-") & source.ne("")
            malformed[f"unparseable_{column}"] = int((expected & numeric(data[column]).isna()).sum())

    categories = HDB_KEY_CATEGORIES if "month" in data else URA_KEY_CATEGORIES
    profile = {
        "path": rel(path), "rows": len(data), "column_count": len(data.columns),
        "columns_in_order": list(data.columns), "columns": columns,
        "exact_duplicate_rows": int(data.duplicated().sum()),
        "date_column": "month" if "month" in data else "Sale Date",
        "date_min": dates.min().strftime("%Y-%m"), "date_max": dates.max().strftime("%Y-%m"),
        "counts_by_year": {str(y): int(n) for y, n in dates.dt.year.value_counts().sort_index().items()},
        "target_column": target, "target_price": target_summary(target_values),
        "key_categories": category_counts(data, categories), "malformed_checks": malformed,
    }
    if "Area (SQFT)" in data:
        sqft, sqm = numeric(data["Area (SQFT)"]), numeric(data["Area (SQM)"])
        psf, psm = numeric(data["Unit Price ($ PSF)"]), numeric(data["Unit Price ($ PSM)"])
        relative_area_error = (sqft - sqm * 10.7639104167).abs() / sqft
        relative_price_error = (target_values / sqft - psf).abs() / psf
        profile["consistency_checks"] = native({
            "area_sqft_vs_sqm_relative_error_max": relative_area_error.max(),
            "area_conversion_rows_over_1_percent_error": (relative_area_error > 0.01).sum(),
            "price_divided_by_sqft_vs_psf_relative_error_max": relative_price_error.max(),
            "price_psf_rows_over_1_percent_error": (relative_price_error > 0.01).sum(),
            "psm_vs_psf_conversion_rows_over_1_percent_error":
                ((psm - psf * 10.7639104167).abs() / psm > 0.01).sum(),
        })
    return data, profile


def cross_file_exact_duplicates(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> int:
    # Count unique exact records common to both files. All source files lack a transaction ID,
    # so a repeated row is not necessarily the same real-world sale.
    lkeys = set(map(tuple, left[columns].astype("string").fillna("<NULL>").to_numpy()))
    rkeys = set(map(tuple, right[columns].astype("string").fillna("<NULL>").to_numpy()))
    return len(lkeys & rkeys)


def split_counts(dates: pd.Series, train_end: str, validation_end: str) -> dict[str, int]:
    train_end_ts = pd.Timestamp(train_end)
    validation_end_ts = pd.Timestamp(validation_end)
    return {
        "train": int((dates <= train_end_ts).sum()),
        "validation": int(((dates > train_end_ts) & (dates <= validation_end_ts)).sum()),
        "test": int((dates > validation_end_ts).sum()),
    }


def money(value: Any) -> str:
    return f"${float(value):,.0f}"


def pct(value: Any) -> str:
    return f"{float(value):.3f}%"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |" for row in rows)
    return "\n".join(lines)


def compact_counts(counts: dict[str, int], limit: int | None = None) -> str:
    items = list(counts.items())[:limit]
    text = ", ".join(f"{k}: {v:,}" for k, v in items)
    if limit and len(counts) > limit:
        text += f", … ({len(counts)} values total)"
    return text


def build_audit() -> tuple[dict[str, Any], str]:
    csv_paths = HDB_FILES + EC_FILES + [LANDED_FILE, PRIVATE_FILE]
    frames: dict[str, pd.DataFrame] = {}
    profiles: dict[str, Any] = {}
    for path in csv_paths:
        frame, profile = profile_csv(path)
        frames[rel(path)] = frame
        profiles[rel(path)] = profile

    hdb_frames = [frames[rel(p)] for p in HDB_FILES]
    hdb = pd.concat(hdb_frames, ignore_index=True, sort=False)
    hdb_dates = parse_dates(hdb)
    # The first file has no remaining_lease, so intersection is the authoritative common schema.
    hdb_common = sorted(set.intersection(*(set(d.columns) for d in hdb_frames)))
    hdb_cross: dict[str, int] = {}
    for i, left in enumerate(HDB_FILES):
        for j in range(i + 1, len(HDB_FILES)):
            right = HDB_FILES[j]
            if set(hdb_frames[i]["month"]).isdisjoint(set(hdb_frames[j]["month"])):
                count = 0
            else:
                count = cross_file_exact_duplicates(hdb_frames[i], hdb_frames[j], hdb_common)
            hdb_cross[f"{left.name} <-> {right.name}"] = count

    normalized_models: dict[str, list[str]] = {}
    for value in sorted(hdb["flat_model"].unique()):
        key = re.sub(r"\s+", " ", value.upper().replace("-", " ")).strip()
        normalized_models.setdefault(key, []).append(value)
    normalized_models = {k: v for k, v in normalized_models.items() if len(v) > 1}

    lease_checks: dict[str, Any] = {}
    for path in HDB_FILES:
        data = frames[rel(path)]
        if "remaining_lease" not in data:
            continue
        dates = parse_dates(data)
        jan_proxy_months = (numeric(data["lease_commence_date"]) + 99 - dates.dt.year) * 12 - (dates.dt.month - 1)
        if pd.api.types.is_numeric_dtype(data["remaining_lease"]):
            official_months = numeric(data["remaining_lease"]) * 12
        else:
            parts = data["remaining_lease"].str.extract(r"(?P<years>\d+) years?(?: (?P<months>\d+) months?)?")
            official_months = numeric(parts["years"]) * 12 + numeric(parts["months"].fillna("0"))
        delta = official_months - jan_proxy_months
        lease_checks[path.name] = native({
            "rows": len(data), "exact_proxy_matches": (delta == 0).sum(),
            "within_12_months": (delta.abs() <= 12).sum(),
            "min_difference_months": delta.min(), "median_difference_months": delta.median(),
            "max_difference_months": delta.max(),
        })

    ref_streets = pd.read_excel(STREETS_FILE)
    street_series = ref_streets["street_names"]
    ref_normalized = set(street_series.dropna().astype(str).str.strip().str.upper())
    hdb_streets = set(hdb["street_name"].str.strip().str.upper())
    xlsx_profile = {
        "path": rel(STREETS_FILE), "sheets": ["Sheet1"], "rows": len(ref_streets),
        "columns_in_order": list(ref_streets.columns),
        "columns": {"street_names": {
            "pandas_dtype": str(street_series.dtype), "inferred_type": "identifier/text",
            "null_count": int(street_series.isna().sum()),
            "null_percent": round(street_series.isna().mean() * 100, 6),
            "cardinality": int(street_series.nunique(dropna=True)),
        }},
        "exact_duplicate_rows": int(ref_streets.duplicated().sum()),
        "duplicate_non_null_street_values": int(street_series.dropna().duplicated().sum()),
        "hdb_unique_streets": len(hdb_streets),
        "hdb_streets_not_in_reference_after_case_normalization": len(hdb_streets - ref_normalized),
        "reference_streets_not_observed_in_transactions": len(ref_normalized - hdb_streets),
    }

    ec1, ec2 = (frames[rel(p)] for p in EC_FILES)
    ec = pd.concat([ec1, ec2], ignore_index=True)
    ec_dates = parse_dates(ec)
    overlap_months = sorted(set(ec1["Sale Date"]) & set(ec2["Sale Date"]))
    ec_cross_duplicates = cross_file_exact_duplicates(ec1, ec2, list(ec1.columns))

    landed = frames[rel(LANDED_FILE)]
    private = frames[rel(PRIVATE_FILE)]
    landed_dates, private_dates = parse_dates(landed), parse_dates(private)

    def location_summary(data: pd.DataFrame, kind: str) -> dict[str, Any]:
        if kind == "hdb":
            address = data["block"].astype(str).str.strip() + " " + data["street_name"].astype(str).str.strip()
            return {
                "unique_block_street_addresses": int(address.nunique()),
                "unique_streets": int(data["street_name"].nunique()),
                "missing_block": int(data["block"].isna().sum()),
                "missing_street": int(data["street_name"].isna().sum()),
            }
        return {
            "unique_projects": int(data["Project Name"].nunique()),
            "unique_streets": int(data["Street Name"].nunique()),
            "unique_project_street_pairs": int(data[["Project Name", "Street Name"]].drop_duplicates().shape[0]),
            "missing_project": int(data["Project Name"].isna().sum()),
            "missing_street": int(data["Street Name"].isna().sum()),
            "missing_postal_district": int(data["Postal District"].isna().sum()),
        }

    private_project_counts = private.groupby("Project Name").size().sort_values(ascending=False)
    landed_project_counts = landed.groupby("Project Name").size().sort_values(ascending=False)
    landed_prices = numeric(landed["Transacted Price ($)"])
    largest_landed = landed.loc[landed_prices.idxmax()].to_dict()

    time_splits = {
        "hdb": {
            "boundaries": {"train_end": "2022-12", "validation": "2023-01 through 2024-12", "test": "2025-01 onward"},
            "counts": split_counts(hdb_dates, "2022-12-31", "2024-12-31"),
        },
        "ec": {
            "boundaries": {"train_end": "2023-12", "validation": "2024-01 through 2024-12", "test": "2025-01 onward"},
            "counts": split_counts(ec_dates, "2023-12-31", "2024-12-31"),
        },
        "landed": {
            "boundaries": {"train_end": "2023-12", "validation": "2024-01 through 2024-12", "test": "2025-01 onward"},
            "counts": split_counts(landed_dates, "2023-12-31", "2024-12-31"),
        },
        "private": {
            "boundaries": {"train_end": "2023-12", "validation": "2024-01 through 2024-12", "test": "2025-01 onward"},
            "counts": split_counts(private_dates, "2023-12-31", "2024-12-31"),
            "warning": "Too few and too narrowly clustered records for a stable general model or reliable holdout evaluation.",
        },
    }

    feature_policy = {
        "hdb": {
            "safe_candidate_predictors": [
                "month (encoded without using future data)", "town", "flat_type", "block/street or later coordinates",
                "storey_range", "floor_area_sqm", "flat_model", "lease_commence_date",
                "remaining_lease when standardized to one numeric representation",
            ],
            "safe_derived_features": [
                "transaction year/month/quarter", "storey lower/upper/midpoint", "property age",
                "approximate remaining lease for old rows", "normalized address", "later coordinates and nearest-MRT distance",
                "historical market indices available as of the transaction quarter",
            ],
            "eda_only": ["resale_price-derived price per sqm", "same-period aggregate/median prices if not lagged"],
            "target_leaking": ["resale_price (target)", "any price per sqm computed from resale_price", "post-sale valuation or outcome fields"],
            "likely_useless_or_redundant": ["raw address after coordinates/address encoding is finalized", "both lease commencement and an exactly derived age/lease feature without regularization"],
        },
        "ura": {
            "safe_candidate_predictors": [
                "Sale Date", "Project Name", "Street Name", "Type of Sale", "Type of Area", "Area (SQFT) or Area (SQM)",
                "Property Type", "Number of Units", "Tenure", "Postal District", "Market Segment", "Floor Level",
            ],
            "safe_derived_features": [
                "transaction year/month/quarter", "one canonical area unit", "tenure class/lease commencement/remaining lease proxy",
                "floor-band midpoint", "project/street normalization", "later coordinates and nearest-MRT distance",
            ],
            "eda_only": ["Unit Price ($ PSF)", "Unit Price ($ PSM)", "Nett Price($) for source validation only"],
            "target_leaking": [
                "Transacted Price ($) (target)", "Unit Price ($ PSF)", "Unit Price ($ PSM)",
                "Nett Price($) if populated", "any total/area or price-derived aggregate computed using the target",
            ],
            "likely_useless_or_redundant": [
                "Area (SQFT) and Area (SQM) together", "Property Type/Type of Area/Number of Units when constant within a category",
                "Floor Level for landed (always '-')", "Nett Price($) in supplied files (always '-')",
            ],
        },
    }

    audit: dict[str, Any] = {
        "scope": {
            "transaction_csv_count": len(csv_paths), "xlsx_count": 1, "pdf_count": 2, "docx_count": 1,
            "raw_data_modified": False, "models_trained": False, "geocoding_performed": False,
        },
        "transaction_files": profiles,
        "xlsx": xlsx_profile,
        "hdb_combined": {
            "rows": len(hdb), "date_min": hdb_dates.min().strftime("%Y-%m"), "date_max": hdb_dates.max().strftime("%Y-%m"),
            "common_columns": hdb_common, "union_columns": sorted(set().union(*(set(d.columns) for d in hdb_frames))),
            "remaining_lease_files": [p.name for p in HDB_FILES if "remaining_lease" in frames[rel(p)]],
            "cross_file_exact_duplicate_rows_by_pair": hdb_cross,
            "cross_file_exact_duplicate_rows_total": sum(hdb_cross.values()),
            "exact_duplicates_on_common_columns_in_combined_data": int(hdb.duplicated(hdb_common).sum()),
            "exact_duplicates_on_all_harmonized_columns": int(hdb.duplicated().sum()),
            "counts_by_year": {str(y): int(n) for y, n in hdb_dates.dt.year.value_counts().sort_index().items()},
            "location": location_summary(hdb, "hdb"), "remaining_lease_proxy_checks": lease_checks,
            "category_spelling_variants": {"flat_type": {"MULTI GENERATION": ["MULTI GENERATION", "MULTI-GENERATION"]}, "flat_model": normalized_models},
            "storey_range_values": sorted(hdb["storey_range"].unique()),
            "target_price": target_summary(numeric(hdb["resale_price"])),
        },
        "ec_combined": {
            "schemas_identical": list(ec1.columns) == list(ec2.columns), "rows": len(ec),
            "date_min": ec_dates.min().strftime("%Y-%m"), "date_max": ec_dates.max().strftime("%Y-%m"),
            "overlap_month_labels": overlap_months, "cross_file_exact_duplicate_rows": ec_cross_duplicates,
            "exact_duplicate_rows_after_concatenation": int(ec.duplicated().sum()),
            "counts_by_year": {str(y): int(n) for y, n in ec_dates.dt.year.value_counts().sort_index().items()},
            "key_categories": category_counts(ec, URA_KEY_CATEGORIES), "location": location_summary(ec, "ura"),
            "target_price": target_summary(numeric(ec["Transacted Price ($)"])),
        },
        "landed": {
            "rows": len(landed), "date_min": landed_dates.min().strftime("%Y-%m"), "date_max": landed_dates.max().strftime("%Y-%m"),
            "property_type_distribution": {str(k): int(v) for k, v in landed["Property Type"].value_counts().items()},
            "project_cardinality": int(landed["Project Name"].nunique()), "street_cardinality": int(landed["Street Name"].nunique()),
            "project_transaction_distribution": native(landed_project_counts.describe(PERCENTILES).to_dict()),
            "postal_districts": sorted(int(x) for x in landed["Postal District"].unique()),
            "market_segment_distribution": {str(k): int(v) for k, v in landed["Market Segment"].value_counts().items()},
            "tenure_cardinality": int(landed["Tenure"].nunique()),
            "tenure_top_counts": {str(k): int(v) for k, v in landed["Tenure"].value_counts().head(15).items()},
            "location": location_summary(landed, "ura"), "largest_transaction": native(largest_landed),
            "transactions_with_multiple_units": int((numeric(landed["Number of Units"]) > 1).sum()),
            "target_price": target_summary(landed_prices),
            "dedicated_model_assessment": "Potentially viable with 8,740 rows, but requires treatment of multi-unit/portfolio sales, extreme luxury transactions, generic project labels, and high-cardinality streets.",
        },
        "private": {
            "rows": len(private), "date_min": private_dates.min().strftime("%Y-%m"), "date_max": private_dates.max().strftime("%Y-%m"),
            "unique_projects": int(private["Project Name"].nunique()), "unique_streets": int(private["Street Name"].nunique()),
            "transactions_per_project": {str(k): int(v) for k, v in private_project_counts.items()},
            "property_type_distribution": {str(k): int(v) for k, v in private["Property Type"].value_counts().items()},
            "postal_district_distribution": {str(k): int(v) for k, v in private["Postal District"].value_counts().items()},
            "market_segment_distribution": {str(k): int(v) for k, v in private["Market Segment"].value_counts().items()},
            "location": location_summary(private, "ura"), "target_price": target_summary(numeric(private["Transacted Price ($)"])),
            "general_model_viable": False,
            "assessment": "Not broad private-residential coverage: 117 rows, five projects/streets, four districts, and 110/117 records are condominiums; one project supplies 87/117 rows. Do not train a general private condo/apartment model from this file.",
        },
        "location_readiness": {
            "hdb": location_summary(hdb, "hdb"), "ec": location_summary(ec, "ura"),
            "landed": location_summary(landed, "ura"), "private": location_summary(private, "ura"),
            "ambiguities": [
                "HDB has no postal code and uses abbreviated/historical street names; 24 of 595 transaction street names are absent from the supplied street reference after simple case normalization.",
                "URA supplies postal district, not full postal code or unit/block number; project/street pairs can span multiple buildings.",
                "Landed Project Name is generic for 2,834 rows ('LANDED HOUSING DEVELOPMENT'), so Street Name is the stronger geocoding key; street names alone may still cover many house numbers.",
                "No source contains latitude/longitude. Geocoding should create a cached property lookup later, without altering raw files.",
            ],
        },
        "time_aware_splits": time_splits,
        "feature_and_leakage_policy": feature_policy,
        "reference_files": {
            rel(RAW / "reference" / "rpi-table.pdf"): {
                "pages": 5, "local_parse": "macOS PDFKit text extraction succeeded",
                "content": "Quarterly HDB Resale Price Index and quarter-over-quarter percentage change",
                "coverage": "1990-Q1 through 2026-Q2", "base_period": "2009-Q1 = 100",
                "join_readiness": "Usable after careful table extraction/validation; join HDB transaction month to calendar quarter. The PDF text order interleaves year labels and quarterly rows, so conversion must be validated before use.",
            },
            rel(RAW / "reference" / "Median resale prices for registered resale applications.pdf"): {
                "pages": 77, "local_parse": "macOS PDFKit text extraction succeeded",
                "content": "Quarterly median resale prices by town and flat type, with suppression for fewer than 20 cases",
                "coverage": "2007-Q2 through 2026-Q2",
                "modeling_note": "Useful for EDA/benchmarking; same-quarter medians would leak contemporaneous target information unless strictly lagged.",
            },
            rel(ROOT / "docs" / "Important Project Info.docx"): {
                "local_parse": "macOS textutil extraction succeeded",
                "content": "Project wishlist for transaction, geospatial, economic, market-index, rental-yield, dashboard, and modeling features",
                "rental_data_present": False,
                "note": "The document requests URA rental data but does not contain transaction-level rental records.",
            },
            "rental_transaction_data_supplied": False,
        },
    }

    # Markdown report
    lines: list[str] = [
        "# Singapore property data audit", "",
        "> Scope: local, read-only inspection. No raw data was changed; no models were trained; no geocoding or downloads were performed.", "",
        "## Executive findings", "",
        f"- HDB is the strongest dataset: **{len(hdb):,} rows**, continuous monthly coverage **{audit['hdb_combined']['date_min']} to {audit['hdb_combined']['date_max']}**, and no null location/target fields. The five files are safely harmonizable after adding nullable `remaining_lease` and normalizing historical labels.",
        f"- EC is usable as a dedicated category: **{len(ec):,} rows**, **{audit['ec_combined']['date_min']} to {audit['ec_combined']['date_max']}**, identical schemas, and no exact cross-part duplicates. The parts overlap only in `23-Aug`, but contain different rows.",
        f"- Landed has **{len(landed):,} rows** and broad district/street coverage. It is plausibly modelable, but portfolio/multi-unit transactions and extreme luxury sales need explicit handling/segmentation.",
        f"- Private is **not viable for a general condo/apartment model**: only **{len(private)} rows**, **5 projects**, and `CASABLANCA` alone contributes **87 rows ({87/len(private)*100:.1f}%)**.",
        "- URA `Unit Price ($ PSF)`, `Unit Price ($ PSM)`, and any populated `Nett Price($)` are target leakage for total-price prediction and must not be model inputs.",
        "- No supplied file contains rental transactions. The DOCX only requests them as a future input.", "",
        "## Dataset inventory", "",
    ]
    inventory_rows = []
    for path in csv_paths:
        p = profiles[rel(path)]
        inventory_rows.append([f"`{rel(path)}`", "CSV", f"{p['rows']:,}", p["column_count"], f"{p['date_min']}–{p['date_max']}"])
    inventory_rows += [
        [f"`{rel(STREETS_FILE)}`", "XLSX", f"{len(ref_streets):,}", 1, "n/a"],
        ["`data/raw/reference/rpi-table.pdf`", "PDF (5 pages)", "n/a", "n/a", "1990-Q1–2026-Q2"],
        ["`data/raw/reference/Median resale prices for registered resale applications.pdf`", "PDF (77 pages)", "n/a", "n/a", "2007-Q2–2026-Q2"],
        ["`docs/Important Project Info.docx`", "DOCX", "n/a", "n/a", "n/a"],
    ]
    lines += [md_table(["File", "Type", "Rows", "Columns", "Date coverage"], inventory_rows), ""]

    lines += ["## Per-file transaction statistics", "",
              "Exact duplicate rows below are byte-equivalent parsed records, not proven duplicate sales: the sources contain no transaction/unit identifier, so multiple real sales can be indistinguishable.", ""]
    stat_rows = []
    for path in csv_paths:
        p, t = profiles[rel(path)], profiles[rel(path)]["target_price"]
        stat_rows.append([Path(path).name, f"{p['rows']:,}", p["exact_duplicate_rows"], f"{p['date_min']}–{p['date_max']}", money(t["min"]), money(t["percentiles"]["p25"]), money(t["median"]), money(t["mean"]), money(t["percentiles"]["p75"]), money(t["percentiles"]["p95"]), money(t["percentiles"]["p99"]), money(t["max"])])
    lines += [md_table(["Dataset", "Rows", "Exact dups", "Dates", "Min", "P25", "Median", "Mean", "P75", "P95", "P99", "Max"], stat_rows), ""]
    outlier_rows = []
    for path in csv_paths:
        rule = profiles[rel(path)]["target_price"]["iqr_outlier_rule"]
        outlier_rows.append([
            Path(path).name, money(rule["lower_fence"]), f"{rule['count_below']:,}",
            money(rule["upper_fence"]), f"{rule['count_above']:,}",
        ])
    lines += ["Mechanical 1.5×IQR flags (diagnostic only; not deletion rules):", "",
              md_table(["Dataset", "Lower fence", "Below", "Upper fence", "Above"], outlier_rows), ""]

    lines += ["### Columns, inferred types, and missingness", ""]
    for path in csv_paths:
        p = profiles[rel(path)]
        lines += [f"#### `{rel(path)}`", "", f"Columns (exact order): `{', '.join(p['columns_in_order'])}`", ""]
        rows = []
        for c, cp in p["columns"].items():
            placeholder = str(cp["dash_placeholder_count"]) if cp["dash_placeholder_count"] else "0"
            rows.append([f"`{c}`", cp["pandas_dtype"], cp["inferred_type"], f"{cp['null_count']:,} ({pct(cp['null_percent'])})", placeholder, f"{cp['cardinality']:,}"])
        lines += [md_table(["Column", "pandas dtype", "Inferred meaning", "Nulls", "`-` placeholders", "Unique"], rows), ""]

    lines += ["## HDB comparison and modeling readiness", ""]
    hdb_schema_rows = []
    for path in HDB_FILES:
        p = profiles[rel(path)]
        hdb_schema_rows.append([path.name, f"{p['rows']:,}", f"{p['date_min']}–{p['date_max']}", "yes" if "remaining_lease" in p["columns_in_order"] else "no", p["exact_duplicate_rows"]])
    lines += [md_table(["File", "Rows", "Coverage", "`remaining_lease`", "Exact dups"], hdb_schema_rows), "",
              f"All ten core columns match. Only `remaining_lease` differs: it exists in the 2015–2016 file as integer years and the 2017+ file as text containing years/months. Add a nullable standardized numeric field before concatenation. Combined: **{len(hdb):,} rows**, **{hdb_dates.min():%Y-%m} to {hdb_dates.max():%Y-%m}**. All file boundaries are month-contiguous; the date definition changes from approval date to registration date at March 2012.", "",
              f"There are **0 exact records shared between any two files**. After concatenation there are **{audit['hdb_combined']['exact_duplicates_on_common_columns_in_combined_data']:,}** repeated rows on the ten common columns (**{audit['hdb_combined']['exact_duplicates_on_all_harmonized_columns']:,}** when nullable `remaining_lease` is included). These are within-file and may be distinct anonymized sales; do not blindly delete them.", "",
              "Naming changes are real but manageable: the 1990s file uses uppercase flat-model labels; later files use title case. `MULTI GENERATION` becomes `MULTI-GENERATION`; equivalent flat-model labels vary by case/hyphen. Newer model values appear over time (`DBSS`, `Type S1/S2`, `3Gen`, loft variants). Storey bands include historical five-floor bins (`01 TO 05`, `06 TO 10`, etc.) alongside modern three-floor bins, especially around 2012; parse bounds rather than treating spelling as immutable.", "",
              "Older exact remaining lease cannot be reconstructed because lease commencement has only a year, not a month. A consistent **approximate** feature can be computed from transaction month and commencement year. Against supplied values, the January-start proxy is within 12 months for 37,152/37,153 rows in 2015–2016 and 224,255/225,320 rows in 2017+; rare larger deviations show why it must be labeled approximate and the supplied detailed value retained where present.", "",
              "Useful HDB predictors: transaction time, town, normalized block+street/location, flat type/model, floor area, parsed storey midpoint/bounds, lease commencement/property age, and standardized remaining lease. Later coordinates and nearest-MRT distance can be joined through normalized address.", ""]

    lines += ["## URA Executive Condominium audit", "",
              f"The two schemas are identical (17 columns). Part 2 covers **{profiles[rel(EC_FILES[1])]['date_min']}–{profiles[rel(EC_FILES[1])]['date_max']}** and Part 1 covers **{profiles[rel(EC_FILES[0])]['date_min']}–{profiles[rel(EC_FILES[0])]['date_max']}**. They overlap in **August 2023** only (61 Part 1 rows and 384 Part 2 rows), with **0 exact cross-file matches**. Therefore concatenate them, preserve source-file provenance, and do not pre-deduplicate the {ec.duplicated().sum():,} indistinguishable within-file/combined repeated rows without a transaction identifier.", "",
              f"Combined: **{len(ec):,} rows**, **{ec_dates.min():%Y-%m}–{ec_dates.max():%Y-%m}**, **82 projects**, **64 streets**, and **82 project/street pairs**. Category distributions:", ""]
    for c in URA_KEY_CATEGORIES:
        counts = audit["ec_combined"]["key_categories"][c]["counts"]
        lines.append(f"- `{c}` ({len(counts)} values): {compact_counts(counts, 12)}")
    lines += ["", "`Property Type`, `Type of Area`, and `Number of Units` are constant for EC; they add no within-EC signal. Tenure is almost entirely a parsable 99-year lease commencement string; seven rows use generic `99 years leasehold` and lack a commencement year.", ""]

    lines += ["## Landed-property audit", "",
              f"Coverage is **{len(landed):,} rows**, **{landed_dates.min():%Y-%m}–{landed_dates.max():%Y-%m}**. Property types: {compact_counts(audit['landed']['property_type_distribution'])}. Projects: **478**; streets: **1,493**; project/street pairs: **2,046**. `LANDED HOUSING DEVELOPMENT` is a generic label on **2,834 rows**, so Project Name quality is mixed and Street Name carries more location detail.", "",
              f"All location fields are populated. Coverage spans **24 postal districts** ({', '.join(map(str, audit['landed']['postal_districts']))}) and all market segments: {compact_counts(audit['landed']['market_segment_distribution'])}. Tenure has **65 raw values**: Freehold dominates (5,852); the remainder include many 99/999-year and a few unusual lease lengths that should be parsed into tenure class, term, and start year.", "",
              f"The target is strongly right-skewed (median {money(audit['landed']['target_price']['median'])}, P99 {money(audit['landed']['target_price']['percentiles']['p99'])}, max {money(audit['landed']['target_price']['max'])}). The maximum is a **$815m, 25-unit, 263,796.58 sqft** transaction; **{audit['landed']['transactions_with_multiple_units']} rows** contain multiple units. These are valid-looking portfolio/development transactions but are not comparable to a single home and need flags/segmentation, not silent deletion.", "",
              "Assessment: sufficient for a dedicated exploratory landed model, subject to chronological evaluation, robust losses/log target, explicit multi-unit treatment, and careful high-cardinality location encoding. Performance should also be reported by house type and price tier.", ""]

    lines += ["## Private-residential audit", "",
              f"The file has **117 rows**, **2021-09–2026-08**, **5 projects**, **5 streets**, **4 postal districts**, and **3 market segments**. Transactions per project: {compact_counts(audit['private']['transactions_per_project'])}. Property types: {compact_counts(audit['private']['property_type_distribution'])}. Districts: {compact_counts(audit['private']['postal_district_distribution'])}. Segments: {compact_counts(audit['private']['market_segment_distribution'])}.", "",
              "This is a narrow, apparently selected subset rather than broad private-residential coverage. It mixes 110 condominium rows with four detached-house, two terrace-house, and one apartment record; project concentration and the $762k–$10.8m target span make a general relationship unlearnable from 117 rows. **Recommendation: do not build a general private condo/apartment predictor from this file.** Keep it for data-pipeline tests/EDA or clearly project-specific experiments only.", ""]

    lines += ["## Data leakage and feature policy", "", "### HDB", ""]
    for label, values in feature_policy["hdb"].items():
        lines.append(f"- **{label.replace('_', ' ').title()}:** " + "; ".join(values) + ".")
    lines += ["", "### URA (EC, landed, private)", ""]
    for label, values in feature_policy["ura"].items():
        lines.append(f"- **{label.replace('_', ' ').title()}:** " + "; ".join(values) + ".")
    lines += ["", "Why PSF/PSM leak: they are effectively `Transacted Price / Area` (rounded), so a model receiving unit price plus area can algebraically reconstruct the target. Their strong correlation is mechanical, not predictive information available before sale. `Nett Price($)` is `-` in every supplied URA row, but if later populated it is another sale outcome and must remain excluded.", ""]

    lines += ["## Data quality, malformed values, and outliers", "",
              "- All transaction files have zero true nulls, blank cells, unparseable dates, unparseable target prices, and non-positive prices. URA uses `-` as a semantic missing-value placeholder: all `Nett Price($)` values, all landed `Floor Level` values, and six private `Floor Level` values are `-`.",
              "- URA area/unit-price pairs are arithmetically consistent up to expected display rounding; retain one area unit only. PSF/PSM remain EDA-only leakage fields.",
              "- HDB `block` is an identifier, not numeric: many values contain suffix letters. Postal District is also categorical despite being read as an integer.",
              "- Exact repeated rows occur in most large files, but absence of unit/transaction IDs means they cannot safely be declared erroneous duplicates.",
              f"- HDB target extremes range from {money(audit['hdb_combined']['target_price']['min'])} (historical) to {money(audit['hdb_combined']['target_price']['max'])}; EC from {money(audit['ec_combined']['target_price']['min'])} to {money(audit['ec_combined']['target_price']['max'])}; landed has the portfolio/luxury extremes noted above; private mixes dissimilar asset types and price tiers. Flag and inspect—do not delete—using time/category-aware rules rather than one global IQR rule.", ""]

    loc_rows = [
        ["HDB", "n/a", audit["location_readiness"]["hdb"]["unique_streets"], audit["location_readiness"]["hdb"]["unique_block_street_addresses"], "0 block; 0 street"],
        ["EC", audit["location_readiness"]["ec"]["unique_projects"], audit["location_readiness"]["ec"]["unique_streets"], audit["location_readiness"]["ec"]["unique_project_street_pairs"], "0 project; 0 street; 0 district"],
        ["Landed", audit["location_readiness"]["landed"]["unique_projects"], audit["location_readiness"]["landed"]["unique_streets"], audit["location_readiness"]["landed"]["unique_project_street_pairs"], "0 project; 0 street; 0 district"],
        ["Private", audit["location_readiness"]["private"]["unique_projects"], audit["location_readiness"]["private"]["unique_streets"], audit["location_readiness"]["private"]["unique_project_street_pairs"], "0 project; 0 street; 0 district"],
    ]
    lines += ["## Location/MRT readiness", "", md_table(["Category", "Unique projects", "Unique streets", "Unique address/project-street keys", "Missing identifiers"], loc_rows), ""]
    lines.extend(f"- {item}" for item in audit["location_readiness"]["ambiguities"])
    lines += ["", "The HDB street workbook contains 740 rows, 81 null rows, 645 unique non-null street names, and 14 duplicated non-null names. It covers most transaction streets after case normalization, but 24/595 transaction street names are absent; historical/abbreviated names require a normalization/alias table before geocoding.", ""]

    lines += ["## Time-aware readiness", "", "Transaction counts by year:", ""]
    for label, counts in [("HDB", audit["hdb_combined"]["counts_by_year"]), ("EC", audit["ec_combined"]["counts_by_year"]), ("Landed", profiles[rel(LANDED_FILE)]["counts_by_year"]), ("Private", profiles[rel(PRIVATE_FILE)]["counts_by_year"])]:
        lines.append(f"- **{label}:** {compact_counts(counts)}")
    lines += ["", "Candidate chronological splits (fit preprocessing only on each training window):", ""]
    split_rows = []
    for label, spec in time_splits.items():
        b, c = spec["boundaries"], spec["counts"]
        split_rows.append([label.title(), f"through {b['train_end']} ({c['train']:,})", f"{b['validation']} ({c['validation']:,})", f"{b['test']} ({c['test']:,})"])
    lines += [md_table(["Category", "Train", "Validation", "Test"], split_rows), "",
              "For HDB, the long history may warrant a rolling/expanding-window backtest and/or a recent-era training window because market regimes and label definitions changed. Private counts are shown only for completeness; the split is too small and concentrated to support meaningful evaluation.", ""]

    lines += ["## Reference files", "",
              "- `HDB Street Names.xlsx`: one sheet/column; useful as a street-name reference but contains blanks, duplicates, and incomplete historical transaction coverage as quantified above.",
              "- `rpi-table.pdf`: local text extraction succeeded. It contains quarterly HDB Resale Price Index data from **1990-Q1 through 2026-Q2**, with 2009-Q1 = 100 and quarter-over-quarter changes. It can later be converted to a validated quarter/index table and joined to HDB by transaction quarter. The extracted text interleaves year labels with rows, so conversion needs sequence/spot checks.",
              "- `Median resale prices for registered resale applications.pdf`: 77 pages, quarterly town × flat-type median tables from **2007-Q2 through 2026-Q2**, with suppressed low-count cells. Useful for EDA/benchmarking; same-quarter values must not become predictors unless lagged.",
              "- `Important Project Info.docx`: a requirements/wishlist document, not data. It requests rental, geospatial, economic, and market data and suggests model/UI outputs. It does **not** supply rental records.",
              "- No CSV/XLSX/PDF/DOCX supplied here contains transaction-level rental data; rental prediction/yield is out of scope.", ""]

    lines += ["## Recommended preprocessing sequence (do not execute yet)", "",
              "1. Create category-specific canonical schemas while retaining `source_file` and raw categorical fields.",
              "2. Normalize HDB flat-model/type spelling; parse all storey ranges; standardize supplied remaining lease and create a clearly named approximate value for older rows.",
              "3. Parse URA money/area/count fields, keep one area unit, parse tenure, convert `-` to missing, and permanently exclude PSF/PSM/nett price from model feature lists.",
              "4. Define single-unit versus portfolio landed segments and document outlier flags without deleting source records.",
              "5. Build cached unique-location tables for later geocoding, starting with HDB block+street and URA project+street+district; preserve unresolved/ambiguous matches.",
              "6. Convert and validate RPI by quarter, then join using only information available at prediction time.",
              "7. Materialize chronological splits before fitting encoders/imputers; establish naive time/location baselines before advanced models.",
              "8. Exclude the supplied private file from a general private model until substantially broader transaction coverage is provided.", ""]

    return audit, "\n".join(lines).rstrip() + "\n"


def main() -> None:
    audit, markdown = build_audit()
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "data_audit.json").write_text(json.dumps(native(audit), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (REPORTS / "data_audit.md").write_text(markdown, encoding="utf-8")
    print(f"Wrote {rel(REPORTS / 'data_audit.md')}")
    print(f"Wrote {rel(REPORTS / 'data_audit.json')}")


if __name__ == "__main__":
    main()
