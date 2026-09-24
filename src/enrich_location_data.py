"""Join cached coordinates and compute nearest-MRT distance locally.

Run after MRT acquisition and (optionally) OneMap geocoding:
    python -m src.enrich_location_data

This command never calls an external API. Unresolved/pending locations remain in
the enriched transaction files with null coordinates and MRT distance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .location.geo import nearest_mrt_exits
from .location.validate_matches import validate_cached_matches


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MRT_PATH = PROCESSED / "mrt_stations.csv"
MRT_METADATA_PATH = PROCESSED / "mrt_stations_metadata.json"
CACHE_PATH = PROCESSED / "property_geocoding_cache.csv"
MANIFEST_PATH = PROCESSED / "feature_manifest.json"
VALIDATION_PATH = PROCESSED / "location_enrichment_validation.json"
SUMMARY_PATH = REPORTS / "location_enrichment_summary.md"
SPOT_CHECKS_PATH = REPORTS / "location_spot_checks.csv"
REPORT_VALIDATION_PATH = REPORTS / "location_validation.json"

MODEL_ELIGIBLE_QUALITIES = {"hdb": {"exact_building"}, "ec": {"project"}, "landed": set()}
HIGH_CONFIDENCE_QUALITIES = {"hdb": {"exact_building"}, "ec": {"project"}, "landed": {"project"}}
CATEGORY_TARGETS = {"hdb": "resale_price", "ec": "transaction_price", "landed": "transaction_price"}


def merge_location_features(transactions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if features["location_key"].duplicated().any():
        raise ValueError("Location features contain duplicate location_key values")
    before = len(transactions)
    merged = transactions.merge(features, on="location_key", how="left", validate="many_to_one", sort=False)
    if len(merged) != before:
        raise ValueError("Location join changed transaction row count")
    return merged


def deterministic_spot_checks(features: pd.DataFrame) -> list[dict[str, Any]]:
    """Select reproducible short/medium/long/fallback/suspicious cached matches."""
    pool = features.loc[
        features["geocode_status"].eq("success") & features["nearest_mrt_distance_reference_m"].notna()
    ].copy()
    if pool.empty:
        return []
    selections: list[tuple[str, int]] = []
    selections.append(("short_distance", int(pool["nearest_mrt_distance_reference_m"].idxmin())))
    median = pool["nearest_mrt_distance_reference_m"].median()
    selections.append(("medium_distance", int((pool["nearest_mrt_distance_reference_m"] - median).abs().idxmin())))
    selections.append(("long_distance", int(pool["nearest_mrt_distance_reference_m"].idxmax())))
    fallback = pool.loc[pd.to_numeric(pool["geocode_query_rank"], errors="coerce").gt(1)].sort_values("location_key")
    if not fallback.empty:
        selections.append(("fallback_query", int(fallback.index[0])))
    suspicious = pool.loc[pool["suspicious_match"]].sort_values("location_key")
    if not suspicious.empty:
        selections.append(("suspicious", int(suspicious.index[0])))

    columns = [
        "location_key", "geocode_query_used", "geocode_query_rank", "matched_address",
        "geocode_quality_original", "geocode_quality", "match_validation_status", "suspicious_match",
        "suspicious_reason", "latitude", "longitude", "nearest_mrt_station", "nearest_mrt_exit",
        "nearest_mrt_distance_reference_m", "nearest_mrt_distance_m",
    ]
    rows: list[dict[str, Any]] = []
    used: set[str] = set()
    for reason, index in selections:
        record = pool.loc[index, columns].replace({np.nan: None}).to_dict()
        key = str(record["location_key"])
        if key in used:
            continue
        used.add(key)
        record = {"sample_reason": reason, **record}
        rows.append(record)
    return rows


def build_location_features(category: str, cache: pd.DataFrame, exits: pd.DataFrame) -> pd.DataFrame:
    category_cache = cache.loc[cache["category"].eq(category)].copy()
    if category_cache["location_key"].duplicated().any():
        raise ValueError(f"{category}: cache contains duplicate location keys")
    category_cache["latitude"] = pd.to_numeric(category_cache["latitude"], errors="coerce")
    category_cache["longitude"] = pd.to_numeric(category_cache["longitude"], errors="coerce")
    category_cache = validate_cached_matches(category_cache)
    successful = category_cache["geocode_status"].eq("success")
    if not category_cache.loc[successful, "latitude"].between(1.15, 1.50).all():
        raise ValueError(f"{category}: successful latitude outside Singapore bounds")
    if not category_cache.loc[successful, "longitude"].between(103.55, 104.15).all():
        raise ValueError(f"{category}: successful longitude outside Singapore bounds")

    usable_coordinates = category_cache.loc[
        successful,
        ["location_key", "latitude", "longitude"],
    ]
    nearest = nearest_mrt_exits(usable_coordinates, exits)
    features = category_cache[[
        "location_key", "latitude", "longitude", "geocode_status", "validated_match_quality",
        "geocode_quality_original", "match_validation_status", "suspicious_match", "suspicious_reason",
        "block_consistent", "street_consistent", "project_consistent", "matched_address", "postal_code",
        "query_used", "query_rank",
    ]].rename(columns={
        "validated_match_quality": "geocode_quality", "query_used": "geocode_query_used",
        "query_rank": "geocode_query_rank",
    })
    features = features.merge(nearest, on="location_key", how="left", validate="one_to_one")
    features = features.rename(columns={"nearest_mrt_distance_m": "nearest_mrt_distance_reference_m"})
    features["mrt_distance_model_eligible"] = (
        features["geocode_status"].eq("success")
        & features["geocode_quality"].isin(MODEL_ELIGIBLE_QUALITIES[category])
        & features["nearest_mrt_distance_reference_m"].notna()
    )
    features["nearest_mrt_distance_m"] = features["nearest_mrt_distance_reference_m"].where(
        features["mrt_distance_model_eligible"]
    )
    features["mrt_distance_missing"] = features["nearest_mrt_distance_m"].isna()
    features["mrt_within_500m"] = (features["nearest_mrt_distance_reference_m"] <= 500).astype("boolean")
    features["mrt_within_1000m"] = (features["nearest_mrt_distance_reference_m"] <= 1000).astype("boolean")
    features.loc[features["nearest_mrt_distance_reference_m"].isna(), ["mrt_within_500m", "mrt_within_1000m"]] = pd.NA
    features["quality_implausible_mrt_distance"] = features["nearest_mrt_distance_reference_m"].gt(60_000).fillna(False)
    return features


def validate_enriched(
    category: str,
    canonical: pd.DataFrame,
    enriched: pd.DataFrame,
    features: pd.DataFrame,
) -> dict[str, Any]:
    target = CATEGORY_TARGETS[category]
    if len(enriched) != len(canonical):
        raise ValueError(f"{category}: enriched row count differs from canonical")
    if not enriched[target].equals(canonical[target]):
        raise ValueError(f"{category}: target changed during location enrichment")
    if not enriched["data_split"].equals(canonical["data_split"]):
        raise ValueError(f"{category}: split assignment changed during location enrichment")
    if enriched["nearest_mrt_distance_m"].dropna().lt(0).any():
        raise ValueError(f"{category}: negative MRT distance")
    if features["location_key"].duplicated().any():
        raise ValueError(f"{category}: one-to-many cache expansion risk")

    status_counts = features["geocode_status"].fillna("missing_cache").value_counts()
    quality_counts = features["geocode_quality"].fillna("unresolved").value_counts()
    original_quality_counts = features["geocode_quality_original"].fillna("unresolved").value_counts()
    distance_mask = enriched["nearest_mrt_distance_reference_m"].notna()
    eligible_mask = enriched["mrt_distance_model_eligible"].fillna(False)
    distances = enriched.loc[eligible_mask, "nearest_mrt_distance_m"]
    high_confidence_mask = enriched["geocode_quality"].isin(HIGH_CONFIDENCE_QUALITIES[category])
    high_confidence_unique_mask = features["geocode_quality"].isin(HIGH_CONFIDENCE_QUALITIES[category])
    suspicious_transaction_mask = enriched["suspicious_match"].fillna(False)
    sample = deterministic_spot_checks(features)
    return {
        "canonical_rows": len(canonical),
        "enriched_rows": len(enriched),
        "unique_locations": len(features),
        "geocode_status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "match_quality_counts": {str(k): int(v) for k, v in quality_counts.items()},
        "match_quality_percent": {
            str(k): round(100.0 * int(v) / len(features), 6) for k, v in quality_counts.items()
        },
        "original_match_quality_counts": {str(k): int(v) for k, v in original_quality_counts.items()},
        "original_match_quality_percent": {
            str(k): round(100.0 * int(v) / len(features), 6) for k, v in original_quality_counts.items()
        },
        "successful_unique_locations": int(features["geocode_status"].eq("success").sum()),
        "geocode_success_percent": round(float(features["geocode_status"].eq("success").mean() * 100), 6),
        "unresolved_or_pending_unique_locations": int(features["geocode_status"].ne("success").sum()),
        "suspicious_successful_locations": int((features["geocode_status"].eq("success") & features["suspicious_match"]).sum()),
        "transactions_with_suspicious_match": int(suspicious_transaction_mask.sum()),
        "validated_high_confidence_unique_locations": int(high_confidence_unique_mask.sum()),
        "validated_high_confidence_transactions": int(high_confidence_mask.sum()),
        "validated_high_confidence_transaction_percent": round(float(high_confidence_mask.mean() * 100), 6),
        "model_eligible_unique_locations": int(features["mrt_distance_model_eligible"].sum()),
        "model_eligible_unique_location_percent": round(float(features["mrt_distance_model_eligible"].mean() * 100), 6),
        "transactions_with_mrt_distance": int(distance_mask.sum()),
        "transaction_mrt_distance_coverage_percent": round(float(distance_mask.mean() * 100), 6),
        "transactions_model_eligible_mrt_distance": int(eligible_mask.sum()),
        "model_eligible_mrt_distance_coverage_percent": round(float(eligible_mask.mean() * 100), 6),
        "mrt_distance_m": {
            "min": None if distances.empty else float(distances.min()),
            "median": None if distances.empty else float(distances.median()),
            "mean": None if distances.empty else float(distances.mean()),
            "p90": None if distances.empty else float(distances.quantile(0.90)),
            "p95": None if distances.empty else float(distances.quantile(0.95)),
            "max": None if distances.empty else float(distances.max()),
        },
        "implausible_distance_flags": int(enriched["quality_implausible_mrt_distance"].sum()),
        "target_unchanged": True,
        "split_unchanged": True,
        "row_count_preserved": True,
        "deterministic_manual_review_sample": sample,
    }


def update_feature_manifest() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for category in ("hdb", "ec", "landed"):
        section = manifest[category]
        for column in [
            "latitude", "longitude", "geocode_status", "geocode_quality", "matched_address",
            "postal_code", "geocode_query_used", "nearest_mrt_station", "nearest_mrt_exit",
            "mrt_distance_model_eligible", "geocode_quality_original", "match_validation_status",
            "suspicious_match", "suspicious_reason", "nearest_mrt_distance_reference_m",
        ]:
            if column not in section["identifier_reference_columns"]:
                section["identifier_reference_columns"].append(column)
        if category in {"hdb", "ec"}:
            for column in ["nearest_mrt_distance_m", "mrt_distance_missing"]:
                if column not in section["safe_numeric_candidate_features"]:
                    section["safe_numeric_candidate_features"].append(column)
        else:
            section["safe_numeric_candidate_features"] = [
                column for column in section["safe_numeric_candidate_features"]
                if column not in {"nearest_mrt_distance_m", "mrt_distance_missing"}
            ]
            for column in ["nearest_mrt_distance_m", "mrt_distance_missing"]:
                if column not in section["eda_only_fields"]:
                    section["eda_only_fields"].append(column)
        for column in ["mrt_within_500m", "mrt_within_1000m"]:
            if column not in section["eda_only_fields"]:
                section["eda_only_fields"].append(column)
        if "quality_implausible_mrt_distance" not in section["quality_diagnostic_fields"]:
            section["quality_diagnostic_fields"].append("quality_implausible_mrt_distance")
        if category == "hdb":
            section.setdefault("safe_feature_conditions", {})["nearest_mrt_distance_m"] = (
                "Candidate feature only for geocode_quality=exact_building; preserve missing as null and use mrt_distance_missing."
            )
            section["location_feature_decision"] = "include_candidate"
        elif category == "ec":
            section.setdefault("safe_feature_conditions", {})["nearest_mrt_distance_m"] = (
                "Candidate feature only for validated geocode_quality=project (exact project phrase plus consistent street); preserve missing as null and use mrt_distance_missing."
            )
            section["location_feature_decision"] = "include_candidate"
        else:
            section.setdefault("safe_feature_conditions", {}).pop("nearest_mrt_distance_m", None)
            section["location_feature_decision"] = (
                "exclude_candidate: only 14 unique project-level locations/65 transactions pass high-confidence validation; street-level coordinates remain approximate/reference-only."
            )
    manifest["location_enrichment"] = {
        "version": 1,
        "coordinate_role": "Reference only; latitude/longitude are not automatically model candidates.",
        "distance_method": "Local Haversine distance to closest physical official LTA MRT exit.",
        "excluded_from_candidates": [
            "latitude", "longitude", "mrt_within_500m", "mrt_within_1000m",
            "geocode_status", "geocode_quality", "matched_address", "postal_code",
            "nearest_mrt_station", "nearest_mrt_exit", "nearest_mrt_distance_reference_m",
        ],
        "missing_value_policy": (
            "Never fill missing MRT distance with zero or an arbitrary constant. Preserve nulls; "
            "HDB/EC may use mrt_distance_missing and training-fold-only imputation or a model with native missing handling."
        ),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}: {value:,}" for key, value in counts.items()) or "none"


def format_counts_with_percent(counts: dict[str, int], total: int) -> str:
    return ", ".join(
        f"{key}: {value:,} ({100.0 * value / total:.2f}%)" for key, value in counts.items()
    ) or "none"


def build_summary(metadata: dict[str, Any], validation: dict[str, Any]) -> str:
    lines = [
        "# Location and MRT enrichment summary", "",
        "## Official MRT reference", "",
        f"- Source: **LTA MRT Station Exit (GEOJSON)** from data.gov.sg (`{metadata['dataset_id']}`).",
        f"- Dataset page: {metadata['dataset_page']}",
        f"- Retrieved: `{metadata['retrieved_at_utc']}`; licence: {metadata['licence']}.",
        f"- Raw layer: **{metadata['raw_feature_count']:,} rail exits** across **{metadata['raw_unique_station_names']:,} station names**. The official layer includes LRT.",
        f"- Model reference: **{metadata['processed_mrt_exit_count']:,} physical MRT exits** across **{metadata['processed_unique_mrt_station_count']:,} MRT station names**. LRT remains preserved in the raw GeoJSON but is excluded from nearest-MRT calculations.", "",
        "## Reproduction", "",
        "```bash",
        ".venv/bin/python -m src.location.download_mrt_data",
        ".venv/bin/python -m src.location.geocode_properties --init-only",
        "# Configure one of these credential options:",
        "export ONEMAP_TOKEN='your-current-token'",
        "# Or: export ONEMAP_EMAIL='you@example.com' ONEMAP_PASSWORD='your-password'",
        ".venv/bin/python -m src.location.geocode_properties",
        ".venv/bin/python -m src.enrich_location_data",
        "```", "",
        "OneMap Search requires authentication. Use `ONEMAP_TOKEN`, or `ONEMAP_EMAIL` plus `ONEMAP_PASSWORD`; secrets are never written to the cache. `.env` is gitignored, and `.env.example` documents variable names. The client defaults to 0.25 seconds between requests (below the documented 300 calls/minute limit), bounded retries, exponential backoff, timeouts, and committed SQLite updates after each attempted unique location.", "",
        "## Strategy and cache", "",
        "- Existing canonical `location_key` values are reused: HDB `BLOCK|STREET`; EC/landed `PROJECT|STREET|DISTRICT`.",
        "- HDB queries block+street, then street only. A block/street evidence match is `exact_building`; street fallback remains `street` precision.",
        "- EC queries project+street, project, then street. Landed uses the same bounded strategy but skips generic project-only searches for `LANDED HOUSING DEVELOPMENT`.",
        "- Results are ranked using block/road/building evidence and Singapore coordinate bounds; the first API result is not accepted blindly.",
        "- SQLite is the durable per-result cache; `property_geocoding_cache.csv` is its auditable snapshot. Successful rows are never re-requested. Explicit flags control retries for unresolved/error rows.", "",
        "## Coverage", "",
        "| Category | Unique locations | API success | Failed/unresolved/pending | ML-eligible unique | Transaction coordinate coverage | ML-eligible transaction coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category in ("hdb", "ec", "landed"):
        item = validation[category]
        lines.append(
            f"| {category.upper()} | {item['unique_locations']:,} | {item['successful_unique_locations']:,} "
            f"({item['geocode_success_percent']:.2f}%) | "
            f"{item['unresolved_or_pending_unique_locations']:,} | "
            f"{item['model_eligible_unique_locations']:,}/{item['unique_locations']:,} "
            f"({item['model_eligible_unique_location_percent']:.2f}%) | "
            f"{item['transactions_with_mrt_distance']:,}/{item['canonical_rows']:,} "
            f"({item['transaction_mrt_distance_coverage_percent']:.2f}%) | "
            f"{item['transactions_model_eligible_mrt_distance']:,}/{item['canonical_rows']:,} "
            f"({item['model_eligible_mrt_distance_coverage_percent']:.2f}%) |"
        )
    lines += ["", "Validated match-quality/status detail:", ""]
    for category in ("hdb", "ec", "landed"):
        item = validation[category]
        lines.append(
            f"- **{category.upper()}:** status [{format_counts(item['geocode_status_counts'])}]; "
            f"original quality [{format_counts_with_percent(item['original_match_quality_counts'], item['unique_locations'])}]; "
            f"validated quality [{format_counts_with_percent(item['match_quality_counts'], item['unique_locations'])}]; "
            f"suspicious successful matches: {item['suspicious_successful_locations']:,} locations / "
            f"{item['transactions_with_suspicious_match']:,} transactions."
        )
    lines += ["", "Nearest-MRT transaction-distance distribution:", ""]
    for category in ("hdb", "ec", "landed"):
        stats = validation[category]["mrt_distance_m"]
        if stats["min"] is None:
            lines.append(f"- **{category.upper()}:** not applicable because no matches are ML-eligible under the final category policy.")
        else:
            lines.append(
                f"- **{category.upper()}:** min {stats['min']:.0f} m; median {stats['median']:.0f} m; "
                f"mean {stats['mean']:.0f} m; P90 {stats['p90']:.0f} m; "
                f"P95 {stats['p95']:.0f} m; max {stats['max']:.0f} m."
            )
    lines += [
        "", "The distributions above use only ML-eligible coordinates, not all API successes.", "",
        "Successful cached matches that fail deterministic address validation remain auditable but are "
        "downgraded to `unresolved` and excluded from ML-eligible MRT coverage.", "",
        "## Feature decisions", "",
        "- **HDB: include as a candidate.** Building-level evidence covers the large majority of unique locations and transactions; street fallbacks remain excluded.",
        "- **EC: include as a candidate.** Cached evidence validation requires the exact normalized project name in the OneMap building/address plus a consistent street. The original classifier under-labeled many of these as street matches; validation corrects only those with both signals.",
        f"- **Landed: exclude from candidate features.** Only "
        f"{validation['landed']['validated_high_confidence_unique_locations']:,}/{validation['landed']['unique_locations']:,} unique locations and "
        f"{validation['landed']['validated_high_confidence_transactions']:,}/{validation['landed']['canonical_rows']:,} transactions "
        f"({validation['landed']['validated_high_confidence_transaction_percent']:.2f}%) have validated project-level evidence. "
        "The conservative policy retains street matches as approximate and rejects weak project/park results.",
        "- Missing MRT distance is retained as null—never zero or an arbitrary constant. HDB/EC include leakage-safe `mrt_distance_missing`; later imputation must be fit on training data only, or use a model with native missing handling.",
        "- Latitude/longitude and nearest-station/exit names remain reference/display-only.", "",
        "## Spot checks", "",
        "Deterministic short/medium/long/fallback/suspicious samples are stored in `reports/location_spot_checks.csv`; machine-readable results are in `reports/location_validation.json` and `data/processed/location_enrichment_validation.json`. They include the original key/query, matched address, original and validated quality, coordinates, nearest exit, distance, and suspicion reason.", "",
        "## Validation and limitations", "",
        "- Enriched row counts, targets, and split assignments are validated against canonical inputs; joins are many-to-one and cannot expand transaction counts.",
        "- Singapore coordinate bounds, nonnegative distances, and a 60 km implausibility flag are checked programmatically.",
        "- Unresolved/pending transactions remain present with null coordinates and MRT distance.",
        "- Coordinates are reference fields, not automatic predictors. `nearest_mrt_distance_m` eligibility is category-specific: HDB exact-building and EC validated-project only; landed is excluded.",
        "- Street-level landed coordinates represent a street search result, not an individual house. Their distances are kept only in `nearest_mrt_distance_reference_m`; the ML candidate `nearest_mrt_distance_m` is null for landed.",
        "- The official layer contains physical exits but does not provide rail line codes; interchange naming is preserved from `STATION_NA`.",
        "- MRT-only exits are used (not LRT), and the nearest physical exit is selected locally with vectorized Haversine distance.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    if not MRT_PATH.exists() or not MRT_METADATA_PATH.exists():
        raise SystemExit("MRT reference missing. Run: python -m src.location.download_mrt_data")
    if not CACHE_PATH.exists():
        raise SystemExit("Geocoding cache missing. Run: python -m src.location.geocode_properties --init-only")

    exits = pd.read_csv(MRT_PATH)
    if exits.empty or not exits["latitude"].between(1.15, 1.50).all() or not exits["longitude"].between(103.55, 104.15).all():
        raise ValueError("MRT reference failed coordinate validation")
    cache = pd.read_csv(CACHE_PATH, dtype={"postal_code": "string", "postal_district": "string"})
    validation: dict[str, Any] = {
        "mrt_reference": {
            "exit_rows": len(exits), "unique_stations": int(exits["station_name"].nunique()),
            "coordinates_plausible": True,
        }
    }
    for category in ("hdb", "ec", "landed"):
        canonical = pd.read_csv(PROCESSED / f"{category}.csv", low_memory=False)
        features = build_location_features(category, cache, exits)
        enriched = merge_location_features(canonical, features)
        validation[category] = validate_enriched(category, canonical, enriched, features)
        enriched.to_csv(PROCESSED / f"{category}_enriched.csv", index=False)
        print(
            f"{category}: {len(enriched):,} rows; "
            f"{validation[category]['transactions_with_mrt_distance']:,} with MRT distance"
        )

    spot_rows = []
    for category in ("hdb", "ec", "landed"):
        for row in validation[category]["deterministic_manual_review_sample"]:
            spot_rows.append({"category": category, **row})
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(spot_rows).to_csv(SPOT_CHECKS_PATH, index=False)

    update_feature_manifest()
    validation_json = json.dumps(validation, indent=2, ensure_ascii=False) + "\n"
    VALIDATION_PATH.write_text(validation_json, encoding="utf-8")
    REPORT_VALIDATION_PATH.write_text(validation_json, encoding="utf-8")
    metadata = json.loads(MRT_METADATA_PATH.read_text(encoding="utf-8"))
    SUMMARY_PATH.write_text(build_summary(metadata, validation), encoding="utf-8")
    print("Location enrichment validation passed; outputs and summary written.")


if __name__ == "__main__":
    main()
