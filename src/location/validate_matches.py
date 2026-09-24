"""Deterministic evidence checks for cached OneMap matches."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .geocode_properties import is_generic_landed_project, normalize_match_text, singapore_coordinate, token_similarity


def exact_phrase(needle: Any, haystack: Any) -> bool:
    normalized_needle = normalize_match_text(needle)
    normalized_haystack = normalize_match_text(haystack)
    return bool(normalized_needle) and normalized_needle in normalized_haystack


def validate_cached_matches(cache: pd.DataFrame) -> pd.DataFrame:
    """Return cache rows with conservative evidence-based final quality labels."""
    result = cache.copy()
    result["geocode_quality_original"] = result["match_quality"].fillna("unresolved")
    result["validated_match_quality"] = "unresolved"
    result["match_validation_status"] = "not_successful"
    result["suspicious_match"] = False
    result["suspicious_reason"] = pd.NA
    result["block_consistent"] = False
    result["street_consistent"] = False
    result["project_consistent"] = False

    for index, row in result.iterrows():
        if row.get("geocode_status") != "success":
            continue
        try:
            coordinate_ok = singapore_coordinate(float(row["latitude"]), float(row["longitude"]))
        except (TypeError, ValueError):
            coordinate_ok = False
        if not coordinate_ok:
            result.loc[index, ["match_validation_status", "suspicious_match", "suspicious_reason"]] = [
                "rejected", True, "coordinate_outside_singapore_or_missing",
            ]
            continue

        street_reference = row.get("matched_road")
        if pd.isna(street_reference) or normalize_match_text(street_reference) in {"", "NIL"}:
            street_reference = row.get("matched_address")
        street_ok = token_similarity(row.get("street_name"), street_reference) >= 0.40
        result.at[index, "street_consistent"] = street_ok
        combined_match = f"{row.get('matched_building') or ''} {row.get('matched_address') or ''}"
        project_ok = exact_phrase(row.get("project_name"), combined_match)
        result.at[index, "project_consistent"] = project_ok

        category = str(row.get("category"))
        original_quality = str(row.get("match_quality") or "unresolved")
        if category == "hdb":
            block_ok = (
                bool(normalize_match_text(row.get("block")))
                and normalize_match_text(row.get("block")) == normalize_match_text(row.get("matched_block"))
            )
            result.at[index, "block_consistent"] = block_ok
            if original_quality == "exact_building" and block_ok and street_ok:
                result.loc[index, ["validated_match_quality", "match_validation_status"]] = [
                    "exact_building", "accepted",
                ]
            elif original_quality == "street" and street_ok:
                result.loc[index, ["validated_match_quality", "match_validation_status"]] = ["street", "accepted"]
            else:
                result.loc[index, ["match_validation_status", "suspicious_match", "suspicious_reason"]] = [
                    "rejected", True, "hdb_block_or_street_evidence_mismatch",
                ]
            continue

        if category == "ec":
            if project_ok and street_ok:
                # Exact project phrase plus consistent street is stronger evidence than
                # the original token-Jaccard label and safely corrects under-classification.
                result.loc[index, ["validated_match_quality", "match_validation_status"]] = ["project", "accepted"]
            elif street_ok:
                result.loc[index, ["validated_match_quality", "match_validation_status"]] = ["street", "accepted"]
            else:
                result.loc[index, ["match_validation_status", "suspicious_match", "suspicious_reason"]] = [
                    "rejected", True, "ec_project_and_street_evidence_mismatch",
                ]
            continue

        # Landed policy deliberately never upgrades an original street result.
        project_name = str(row.get("project_name") or "")
        if (
            original_quality == "project"
            and not is_generic_landed_project(project_name)
            and project_ok
            and street_ok
        ):
            result.loc[index, ["validated_match_quality", "match_validation_status"]] = ["project", "accepted"]
        elif original_quality == "street" and street_ok:
            result.loc[index, ["validated_match_quality", "match_validation_status"]] = ["street", "accepted"]
        else:
            result.loc[index, ["match_validation_status", "suspicious_match", "suspicious_reason"]] = [
                "rejected", True, "landed_project_or_street_evidence_mismatch",
            ]
    return result
