"""Geocode unique canonical property keys through OneMap with durable caching.

Initialize/export the cache without credentials:
    python -m src.location.geocode_properties --init-only

Run live geocoding after configuring ONEMAP_TOKEN, or ONEMAP_EMAIL and
ONEMAP_PASSWORD:
    python -m src.location.geocode_properties

The SQLite file is the transactional/resumable cache. A CSV snapshot is
exported after initialization and after each run for auditability.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
DATABASE_PATH = PROCESSED / "property_geocoding_cache.sqlite"
CSV_PATH = PROCESSED / "property_geocoding_cache.csv"
SEARCH_URL = "https://www.onemap.gov.sg/api/common/elastic/search"
AUTH_URL = "https://www.onemap.gov.sg/api/auth/post/getToken"
SOURCE = "OneMap Search API, Singapore Land Authority"
SUCCESS_STATUS = "success"
SUCCESS_QUALITIES = {"exact_building", "project", "street", "district_approximate"}

CACHE_COLUMNS = [
    "category", "location_key", "block", "project_name", "street_name", "postal_district",
    "query_used", "query_rank", "latitude", "longitude", "matched_address", "matched_building",
    "matched_block", "matched_road", "postal_code", "geocode_status", "match_quality", "source",
    "error_reason", "response_found", "attempted_at_utc",
]


@dataclass(frozen=True)
class QueryCandidate:
    query: str
    kind: str


def normalize_match_text(value: Any) -> str:
    raw = str(value or "").upper().replace("C'WEALTH", "COMMONWEALTH")
    text = re.sub(r"[^A-Z0-9]+", " ", raw).strip()
    replacements = {
        "AVE": "AVENUE", "ST": "STREET", "RD": "ROAD", "DR": "DRIVE",
        "CRES": "CRESCENT", "CTRL": "CENTRAL", "NTH": "NORTH", "STH": "SOUTH",
        "JLN": "JALAN", "LOR": "LORONG", "BT": "BUKIT", "UPP": "UPPER", "PK": "PARK",
    }
    return " ".join(replacements.get(token, token) for token in text.split())


def token_similarity(left: Any, right: Any) -> float:
    lset, rset = set(normalize_match_text(left).split()), set(normalize_match_text(right).split())
    if not lset or not rset:
        return 0.0
    return len(lset & rset) / len(lset | rset)


def is_generic_landed_project(project_name: str) -> bool:
    return normalize_match_text(project_name) in {"LANDED HOUSING DEVELOPMENT", "NIL", ""}


def build_query_fallbacks(record: dict[str, Any]) -> list[QueryCandidate]:
    category = str(record["category"])
    block = str(record.get("block") or "").strip()
    project = str(record.get("project_name") or "").strip()
    street = str(record.get("street_name") or "").strip()
    if category == "hdb":
        return [QueryCandidate(f"{block} {street}", "block_street"), QueryCandidate(street, "street")]
    if category == "ec":
        return [
            QueryCandidate(f"{project} {street}", "project_street"),
            QueryCandidate(project, "project"),
            QueryCandidate(street, "street"),
        ]
    candidates = [QueryCandidate(f"{project} {street}", "project_street")]
    if not is_generic_landed_project(project):
        candidates.append(QueryCandidate(project, "project"))
    candidates.append(QueryCandidate(street, "street"))
    # Preserve order while removing identical normalized queries.
    unique: list[QueryCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = normalize_match_text(candidate.query)
        if key and key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def singapore_coordinate(latitude: float, longitude: float) -> bool:
    return 1.15 <= latitude <= 1.50 and 103.55 <= longitude <= 104.15


def assess_result(
    record: dict[str, Any], candidate: QueryCandidate, result: dict[str, Any]
) -> tuple[str, float] | None:
    try:
        latitude = float(result.get("LATITUDE"))
        longitude = float(result.get("LONGITUDE") or result.get("LONGTITUDE"))
    except (TypeError, ValueError):
        return None
    if not singapore_coordinate(latitude, longitude):
        return None

    category = str(record["category"])
    address = " ".join(str(result.get(field) or "") for field in ("SEARCHVAL", "BUILDING", "ADDRESS"))
    road = result.get("ROAD_NAME") or address
    road_score = token_similarity(record.get("street_name"), road)

    if category == "hdb":
        expected_block = normalize_match_text(record.get("block"))
        actual_block = normalize_match_text(result.get("BLK_NO"))
        if expected_block and expected_block == actual_block and road_score >= 0.35:
            return "exact_building", 2.0 + road_score
        if candidate.kind == "street" and road_score >= 0.50:
            return "street", road_score
        return None

    project = str(record.get("project_name") or "")
    project_score = token_similarity(project, address)
    if category == "ec" and candidate.kind in {"project_street", "project"} and project_score >= 0.40:
        return "project", 1.0 + project_score + min(road_score, 0.5)
    if category == "landed" and not is_generic_landed_project(project) and project_score >= 0.65:
        return "project", 1.0 + project_score
    if road_score >= 0.50:
        return "street", road_score
    return None


def choose_result(
    record: dict[str, Any], candidate: QueryCandidate, results: list[dict[str, Any]]
) -> tuple[dict[str, Any], str] | None:
    ranked: list[tuple[float, dict[str, Any], str]] = []
    for result in results:
        assessed = assess_result(record, candidate, result)
        if assessed:
            quality, score = assessed
            ranked.append((score, result, quality))
    if not ranked:
        return None
    _, result, quality = max(ranked, key=lambda item: item[0])
    return result, quality


def load_location_inventory(processed_directory: Path = PROCESSED) -> list[dict[str, Any]]:
    specifications = {
        "hdb": ["block", "street_name", "location_key"],
        "ec": ["project_name", "street_name", "postal_district", "location_key"],
        "landed": ["project_name", "street_name", "postal_district", "location_key"],
    }
    inventory: list[dict[str, Any]] = []
    for category, columns in specifications.items():
        data = pd.read_csv(processed_directory / f"{category}.csv", usecols=columns, dtype="string")
        unique = data.drop_duplicates("location_key", keep="first")
        for row in unique.to_dict("records"):
            inventory.append({
                "category": category,
                "location_key": row["location_key"],
                "block": row.get("block"),
                "project_name": row.get("project_name"),
                "street_name": row.get("street_name"),
                "postal_district": row.get("postal_district"),
            })
    return inventory


class GeocodeCache:
    def __init__(self, path: Path = DATABASE_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS geocodes (
                category TEXT NOT NULL,
                location_key TEXT NOT NULL,
                block TEXT,
                project_name TEXT,
                street_name TEXT,
                postal_district TEXT,
                query_used TEXT,
                query_rank INTEGER,
                latitude REAL,
                longitude REAL,
                matched_address TEXT,
                matched_building TEXT,
                matched_block TEXT,
                matched_road TEXT,
                postal_code TEXT,
                geocode_status TEXT NOT NULL DEFAULT 'pending',
                match_quality TEXT NOT NULL DEFAULT 'unresolved',
                source TEXT,
                error_reason TEXT,
                response_found INTEGER,
                attempted_at_utc TEXT,
                PRIMARY KEY (category, location_key)
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def seed(self, records: Iterable[dict[str, Any]]) -> int:
        before = self.connection.total_changes
        self.connection.executemany(
            """
            INSERT OR IGNORE INTO geocodes
                (category, location_key, block, project_name, street_name, postal_district)
            VALUES (:category, :location_key, :block, :project_name, :street_name, :postal_district)
            """,
            records,
        )
        self.connection.commit()
        return self.connection.total_changes - before

    def get(self, category: str, location_key: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM geocodes WHERE category=? AND location_key=?", (category, location_key)
        ).fetchone()
        return dict(row) if row else None

    def pending(
        self,
        *,
        category: str | None = None,
        retry_unresolved: bool = False,
        retry_errors: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        statuses = ["pending"]
        if retry_unresolved:
            statuses.append("unresolved")
        if retry_errors:
            statuses.append("error")
        placeholders = ",".join("?" for _ in statuses)
        sql = f"SELECT * FROM geocodes WHERE geocode_status IN ({placeholders})"
        parameters: list[Any] = list(statuses)
        if category:
            sql += " AND category=?"
            parameters.append(category)
        sql += " ORDER BY category, location_key"
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(limit)
        return [dict(row) for row in self.connection.execute(sql, parameters).fetchall()]

    def update(self, row: dict[str, Any]) -> None:
        assignments = [f"{column} = :{column}" for column in CACHE_COLUMNS if column not in {"category", "location_key"}]
        self.connection.execute(
            f"UPDATE geocodes SET {', '.join(assignments)} WHERE category=:category AND location_key=:location_key",
            {column: row.get(column) for column in CACHE_COLUMNS},
        )
        self.connection.commit()

    def dataframe(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM geocodes ORDER BY category, location_key", self.connection
        )[CACHE_COLUMNS]

    def export_csv(self, path: Path = CSV_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.dataframe().to_csv(path, index=False)


class OneMapClient:
    def __init__(
        self,
        token: str,
        *,
        timeout: int = 20,
        max_retries: int = 3,
        request_delay: float = 0.25,
    ):
        self.token = token
        self.timeout = timeout
        self.max_retries = max_retries
        self.request_delay = request_delay
        self.last_request_at = 0.0

    def search(self, query: str) -> dict[str, Any]:
        parameters = urllib.parse.urlencode({
            "searchVal": query, "returnGeom": "Y", "getAddrDetails": "Y", "pageNum": 1,
        })
        url = f"{SEARCH_URL}?{parameters}"
        for attempt in range(self.max_retries + 1):
            wait = self.request_delay - (time.monotonic() - self.last_request_at)
            if wait > 0:
                time.sleep(wait)
            request = urllib.request.Request(url, headers={
                "Authorization": self.token,
                "User-Agent": "singapore-property-predictor/1.0 (local data preparation)",
            })
            try:
                self.last_request_at = time.monotonic()
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if payload.get("error"):
                    raise RuntimeError(f"OneMap rejected the request: {payload['error']}")
                return payload
            except urllib.error.HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt == self.max_retries:
                    raise
            except urllib.error.URLError:
                if attempt == self.max_retries:
                    raise
            time.sleep(min(2**attempt, 8))
        raise RuntimeError("Unreachable OneMap retry state")


def resolve_onemap_token() -> str:
    direct = os.environ.get("ONEMAP_TOKEN", "").strip()
    if direct:
        return direct
    email = os.environ.get("ONEMAP_EMAIL", "").strip()
    password = os.environ.get("ONEMAP_PASSWORD", "")
    if not email or not password:
        raise RuntimeError(
            "OneMap Search requires authentication. Set ONEMAP_TOKEN, or set both "
            "ONEMAP_EMAIL and ONEMAP_PASSWORD, then rerun. See .env.example."
        )
    body = json.dumps({"email": email, "password": password}).encode("utf-8")
    request = urllib.request.Request(
        AUTH_URL,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "singapore-property-predictor/1.0"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    token = str(payload.get("access_token") or "").strip()
    if not token:
        raise RuntimeError("OneMap authentication did not return an access token")
    return token


def geocode_record(record: dict[str, Any], client: Any) -> dict[str, Any]:
    completed = dict(record)
    completed.update({
        "query_used": None, "query_rank": None, "latitude": None, "longitude": None,
        "matched_address": None, "matched_building": None, "matched_block": None,
        "matched_road": None, "postal_code": None, "geocode_status": "unresolved",
        "match_quality": "unresolved", "source": SOURCE, "error_reason": None,
        "response_found": 0, "attempted_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    try:
        for rank, candidate in enumerate(build_query_fallbacks(record), start=1):
            payload = client.search(candidate.query)
            results = payload.get("results") or []
            completed["response_found"] = max(int(payload.get("found") or len(results)), completed["response_found"])
            chosen = choose_result(record, candidate, results)
            if not chosen:
                continue
            result, quality = chosen
            completed.update({
                "query_used": candidate.query,
                "query_rank": rank,
                "latitude": float(result["LATITUDE"]),
                "longitude": float(result.get("LONGITUDE") or result.get("LONGTITUDE")),
                "matched_address": result.get("ADDRESS") or result.get("SEARCHVAL"),
                "matched_building": result.get("BUILDING"),
                "matched_block": result.get("BLK_NO"),
                "matched_road": result.get("ROAD_NAME"),
                "postal_code": result.get("POSTAL"),
                "geocode_status": SUCCESS_STATUS,
                "match_quality": quality,
            })
            return completed
    except Exception as exc:  # Persist the failure without exposing credentials/request headers.
        completed["geocode_status"] = "error"
        completed["error_reason"] = f"{type(exc).__name__}: {exc}"
    return completed


def process_records(cache: GeocodeCache, records: list[dict[str, Any]], client: Any) -> dict[str, int]:
    counts = {"requested": 0, "reused_success": 0, "success": 0, "unresolved": 0, "error": 0}
    for index, record in enumerate(records, start=1):
        current = cache.get(str(record["category"]), str(record["location_key"]))
        if current and current["geocode_status"] == SUCCESS_STATUS:
            counts["reused_success"] += 1
            continue
        counts["requested"] += 1
        completed = geocode_record(record, client)
        cache.update(completed)
        counts[completed["geocode_status"]] += 1
        if index == 1 or index % 25 == 0 or index == len(records):
            print(f"Geocoded {index:,}/{len(records):,}; status={completed['geocode_status']}")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-only", action="store_true", help="Seed/export pending unique keys; do not call OneMap")
    parser.add_argument("--category", choices=["hdb", "ec", "landed"])
    parser.add_argument("--limit", type=int, help="Maximum cache rows to attempt in this run")
    parser.add_argument("--retry-unresolved", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--request-delay", type=float, default=0.25, help="Seconds between requests; 0.25 stays below 300/min")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = GeocodeCache()
    try:
        inventory = load_location_inventory()
        inserted = cache.seed(inventory)
        cache.export_csv()
        print(f"Cache contains {len(cache.dataframe()):,} unique category/location keys ({inserted:,} inserted).")
        if args.init_only:
            print("Initialization only: no OneMap requests were made.")
            return
        try:
            token = resolve_onemap_token()
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        records = cache.pending(
            category=args.category,
            retry_unresolved=args.retry_unresolved,
            retry_errors=args.retry_errors,
            limit=args.limit,
        )
        client = OneMapClient(token, request_delay=args.request_delay)
        counts = process_records(cache, records, client)
        cache.export_csv()
        print(json.dumps(counts, indent=2))
    finally:
        cache.export_csv()
        cache.close()


if __name__ == "__main__":
    main()
