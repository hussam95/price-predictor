"""Download and normalize the official LTA MRT Station Exit dataset.

Run:
    python -m src.location.download_mrt_data

The original GeoJSON response is retained under data/raw/reference. The
normalized CSV contains MRT (not LRT) exits because the model feature is
specifically distance to MRT. Raw LRT features remain in the GeoJSON.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_OUTPUT = ROOT / "data" / "raw" / "reference" / "mrt_station_exits.geojson"
PROCESSED_OUTPUT = ROOT / "data" / "processed" / "mrt_stations.csv"
METADATA_OUTPUT = ROOT / "data" / "processed" / "mrt_stations_metadata.json"

DATASET_ID = "d_b39d3a0871985372d7e1637193335da5"
DATASET_PAGE = f"https://data.gov.sg/datasets/{DATASET_ID}/view"
POLL_URL = f"https://api-open.data.gov.sg/v1/public/api/datasets/{DATASET_ID}/poll-download"
INITIATE_URL = f"https://api-open.data.gov.sg/v1/public/api/datasets/{DATASET_ID}/initiate-download"
USER_AGENT = "singapore-property-predictor/1.0 (local data preparation)"


def _request_json(url: str, *, method: str = "GET", timeout: int = 30) -> dict[str, Any]:
    request = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def resolve_download_url(max_polls: int = 12) -> str:
    """Resolve data.gov.sg's temporary download URL, initiating when necessary."""
    for attempt in range(max_polls):
        payload = _request_json(POLL_URL)
        if payload.get("code") == 0 and payload.get("data", {}).get("url"):
            return str(payload["data"]["url"])
        if attempt == 0:
            _request_json(INITIATE_URL, method="GET")
        time.sleep(min(2**attempt, 10))
    raise RuntimeError("data.gov.sg did not provide a download URL after bounded polling")


def download_geojson(url: str, timeout: int = 60) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    data = json.loads(payload.decode("utf-8"))
    if data.get("type") != "FeatureCollection" or not isinstance(data.get("features"), list):
        raise ValueError("Downloaded MRT payload is not a GeoJSON FeatureCollection")
    return data


def normalize_mrt_exits(geojson: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in geojson["features"]:
        geometry = feature.get("geometry") or {}
        properties = feature.get("properties") or {}
        coordinates = geometry.get("coordinates") or []
        if geometry.get("type") != "Point" or len(coordinates) < 2:
            continue
        raw_name = str(properties.get("STATION_NA") or "").strip()
        # The official layer also contains LRT exits. Preserve those in raw GeoJSON,
        # but use only named MRT stations for the requested nearest-MRT feature.
        if "MRT STATION" not in raw_name.upper() or "LRT STATION" in raw_name.upper():
            continue
        station_name = raw_name.upper().removesuffix(" MRT STATION").strip()
        rows.append({
            "source_object_id": properties.get("OBJECTID"),
            "station_name": station_name,
            "official_station_name": raw_name,
            "exit_code": str(properties.get("EXIT_CODE") or "").strip(),
            "latitude": float(coordinates[1]),
            "longitude": float(coordinates[0]),
            "source_updated_at": str(properties.get("FMEL_UPD_D") or ""),
            "source": "LTA MRT Station Exit (GEOJSON), data.gov.sg",
        })
    result = pd.DataFrame(rows).sort_values(
        ["station_name", "exit_code", "source_object_id"], kind="stable"
    ).reset_index(drop=True)
    if result.empty:
        raise ValueError("No MRT exits were found in the official dataset")
    if not result["latitude"].between(1.15, 1.50).all() or not result["longitude"].between(103.55, 104.15).all():
        raise ValueError("Official MRT coordinates failed Singapore bounds validation")
    return result


def main() -> None:
    try:
        download_url = resolve_download_url()
        geojson = download_geojson(download_url)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(f"Could not download official MRT data: {exc}") from exc

    exits = normalize_mrt_exits(geojson)
    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUT.write_text(json.dumps(geojson, ensure_ascii=False) + "\n", encoding="utf-8")
    exits.to_csv(PROCESSED_OUTPUT, index=False)

    raw_names = [str((f.get("properties") or {}).get("STATION_NA") or "") for f in geojson["features"]]
    metadata = {
        "dataset_id": DATASET_ID,
        "dataset_page": DATASET_PAGE,
        "poll_endpoint": POLL_URL,
        "download_url_retained": False,
        "download_note": "data.gov.sg temporary signed download URLs are intentionally not persisted",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "publisher": "Land Transport Authority (LTA), Singapore",
        "licence": "Singapore Open Data Licence (see dataset page)",
        "coordinate_type": "Physical station entrance/exit points in WGS84",
        "raw_feature_count": len(geojson["features"]),
        "raw_unique_station_names": len(set(raw_names)),
        "processed_scope": "MRT exits only; official LRT features remain in raw GeoJSON",
        "processed_mrt_exit_count": len(exits),
        "processed_unique_mrt_station_count": int(exits["station_name"].nunique()),
    }
    METADATA_OUTPUT.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"Downloaded {metadata['raw_feature_count']} official rail exits; "
        f"normalized {len(exits)} MRT exits across {exits['station_name'].nunique()} MRT stations."
    )


if __name__ == "__main__":
    main()
