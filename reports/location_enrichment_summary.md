# Location and MRT enrichment summary

## Official MRT reference

- Source: **LTA MRT Station Exit (GEOJSON)** from data.gov.sg (`d_b39d3a0871985372d7e1637193335da5`).
- Dataset page: https://data.gov.sg/datasets/d_b39d3a0871985372d7e1637193335da5/view
- Retrieved: `2026-09-24T07:04:04.358825+00:00`; licence: Singapore Open Data Licence (see dataset page).
- Raw layer: **613 rail exits** across **190 station names**. The official layer includes LRT.
- Model reference: **524 physical MRT exits** across **142 MRT station names**. LRT remains preserved in the raw GeoJSON but is excluded from nearest-MRT calculations.

## Reproduction

```bash
.venv/bin/python -m src.location.download_mrt_data
.venv/bin/python -m src.location.geocode_properties --init-only
# Configure one of these credential options:
export ONEMAP_TOKEN='your-current-token'
# Or: export ONEMAP_EMAIL='you@example.com' ONEMAP_PASSWORD='your-password'
.venv/bin/python -m src.location.geocode_properties
.venv/bin/python -m src.enrich_location_data
```

OneMap Search requires authentication. Use `ONEMAP_TOKEN`, or `ONEMAP_EMAIL` plus `ONEMAP_PASSWORD`; secrets are never written to the cache. `.env` is gitignored, and `.env.example` documents variable names. The client defaults to 0.25 seconds between requests (below the documented 300 calls/minute limit), bounded retries, exponential backoff, timeouts, and committed SQLite updates after each attempted unique location.

## Strategy and cache

- Existing canonical `location_key` values are reused: HDB `BLOCK|STREET`; EC/landed `PROJECT|STREET|DISTRICT`.
- HDB queries block+street, then street only. A block/street evidence match is `exact_building`; street fallback remains `street` precision.
- EC queries project+street, project, then street. Landed uses the same bounded strategy but skips generic project-only searches for `LANDED HOUSING DEVELOPMENT`.
- Results are ranked using block/road/building evidence and Singapore coordinate bounds; the first API result is not accepted blindly.
- SQLite is the durable per-result cache; `property_geocoding_cache.csv` is its auditable snapshot. Successful rows are never re-requested. Explicit flags control retries for unresolved/error rows.

## Coverage

| Category | Unique locations | API success | Failed/unresolved/pending | ML-eligible unique | Transaction coordinate coverage | ML-eligible transaction coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HDB | 9,947 | 9,649 (97.00%) | 298 | 9,542/9,947 (95.93%) | 944,274/971,523 (97.20%) | 933,678/971,523 (96.10%) |
| EC | 82 | 82 (100.00%) | 0 | 81/82 (98.78%) | 16,495/16,495 (100.00%) | 16,285/16,495 (98.73%) |
| LANDED | 2,047 | 2,042 (99.76%) | 5 | 0/2,047 (0.00%) | 8,724/8,740 (99.82%) | 0/8,740 (0.00%) |

Validated match-quality/status detail:

- **HDB:** status [success: 9,649, unresolved: 216, error: 82]; original quality [exact_building: 9,542 (95.93%), unresolved: 298 (3.00%), street: 107 (1.08%)]; validated quality [exact_building: 9,542 (95.93%), unresolved: 298 (3.00%), street: 107 (1.08%)]; suspicious successful matches: 0 locations / 0 transactions.
- **EC:** status [success: 82]; original quality [street: 80 (97.56%), project: 2 (2.44%)]; validated quality [project: 81 (98.78%), street: 1 (1.22%)]; suspicious successful matches: 0 locations / 0 transactions.
- **LANDED:** status [success: 2,042, unresolved: 5]; original quality [street: 2,012 (98.29%), project: 30 (1.47%), unresolved: 5 (0.24%)]; validated quality [street: 2,012 (98.29%), unresolved: 21 (1.03%), project: 14 (0.68%)]; suspicious successful matches: 16 locations / 41 transactions.

Nearest-MRT transaction-distance distribution:

- **HDB:** min 15 m; median 604 m; mean 668 m; P90 1170 m; P95 1393 m; max 5433 m.
- **EC:** min 238 m; median 934 m; mean 1063 m; P90 1861 m; P95 2175 m; max 2586 m.
- **LANDED:** not applicable because no matches are ML-eligible under the final category policy.

The distributions above use only ML-eligible coordinates, not all API successes.

Successful cached matches that fail deterministic address validation remain auditable but are downgraded to `unresolved` and excluded from ML-eligible MRT coverage.

## Feature decisions

- **HDB: include as a candidate.** Building-level evidence covers the large majority of unique locations and transactions; street fallbacks remain excluded.
- **EC: include as a candidate.** Cached evidence validation requires the exact normalized project name in the OneMap building/address plus a consistent street. The original classifier under-labeled many of these as street matches; validation corrects only those with both signals.
- **Landed: exclude from candidate features.** Only 14/2,047 unique locations and 65/8,740 transactions (0.74%) have validated project-level evidence. The conservative policy retains street matches as approximate and rejects weak project/park results.
- Missing MRT distance is retained as null—never zero or an arbitrary constant. HDB/EC include leakage-safe `mrt_distance_missing`; later imputation must be fit on training data only, or use a model with native missing handling.
- Latitude/longitude and nearest-station/exit names remain reference/display-only.

## Spot checks

Deterministic short/medium/long/fallback/suspicious samples are stored in `reports/location_spot_checks.csv`; machine-readable results are in `reports/location_validation.json` and `data/processed/location_enrichment_validation.json`. They include the original key/query, matched address, original and validated quality, coordinates, nearest exit, distance, and suspicion reason.

## Validation and limitations

- Enriched row counts, targets, and split assignments are validated against canonical inputs; joins are many-to-one and cannot expand transaction counts.
- Singapore coordinate bounds, nonnegative distances, and a 60 km implausibility flag are checked programmatically.
- Unresolved/pending transactions remain present with null coordinates and MRT distance.
- Coordinates are reference fields, not automatic predictors. `nearest_mrt_distance_m` eligibility is category-specific: HDB exact-building and EC validated-project only; landed is excluded.
- Street-level landed coordinates represent a street search result, not an individual house. Their distances are kept only in `nearest_mrt_distance_reference_m`; the ML candidate `nearest_mrt_distance_m` is null for landed.
- The official layer contains physical exits but does not provide rail line codes; interchange naming is preserved from `STATION_NA`.
- MRT-only exits are used (not LRT), and the nearest physical exit is selected locally with vectorized Haversine distance.
