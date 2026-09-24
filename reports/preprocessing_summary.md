# Canonical preprocessing summary

## Outcome

The pipeline combines and canonicalizes the audited HDB, EC, and landed transactions without modifying raw files or dropping rows. Private residential, rental, MRT/geocoding, modeling, and application work remain out of scope.

Run with `.venv/bin/python -m src.preprocess_data`. Run focused tests with `.venv/bin/python -m unittest discover -s tests -v`.

| Dataset | Processed rows | Columns | Rows dropped |
| --- | ---: | ---: | ---: |
| HDB | 971,523 | 30 | 0 |
| EC | 16,495 | 37 | 0 |
| LANDED | 8,740 | 38 | 0 |

## Canonical schemas

### HDB

`source_file` (string), `source_row_number` (Int64), `transaction_date` (datetime64[us]), `transaction_year` (Int64), `transaction_month` (Int64), `transaction_quarter` (Int64), `town` (string), `flat_type` (string), `block` (string), `street_name` (string), `storey_range` (string), `storey_lower` (Float64), `storey_upper` (Float64), `storey_midpoint` (Float64), `floor_area_sqm` (Float64), `flat_model` (string), `lease_commence_year` (Int64), `property_age_at_transaction` (Float64), `approx_remaining_lease_years` (Float64), `source_remaining_lease` (string), `source_remaining_lease_months` (Float64), `location_key` (string), `resale_price` (Float64), `quality_invalid_transaction_date` (bool), `quality_invalid_target` (boolean), `quality_invalid_area` (boolean), `quality_negative_property_age` (boolean), `quality_unparsed_storey` (bool), `quality_any` (boolean), `data_split` (string)

### EC

`source_file` (string), `source_row_number` (Int64), `transaction_date` (datetime64[us]), `transaction_year` (Int64), `transaction_month` (Int64), `transaction_quarter` (Int64), `project_name` (string), `street_name` (string), `sale_type` (string), `area_type` (string), `area_sqm` (Float64), `property_type` (string), `number_of_units` (Int64), `tenure` (string), `tenure_category` (string), `lease_duration_years` (Float64), `lease_commence_year` (Float64), `postal_district` (Int64), `market_segment` (string), `floor_level` (string), `floor_lower` (Float64), `floor_upper` (Float64), `floor_midpoint` (Float64), `property_age_at_transaction` (Float64), `approx_remaining_lease_years` (Float64), `quality_lease_starts_after_transaction` (bool), `location_key` (string), `transaction_price` (Float64), `is_multi_unit_transaction` (boolean), `quality_invalid_transaction_date` (bool), `quality_invalid_target` (boolean), `quality_invalid_area` (boolean), `quality_unparsed_tenure` (boolean), `quality_leasehold_missing_commencement` (boolean), `quality_unparsed_floor` (bool), `quality_any` (boolean), `data_split` (string)

### LANDED

`source_file` (string), `source_row_number` (Int64), `transaction_date` (datetime64[us]), `transaction_year` (Int64), `transaction_month` (Int64), `transaction_quarter` (Int64), `project_name` (string), `street_name` (string), `sale_type` (string), `area_type` (string), `area_sqm` (Float64), `property_type` (string), `number_of_units` (Int64), `tenure` (string), `tenure_category` (string), `lease_duration_years` (Float64), `lease_commence_year` (Float64), `postal_district` (Int64), `market_segment` (string), `floor_level` (string), `floor_lower` (Float64), `floor_upper` (Float64), `floor_midpoint` (Float64), `property_age_at_transaction` (Float64), `approx_remaining_lease_years` (Float64), `quality_lease_starts_after_transaction` (bool), `location_key` (string), `transaction_price` (Float64), `is_multi_unit_transaction` (boolean), `quality_invalid_transaction_date` (bool), `quality_invalid_target` (boolean), `quality_invalid_area` (boolean), `quality_unparsed_tenure` (boolean), `quality_leasehold_missing_commencement` (boolean), `quality_unparsed_floor` (bool), `quality_any` (boolean), `data_split` (string), `diagnostic_price_outlier_iqr` (boolean)

## Important transformations and assumptions

- All transaction dates are month timestamps at the first day of the source month, with year/month/quarter columns.
- Text is trimmed and internal whitespace collapsed. HDB categories/addresses are uppercased; the audited `MULTI-GENERATION`/`MULTI GENERATION` flat-type spelling is unified to `MULTI GENERATION`. Semantically distinct labels are otherwise preserved.
- HDB storey bands and URA floor bands are parsed into lower/upper/midpoint values.
- HDB `property_age_at_transaction` is transaction year minus lease commencement year. `approx_remaining_lease_years = 99 - property age` is an explicitly approximate year-granularity feature. Source-provided remaining lease is retained separately and historical gaps are not filled.
- URA tenure is parsed into freehold/leasehold, lease duration, and commencement year. Freehold has no finite duration or remaining-lease value. Lease age/remaining lease are null where commencement is unknown or after the transaction month/year.
- Landed multi-unit rows and price extremes remain present. The price IQR flag/fence are target-derived EDA diagnostics and prohibited from model inputs.
- `source_file` and one-based CSV `source_row_number` (header is row 1) preserve provenance.

## Quality and diagnostic flags

- **HDB:** `quality_invalid_transaction_date`=0, `quality_invalid_target`=0, `quality_invalid_area`=0, `quality_negative_property_age`=51, `quality_unparsed_storey`=0, `quality_any`=51
- **EC:** `quality_lease_starts_after_transaction`=0, `is_multi_unit_transaction`=0, `quality_invalid_transaction_date`=0, `quality_invalid_target`=0, `quality_invalid_area`=0, `quality_unparsed_tenure`=0, `quality_leasehold_missing_commencement`=7, `quality_unparsed_floor`=0, `quality_any`=7
- **LANDED:** `quality_lease_starts_after_transaction`=0, `is_multi_unit_transaction`=18, `quality_invalid_transaction_date`=0, `quality_invalid_target`=0, `quality_invalid_area`=0, `quality_unparsed_tenure`=0, `quality_leasehold_missing_commencement`=2, `quality_unparsed_floor`=0, `quality_any`=2, `diagnostic_price_outlier_iqr`=748

The landed 1.5×IQR upper fence is **$10,458,000**; **748** rows are flagged. **18** landed rows contain more than one unit. No rows are deleted.

The 51 HDB rows whose commencement year follows the transaction year are retained and flagged; their derived property-age/remaining-lease fields are null. No EC or landed row has this condition.

## Leakage protection

- Canonical EC/landed CSVs omit raw `Unit Price ($ PSF)`, `Unit Price ($ PSM)`, and `Nett Price($)` entirely.
- No price-per-area or same-period aggregate target feature is created for any category.
- Landed target-derived IQR diagnostics are listed as EDA-only and leakage-excluded, never as candidates.
- `feature_manifest.json` is the allowlist for later modeling. Validation fails if targets or recognized price-derived fields enter candidate lists.

## Chronological split plan

| Dataset | Train | Validation | Test |
| --- | --- | --- | --- |
| HDB | through 2022-12 (889,599) | 2023-01–2024-12 (53,586) | 2025-01–2026-02 (28,338) |
| EC | through 2023-12 (7,418) | 2024-01–2024-12 (3,245) | 2025-01–2026-09 (5,832) |
| LANDED | through 2023-12 (3,838) | 2024-01–2024-12 (1,695) | 2025-01–2026-09 (3,207) |

These boundaries reproduce the audit recommendations. The `data_split` column records membership without duplicating datasets.

## MRT/location readiness

- HDB: normalized `block`, `street_name`, and `location_key = BLOCK|STREET`.
- EC/landed: normalized `project_name`, `street_name`, `postal_district`, and `location_key = PROJECT|STREET|DISTRICT`.
- No coordinates, MRT data, API calls, or geocoding are included.

## Unresolved issues

- Identical source rows remain because transaction IDs/unit numbers are unavailable.
- HDB historical remaining lease remains unavailable; the arithmetic proxy is not exact.
- URA generic leasehold strings lack commencement years (seven EC rows and two landed rows), so lease-age features remain null and the rows are flagged.
- Landed generic project labels, multi-unit transactions, and extreme luxury transactions need explicit modeling policy later.
