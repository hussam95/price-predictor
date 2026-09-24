# Singapore property data audit

> Scope: local, read-only inspection. No raw data was changed; no models were trained; no geocoding or downloads were performed.

## Executive findings

- HDB is the strongest dataset: **971,523 rows**, continuous monthly coverage **1990-01 to 2026-02**, and no null location/target fields. The five files are safely harmonizable after adding nullable `remaining_lease` and normalizing historical labels.
- EC is usable as a dedicated category: **16,495 rows**, **2021-09 to 2026-09**, identical schemas, and no exact cross-part duplicates. The parts overlap only in `23-Aug`, but contain different rows.
- Landed has **8,740 rows** and broad district/street coverage. It is plausibly modelable, but portfolio/multi-unit transactions and extreme luxury sales need explicit handling/segmentation.
- Private is **not viable for a general condo/apartment model**: only **117 rows**, **5 projects**, and `CASABLANCA` alone contributes **87 rows (74.4%)**.
- URA `Unit Price ($ PSF)`, `Unit Price ($ PSM)`, and any populated `Nett Price($)` are target leakage for total-price prediction and must not be model inputs.
- No supplied file contains rental transactions. The DOCX only requests them as a future input.

## Dataset inventory

| File | Type | Rows | Columns | Date coverage |
| --- | --- | --- | --- | --- |
| `data/raw/hdb/Resale Flat Prices (Based on Approval Date), 1990 - 1999.csv` | CSV | 287,196 | 10 | 1990-01–1999-12 |
| `data/raw/hdb/Resale Flat Prices (Based on Approval Date), 2000 - Feb 2012.csv` | CSV | 369,651 | 10 | 2000-01–2012-02 |
| `data/raw/hdb/Resale Flat Prices (Based on Registration Date), From Jan 2015 to Dec 2016.csv` | CSV | 37,153 | 11 | 2015-01–2016-12 |
| `data/raw/hdb/Resale Flat Prices (Based on Registration Date), From Mar 2012 to Dec 2014.csv` | CSV | 52,203 | 10 | 2012-03–2014-12 |
| `data/raw/hdb/Resale flat prices based on registration date from Jan-2017 onwards.csv` | CSV | 225,320 | 11 | 2017-01–2026-02 |
| `data/raw/ura/ec/URA Executive Condominium Data (Part 1).csv` | CSV | 10,000 | 17 | 2023-08–2026-09 |
| `data/raw/ura/ec/URA Executive Condominium Data (Part 2).csv` | CSV | 6,495 | 17 | 2021-09–2023-08 |
| `data/raw/ura/landed/URA Private Property Transactions (By District) for Landed Properties.csv` | CSV | 8,740 | 17 | 2021-09–2026-09 |
| `data/raw/ura/private/URA Private Residential Property Transactions (1).csv` | CSV | 117 | 17 | 2021-09–2026-08 |
| `data/raw/reference/HDB Street Names.xlsx` | XLSX | 740 | 1 | n/a |
| `data/raw/reference/rpi-table.pdf` | PDF (5 pages) | n/a | n/a | 1990-Q1–2026-Q2 |
| `data/raw/reference/Median resale prices for registered resale applications.pdf` | PDF (77 pages) | n/a | n/a | 2007-Q2–2026-Q2 |
| `docs/Important Project Info.docx` | DOCX | n/a | n/a | n/a |

## Per-file transaction statistics

Exact duplicate rows below are byte-equivalent parsed records, not proven duplicate sales: the sources contain no transaction/unit identifier, so multiple real sales can be indistinguishable.

| Dataset | Rows | Exact dups | Dates | Min | P25 | Median | Mean | P75 | P95 | P99 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Resale Flat Prices (Based on Approval Date), 1990 - 1999.csv | 287,196 | 826 | 1990-01–1999-12 | $5,000 | $127,000 | $195,000 | $219,542 | $298,000 | $465,000 | $575,000 | $900,000 |
| Resale Flat Prices (Based on Approval Date), 2000 - Feb 2012.csv | 369,651 | 513 | 2000-01–2012-02 | $28,000 | $195,000 | $263,000 | $281,272 | $350,000 | $490,000 | $610,000 | $903,000 |
| Resale Flat Prices (Based on Registration Date), From Jan 2015 to Dec 2016.csv | 37,153 | 24 | 2015-01–2016-12 | $190,000 | $340,000 | $408,000 | $436,863 | $495,000 | $718,000 | $880,000 | $1,150,000 |
| Resale Flat Prices (Based on Registration Date), From Mar 2012 to Dec 2014.csv | 52,203 | 248 | 2012-03–2014-12 | $195,000 | $370,000 | $440,000 | $461,215 | $525,000 | $702,879 | $830,000 | $1,088,888 |
| Resale flat prices based on registration date from Jan-2017 onwards.csv | 225,320 | 311 | 2017-01–2026-02 | $140,000 | $388,000 | $495,000 | $526,368 | $630,000 | $880,000 | $1,080,000 | $1,700,000 |
| URA Executive Condominium Data (Part 1).csv | 10,000 | 906 | 2023-08–2026-09 | $710,000 | $1,380,000 | $1,550,000 | $1,594,311 | $1,780,000 | $2,198,804 | $2,500,000 | $3,700,000 |
| URA Executive Condominium Data (Part 2).csv | 6,495 | 522 | 2021-09–2023-08 | $530,000 | $1,160,000 | $1,300,000 | $1,330,919 | $1,461,000 | $1,848,000 | $2,206,180 | $3,288,000 |
| URA Private Property Transactions (By District) for Landed Properties.csv | 8,740 | 10 | 2021-09–2026-09 | $320,000 | $3,628,000 | $4,608,280 | $6,006,217 | $6,360,000 | $13,680,000 | $26,500,000 | $815,000,000 |
| URA Private Residential Property Transactions (1).csv | 117 | 0 | 2021-09–2026-08 | $762,000 | $1,010,000 | $1,180,000 | $2,780,065 | $2,000,000 | $9,723,600 | $10,284,000 | $10,800,000 |

Mechanical 1.5×IQR flags (diagnostic only; not deletion rules):

| Dataset | Lower fence | Below | Upper fence | Above |
| --- | --- | --- | --- | --- |
| Resale Flat Prices (Based on Approval Date), 1990 - 1999.csv | $-129,500 | 0 | $554,500 | 3,906 |
| Resale Flat Prices (Based on Approval Date), 2000 - Feb 2012.csv | $-37,500 | 0 | $582,500 | 5,432 |
| Resale Flat Prices (Based on Registration Date), From Jan 2015 to Dec 2016.csv | $107,500 | 0 | $727,500 | 1,720 |
| Resale Flat Prices (Based on Registration Date), From Mar 2012 to Dec 2014.csv | $137,500 | 0 | $757,500 | 1,407 |
| Resale flat prices based on registration date from Jan-2017 onwards.csv | $25,000 | 0 | $993,000 | 4,344 |
| URA Executive Condominium Data (Part 1).csv | $780,000 | 13 | $2,380,000 | 164 |
| URA Executive Condominium Data (Part 2).csv | $708,500 | 26 | $1,912,500 | 239 |
| URA Private Property Transactions (By District) for Landed Properties.csv | $-470,000 | 0 | $10,458,000 | 748 |
| URA Private Residential Property Transactions (1).csv | $-475,000 | 0 | $3,485,000 | 27 |

### Columns, inferred types, and missingness

#### `data/raw/hdb/Resale Flat Prices (Based on Approval Date), 1990 - 1999.csv`

Columns (exact order): `month, town, flat_type, block, street_name, storey_range, floor_area_sqm, flat_model, lease_commence_date, resale_price`

| Column | pandas dtype | Inferred meaning | Nulls | `-` placeholders | Unique |
| --- | --- | --- | --- | --- | --- |
| `month` | str | year-month | 0 (0.000%) | 0 | 120 |
| `town` | str | categorical/text | 0 (0.000%) | 0 | 26 |
| `flat_type` | str | categorical/text | 0 (0.000%) | 0 | 7 |
| `block` | str | identifier/text | 0 (0.000%) | 0 | 1,094 |
| `street_name` | str | identifier/text | 0 (0.000%) | 0 | 417 |
| `storey_range` | str | categorical/text | 0 (0.000%) | 0 | 9 |
| `floor_area_sqm` | float64 | numeric | 0 (0.000%) | 0 | 199 |
| `flat_model` | str | categorical/text | 0 (0.000%) | 0 | 13 |
| `lease_commence_date` | int64 | integer-coded numeric | 0 (0.000%) | 0 | 31 |
| `resale_price` | int64 | numeric | 0 (0.000%) | 0 | 3,782 |

#### `data/raw/hdb/Resale Flat Prices (Based on Approval Date), 2000 - Feb 2012.csv`

Columns (exact order): `month, town, flat_type, block, street_name, storey_range, floor_area_sqm, flat_model, lease_commence_date, resale_price`

| Column | pandas dtype | Inferred meaning | Nulls | `-` placeholders | Unique |
| --- | --- | --- | --- | --- | --- |
| `month` | str | year-month | 0 (0.000%) | 0 | 146 |
| `town` | str | categorical/text | 0 (0.000%) | 0 | 26 |
| `flat_type` | str | categorical/text | 0 (0.000%) | 0 | 7 |
| `block` | str | identifier/text | 0 (0.000%) | 0 | 1,991 |
| `street_name` | str | identifier/text | 0 (0.000%) | 0 | 523 |
| `storey_range` | str | categorical/text | 0 (0.000%) | 0 | 14 |
| `floor_area_sqm` | float64 | numeric | 0 (0.000%) | 0 | 189 |
| `flat_model` | str | categorical/text | 0 (0.000%) | 0 | 16 |
| `lease_commence_date` | int64 | integer-coded numeric | 0 (0.000%) | 0 | 46 |
| `resale_price` | float64 | numeric | 0 (0.000%) | 0 | 5,615 |

#### `data/raw/hdb/Resale Flat Prices (Based on Registration Date), From Jan 2015 to Dec 2016.csv`

Columns (exact order): `month, town, flat_type, block, street_name, storey_range, floor_area_sqm, flat_model, lease_commence_date, remaining_lease, resale_price`

| Column | pandas dtype | Inferred meaning | Nulls | `-` placeholders | Unique |
| --- | --- | --- | --- | --- | --- |
| `month` | str | year-month | 0 (0.000%) | 0 | 24 |
| `town` | str | categorical/text | 0 (0.000%) | 0 | 26 |
| `flat_type` | str | categorical/text | 0 (0.000%) | 0 | 7 |
| `block` | str | identifier/text | 0 (0.000%) | 0 | 2,108 |
| `street_name` | str | identifier/text | 0 (0.000%) | 0 | 519 |
| `storey_range` | str | categorical/text | 0 (0.000%) | 0 | 17 |
| `floor_area_sqm` | float64 | numeric | 0 (0.000%) | 0 | 156 |
| `flat_model` | str | categorical/text | 0 (0.000%) | 0 | 20 |
| `lease_commence_date` | int64 | integer-coded numeric | 0 (0.000%) | 0 | 48 |
| `remaining_lease` | int64 | numeric | 0 (0.000%) | 0 | 50 |
| `resale_price` | float64 | numeric | 0 (0.000%) | 0 | 1,748 |

#### `data/raw/hdb/Resale Flat Prices (Based on Registration Date), From Mar 2012 to Dec 2014.csv`

Columns (exact order): `month, town, flat_type, block, street_name, storey_range, floor_area_sqm, flat_model, lease_commence_date, resale_price`

| Column | pandas dtype | Inferred meaning | Nulls | `-` placeholders | Unique |
| --- | --- | --- | --- | --- | --- |
| `month` | str | year-month | 0 (0.000%) | 0 | 34 |
| `town` | str | categorical/text | 0 (0.000%) | 0 | 26 |
| `flat_type` | str | categorical/text | 0 (0.000%) | 0 | 7 |
| `block` | str | identifier/text | 0 (0.000%) | 0 | 2,047 |
| `street_name` | str | identifier/text | 0 (0.000%) | 0 | 515 |
| `storey_range` | str | categorical/text | 0 (0.000%) | 0 | 22 |
| `floor_area_sqm` | float64 | numeric | 0 (0.000%) | 0 | 163 |
| `flat_model` | str | categorical/text | 0 (0.000%) | 0 | 17 |
| `lease_commence_date` | int64 | integer-coded numeric | 0 (0.000%) | 0 | 47 |
| `resale_price` | float64 | numeric | 0 (0.000%) | 0 | 2,067 |

#### `data/raw/hdb/Resale flat prices based on registration date from Jan-2017 onwards.csv`

Columns (exact order): `month, town, flat_type, block, street_name, storey_range, floor_area_sqm, flat_model, lease_commence_date, remaining_lease, resale_price`

| Column | pandas dtype | Inferred meaning | Nulls | `-` placeholders | Unique |
| --- | --- | --- | --- | --- | --- |
| `month` | str | year-month | 0 (0.000%) | 0 | 110 |
| `town` | str | categorical/text | 0 (0.000%) | 0 | 26 |
| `flat_type` | str | categorical/text | 0 (0.000%) | 0 | 7 |
| `block` | str | identifier/text | 0 (0.000%) | 0 | 2,750 |
| `street_name` | str | identifier/text | 0 (0.000%) | 0 | 577 |
| `storey_range` | str | categorical/text | 0 (0.000%) | 0 | 17 |
| `floor_area_sqm` | float64 | numeric | 0 (0.000%) | 0 | 187 |
| `flat_model` | str | categorical/text | 0 (0.000%) | 0 | 21 |
| `lease_commence_date` | int64 | integer-coded numeric | 0 (0.000%) | 0 | 56 |
| `remaining_lease` | str | categorical/text | 0 (0.000%) | 0 | 696 |
| `resale_price` | float64 | numeric | 0 (0.000%) | 0 | 4,595 |

#### `data/raw/ura/ec/URA Executive Condominium Data (Part 1).csv`

Columns (exact order): `Project Name, Transacted Price ($), Area (SQFT), Unit Price ($ PSF), Sale Date, Street Name, Type of Sale, Type of Area, Area (SQM), Unit Price ($ PSM), Nett Price($), Property Type, Number of Units, Tenure, Postal District, Market Segment, Floor Level`

| Column | pandas dtype | Inferred meaning | Nulls | `-` placeholders | Unique |
| --- | --- | --- | --- | --- | --- |
| `Project Name` | str | identifier/text | 0 (0.000%) | 0 | 82 |
| `Transacted Price ($)` | str | numeric | 0 (0.000%) | 0 | 1,724 |
| `Area (SQFT)` | str | numeric | 0 (0.000%) | 0 | 155 |
| `Unit Price ($ PSF)` | str | numeric | 0 (0.000%) | 0 | 1,167 |
| `Sale Date` | str | year-month | 0 (0.000%) | 0 | 38 |
| `Street Name` | str | identifier/text | 0 (0.000%) | 0 | 64 |
| `Type of Sale` | str | categorical/text | 0 (0.000%) | 0 | 2 |
| `Type of Area` | str | categorical/text | 0 (0.000%) | 0 | 1 |
| `Area (SQM)` | int64 | numeric | 0 (0.000%) | 0 | 155 |
| `Unit Price ($ PSM)` | str | numeric | 0 (0.000%) | 0 | 4,878 |
| `Nett Price($)` | str | categorical/text | 0 (0.000%) | 10000 | 1 |
| `Property Type` | str | categorical/text | 0 (0.000%) | 0 | 1 |
| `Number of Units` | int64 | integer-coded numeric | 0 (0.000%) | 0 | 1 |
| `Tenure` | str | categorical/text | 0 (0.000%) | 0 | 23 |
| `Postal District` | int64 | integer-coded numeric | 0 (0.000%) | 0 | 10 |
| `Market Segment` | str | categorical/text | 0 (0.000%) | 0 | 2 |
| `Floor Level` | str | categorical/text | 0 (0.000%) | 0 | 7 |

#### `data/raw/ura/ec/URA Executive Condominium Data (Part 2).csv`

Columns (exact order): `Project Name, Transacted Price ($), Area (SQFT), Unit Price ($ PSF), Sale Date, Street Name, Type of Sale, Type of Area, Area (SQM), Unit Price ($ PSM), Nett Price($), Property Type, Number of Units, Tenure, Postal District, Market Segment, Floor Level`

| Column | pandas dtype | Inferred meaning | Nulls | `-` placeholders | Unique |
| --- | --- | --- | --- | --- | --- |
| `Project Name` | str | identifier/text | 0 (0.000%) | 0 | 76 |
| `Transacted Price ($)` | str | numeric | 0 (0.000%) | 0 | 1,352 |
| `Area (SQFT)` | str | numeric | 0 (0.000%) | 0 | 152 |
| `Unit Price ($ PSF)` | str | numeric | 0 (0.000%) | 0 | 819 |
| `Sale Date` | str | year-month | 0 (0.000%) | 0 | 24 |
| `Street Name` | str | identifier/text | 0 (0.000%) | 0 | 60 |
| `Type of Sale` | str | categorical/text | 0 (0.000%) | 0 | 2 |
| `Type of Area` | str | categorical/text | 0 (0.000%) | 0 | 1 |
| `Area (SQM)` | int64 | numeric | 0 (0.000%) | 0 | 152 |
| `Unit Price ($ PSM)` | str | numeric | 0 (0.000%) | 0 | 3,356 |
| `Nett Price($)` | str | categorical/text | 0 (0.000%) | 6495 | 1 |
| `Property Type` | str | categorical/text | 0 (0.000%) | 0 | 1 |
| `Number of Units` | int64 | integer-coded numeric | 0 (0.000%) | 0 | 1 |
| `Tenure` | str | categorical/text | 0 (0.000%) | 0 | 19 |
| `Postal District` | int64 | integer-coded numeric | 0 (0.000%) | 0 | 9 |
| `Market Segment` | str | categorical/text | 0 (0.000%) | 0 | 2 |
| `Floor Level` | str | categorical/text | 0 (0.000%) | 0 | 6 |

#### `data/raw/ura/landed/URA Private Property Transactions (By District) for Landed Properties.csv`

Columns (exact order): `Project Name, Transacted Price ($), Area (SQFT), Unit Price ($ PSF), Sale Date, Street Name, Type of Sale, Type of Area, Area (SQM), Unit Price ($ PSM), Nett Price($), Property Type, Number of Units, Tenure, Postal District, Market Segment, Floor Level`

| Column | pandas dtype | Inferred meaning | Nulls | `-` placeholders | Unique |
| --- | --- | --- | --- | --- | --- |
| `Project Name` | str | identifier/text | 0 (0.000%) | 0 | 478 |
| `Transacted Price ($)` | str | numeric | 0 (0.000%) | 0 | 2,074 |
| `Area (SQFT)` | str | numeric | 0 (0.000%) | 0 | 3,497 |
| `Unit Price ($ PSF)` | str | numeric | 0 (0.000%) | 0 | 2,566 |
| `Sale Date` | str | year-month | 0 (0.000%) | 0 | 61 |
| `Street Name` | str | identifier/text | 0 (0.000%) | 0 | 1,493 |
| `Type of Sale` | str | categorical/text | 0 (0.000%) | 0 | 3 |
| `Type of Area` | str | categorical/text | 0 (0.000%) | 0 | 1 |
| `Area (SQM)` | str | numeric | 0 (0.000%) | 0 | 3,497 |
| `Unit Price ($ PSM)` | str | numeric | 0 (0.000%) | 0 | 7,054 |
| `Nett Price($)` | str | categorical/text | 0 (0.000%) | 8740 | 1 |
| `Property Type` | str | categorical/text | 0 (0.000%) | 0 | 3 |
| `Number of Units` | int64 | integer-coded numeric | 0 (0.000%) | 0 | 5 |
| `Tenure` | str | categorical/text | 0 (0.000%) | 0 | 65 |
| `Postal District` | int64 | integer-coded numeric | 0 (0.000%) | 0 | 24 |
| `Market Segment` | str | categorical/text | 0 (0.000%) | 0 | 3 |
| `Floor Level` | str | categorical/text | 0 (0.000%) | 8740 | 1 |

#### `data/raw/ura/private/URA Private Residential Property Transactions (1).csv`

Columns (exact order): `Project Name, Transacted Price ($), Area (SQFT), Unit Price ($ PSF), Sale Date, Street Name, Type of Sale, Type of Area, Area (SQM), Unit Price ($ PSM), Nett Price($), Property Type, Number of Units, Tenure, Postal District, Market Segment, Floor Level`

| Column | pandas dtype | Inferred meaning | Nulls | `-` placeholders | Unique |
| --- | --- | --- | --- | --- | --- |
| `Project Name` | str | identifier/text | 0 (0.000%) | 0 | 5 |
| `Transacted Price ($)` | str | numeric | 0 (0.000%) | 0 | 83 |
| `Area (SQFT)` | str | numeric | 0 (0.000%) | 0 | 21 |
| `Unit Price ($ PSF)` | str | numeric | 0 (0.000%) | 0 | 91 |
| `Sale Date` | str | year-month | 0 (0.000%) | 0 | 53 |
| `Street Name` | str | identifier/text | 0 (0.000%) | 0 | 5 |
| `Type of Sale` | str | categorical/text | 0 (0.000%) | 0 | 1 |
| `Type of Area` | str | categorical/text | 0 (0.000%) | 0 | 1 |
| `Area (SQM)` | int64 | numeric | 0 (0.000%) | 0 | 21 |
| `Unit Price ($ PSM)` | str | numeric | 0 (0.000%) | 0 | 100 |
| `Nett Price($)` | str | categorical/text | 0 (0.000%) | 117 | 1 |
| `Property Type` | str | categorical/text | 0 (0.000%) | 0 | 4 |
| `Number of Units` | int64 | integer-coded numeric | 0 (0.000%) | 0 | 1 |
| `Tenure` | str | categorical/text | 0 (0.000%) | 0 | 2 |
| `Postal District` | int64 | integer-coded numeric | 0 (0.000%) | 0 | 4 |
| `Market Segment` | str | categorical/text | 0 (0.000%) | 0 | 3 |
| `Floor Level` | str | categorical/text | 0 (0.000%) | 6 | 5 |

## HDB comparison and modeling readiness

| File | Rows | Coverage | `remaining_lease` | Exact dups |
| --- | --- | --- | --- | --- |
| Resale Flat Prices (Based on Approval Date), 1990 - 1999.csv | 287,196 | 1990-01–1999-12 | no | 826 |
| Resale Flat Prices (Based on Approval Date), 2000 - Feb 2012.csv | 369,651 | 2000-01–2012-02 | no | 513 |
| Resale Flat Prices (Based on Registration Date), From Jan 2015 to Dec 2016.csv | 37,153 | 2015-01–2016-12 | yes | 24 |
| Resale Flat Prices (Based on Registration Date), From Mar 2012 to Dec 2014.csv | 52,203 | 2012-03–2014-12 | no | 248 |
| Resale flat prices based on registration date from Jan-2017 onwards.csv | 225,320 | 2017-01–2026-02 | yes | 311 |

All ten core columns match. Only `remaining_lease` differs: it exists in the 2015–2016 file as integer years and the 2017+ file as text containing years/months. Add a nullable standardized numeric field before concatenation. Combined: **971,523 rows**, **1990-01 to 2026-02**. All file boundaries are month-contiguous; the date definition changes from approval date to registration date at March 2012.

There are **0 exact records shared between any two files**. After concatenation there are **2,006** repeated rows on the ten common columns (**1,922** when nullable `remaining_lease` is included). These are within-file and may be distinct anonymized sales; do not blindly delete them.

Naming changes are real but manageable: the 1990s file uses uppercase flat-model labels; later files use title case. `MULTI GENERATION` becomes `MULTI-GENERATION`; equivalent flat-model labels vary by case/hyphen. Newer model values appear over time (`DBSS`, `Type S1/S2`, `3Gen`, loft variants). Storey bands include historical five-floor bins (`01 TO 05`, `06 TO 10`, etc.) alongside modern three-floor bins, especially around 2012; parse bounds rather than treating spelling as immutable.

Older exact remaining lease cannot be reconstructed because lease commencement has only a year, not a month. A consistent **approximate** feature can be computed from transaction month and commencement year. Against supplied values, the January-start proxy is within 12 months for 37,152/37,153 rows in 2015–2016 and 224,255/225,320 rows in 2017+; rare larger deviations show why it must be labeled approximate and the supplied detailed value retained where present.

Useful HDB predictors: transaction time, town, normalized block+street/location, flat type/model, floor area, parsed storey midpoint/bounds, lease commencement/property age, and standardized remaining lease. Later coordinates and nearest-MRT distance can be joined through normalized address.

## URA Executive Condominium audit

The two schemas are identical (17 columns). Part 2 covers **2021-09–2023-08** and Part 1 covers **2023-08–2026-09**. They overlap in **August 2023** only (61 Part 1 rows and 384 Part 2 rows), with **0 exact cross-file matches**. Therefore concatenate them, preserve source-file provenance, and do not pre-deduplicate the 1,428 indistinguishable within-file/combined repeated rows without a transaction identifier.

Combined: **16,495 rows**, **2021-09–2026-09**, **82 projects**, **64 streets**, and **82 project/street pairs**. Category distributions:

- `Project Name` (82 values): AURELLE OF TAMPINES: 757, COPEN GRAND: 638, COASTAL CABANA: 619, TENET: 618, NORTH GAIA: 615, OTTO PLACE: 596, SOL ACRES: 567, RIVELLE TAMPINES: 558, LUMINA GRAND: 512, NOVO PLACE: 504, PARC GREENWICH: 496, ALTURA: 360, … (82 values total)
- `Street Name` (64 values): TAMPINES STREET 62: 1,375, PLANTATION CLOSE: 1,100, ANCHORVALE CRESCENT: 893, FERNVALE LANE: 783, CANBERRA DRIVE: 731, TENGAH GARDEN WALK: 638, JALAN LOYANG BESAR: 619, YISHUN CLOSE: 615, CHOA CHU KANG GROVE: 567, TAMPINES STREET 95: 558, BUKIT BATOK WEST AVENUE 5: 512, EDGEDALE PLAINS: 478, … (64 values total)
- `Property Type` (1 values): Executive Condominium: 16,495
- `Type of Sale` (2 values): Resale: 9,892, New Sale: 6,603
- `Type of Area` (1 values): Strata: 16,495
- `Tenure` (23 values): 99 yrs lease commencing from 2014: 2,115, 99 yrs lease commencing from 2024: 1,970, 99 yrs lease commencing from 2021: 1,870, 99 yrs lease commencing from 2013: 1,758, 99 yrs lease commencing from 2012: 1,573, 99 yrs lease commencing from 2011: 911, 99 yrs lease commencing from 2022: 871, 99 yrs lease commencing from 1997: 870, 99 yrs lease commencing from 2010: 806, 99 yrs lease commencing from 2015: 778, 99 yrs lease commencing from 2020: 630, 99 yrs lease commencing from 2025: 558, … (23 values total)
- `Postal District` (10 values): 19: 3,402, 18: 3,345, 27: 2,525, 23: 2,280, 24: 1,738, 25: 1,027, 28: 913, 17: 619, 22: 559, 20: 87
- `Market Segment` (2 values): Outside Central Region: 16,453, Rest of Central Region: 42
- `Floor Level` (7 values): 06 to 10: 5,676, 01 to 05: 5,474, 11 to 15: 4,268, 16 to 20: 935, 21 to 25: 131, 26 to 30: 7, 31 to 35: 4

`Property Type`, `Type of Area`, and `Number of Units` are constant for EC; they add no within-EC signal. Tenure is almost entirely a parsable 99-year lease commencement string; seven rows use generic `99 years leasehold` and lack a commencement year.

## Landed-property audit

Coverage is **8,740 rows**, **2021-09–2026-09**. Property types: Terrace House: 4,938, Semi-Detached House: 2,725, Detached House: 1,077. Projects: **478**; streets: **1,493**; project/street pairs: **2,046**. `LANDED HOUSING DEVELOPMENT` is a generic label on **2,834 rows**, so Project Name quality is mixed and Street Name carries more location detail.

All location fields are populated. Coverage spans **24 postal districts** (2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28) and all market segments: Outside Central Region: 5,953, Rest of Central Region: 1,648, Core Central Region: 1,139. Tenure has **65 raw values**: Freehold dominates (5,852); the remainder include many 99/999-year and a few unusual lease lengths that should be parsed into tenure class, term, and start year.

The target is strongly right-skewed (median $4,608,280, P99 $26,500,000, max $815,000,000). The maximum is a **$815m, 25-unit, 263,796.58 sqft** transaction; **18 rows** contain multiple units. These are valid-looking portfolio/development transactions but are not comparable to a single home and need flags/segmentation, not silent deletion.

Assessment: sufficient for a dedicated exploratory landed model, subject to chronological evaluation, robust losses/log target, explicit multi-unit treatment, and careful high-cardinality location encoding. Performance should also be reported by house type and price tier.

## Private-residential audit

The file has **117 rows**, **2021-09–2026-08**, **5 projects**, **5 streets**, **4 postal districts**, and **3 market segments**. Transactions per project: CASABLANCA: 87, GRANGE RESIDENCES: 23, SEVEN CRESCENT: 4, EIGHT @ EAST COAST: 2, BALESTIER PLAZA: 1. Property types: Condominium: 110, Detached House: 4, Terrace House: 2, Apartment: 1. Districts: 25: 87, 10: 23, 15: 6, 12: 1. Segments: Outside Central Region: 89, Core Central Region: 23, Rest of Central Region: 5.

This is a narrow, apparently selected subset rather than broad private-residential coverage. It mixes 110 condominium rows with four detached-house, two terrace-house, and one apartment record; project concentration and the $762k–$10.8m target span make a general relationship unlearnable from 117 rows. **Recommendation: do not build a general private condo/apartment predictor from this file.** Keep it for data-pipeline tests/EDA or clearly project-specific experiments only.

## Data leakage and feature policy

### HDB

- **Safe Candidate Predictors:** month (encoded without using future data); town; flat_type; block/street or later coordinates; storey_range; floor_area_sqm; flat_model; lease_commence_date; remaining_lease when standardized to one numeric representation.
- **Safe Derived Features:** transaction year/month/quarter; storey lower/upper/midpoint; property age; approximate remaining lease for old rows; normalized address; later coordinates and nearest-MRT distance; historical market indices available as of the transaction quarter.
- **Eda Only:** resale_price-derived price per sqm; same-period aggregate/median prices if not lagged.
- **Target Leaking:** resale_price (target); any price per sqm computed from resale_price; post-sale valuation or outcome fields.
- **Likely Useless Or Redundant:** raw address after coordinates/address encoding is finalized; both lease commencement and an exactly derived age/lease feature without regularization.

### URA (EC, landed, private)

- **Safe Candidate Predictors:** Sale Date; Project Name; Street Name; Type of Sale; Type of Area; Area (SQFT) or Area (SQM); Property Type; Number of Units; Tenure; Postal District; Market Segment; Floor Level.
- **Safe Derived Features:** transaction year/month/quarter; one canonical area unit; tenure class/lease commencement/remaining lease proxy; floor-band midpoint; project/street normalization; later coordinates and nearest-MRT distance.
- **Eda Only:** Unit Price ($ PSF); Unit Price ($ PSM); Nett Price($) for source validation only.
- **Target Leaking:** Transacted Price ($) (target); Unit Price ($ PSF); Unit Price ($ PSM); Nett Price($) if populated; any total/area or price-derived aggregate computed using the target.
- **Likely Useless Or Redundant:** Area (SQFT) and Area (SQM) together; Property Type/Type of Area/Number of Units when constant within a category; Floor Level for landed (always '-'); Nett Price($) in supplied files (always '-').

Why PSF/PSM leak: they are effectively `Transacted Price / Area` (rounded), so a model receiving unit price plus area can algebraically reconstruct the target. Their strong correlation is mechanical, not predictive information available before sale. `Nett Price($)` is `-` in every supplied URA row, but if later populated it is another sale outcome and must remain excluded.

## Data quality, malformed values, and outliers

- All transaction files have zero true nulls, blank cells, unparseable dates, unparseable target prices, and non-positive prices. URA uses `-` as a semantic missing-value placeholder: all `Nett Price($)` values, all landed `Floor Level` values, and six private `Floor Level` values are `-`.
- URA area/unit-price pairs are arithmetically consistent up to expected display rounding; retain one area unit only. PSF/PSM remain EDA-only leakage fields.
- HDB `block` is an identifier, not numeric: many values contain suffix letters. Postal District is also categorical despite being read as an integer.
- Exact repeated rows occur in most large files, but absence of unit/transaction IDs means they cannot safely be declared erroneous duplicates.
- HDB target extremes range from $5,000 (historical) to $1,700,000; EC from $530,000 to $3,700,000; landed has the portfolio/luxury extremes noted above; private mixes dissimilar asset types and price tiers. Flag and inspect—do not delete—using time/category-aware rules rather than one global IQR rule.

## Location/MRT readiness

| Category | Unique projects | Unique streets | Unique address/project-street keys | Missing identifiers |
| --- | --- | --- | --- | --- |
| HDB | n/a | 595 | 9947 | 0 block; 0 street |
| EC | 82 | 64 | 82 | 0 project; 0 street; 0 district |
| Landed | 478 | 1493 | 2046 | 0 project; 0 street; 0 district |
| Private | 5 | 5 | 5 | 0 project; 0 street; 0 district |

- HDB has no postal code and uses abbreviated/historical street names; 24 of 595 transaction street names are absent from the supplied street reference after simple case normalization.
- URA supplies postal district, not full postal code or unit/block number; project/street pairs can span multiple buildings.
- Landed Project Name is generic for 2,834 rows ('LANDED HOUSING DEVELOPMENT'), so Street Name is the stronger geocoding key; street names alone may still cover many house numbers.
- No source contains latitude/longitude. Geocoding should create a cached property lookup later, without altering raw files.

The HDB street workbook contains 740 rows, 81 null rows, 645 unique non-null street names, and 14 duplicated non-null names. It covers most transaction streets after case normalization, but 24/595 transaction street names are absent; historical/abbreviated names require a normalization/alias table before geocoding.

## Time-aware readiness

Transaction counts by year:

- **HDB:** 1990: 12,505, 1991: 12,855, 1992: 14,503, 1993: 18,116, 1994: 26,373, 1995: 27,289, 1996: 34,919, 1997: 31,759, 1998: 51,095, 1999: 57,782, 2000: 34,862, 2001: 38,055, 2002: 36,098, 2003: 29,003, 2004: 29,112, 2005: 30,045, 2006: 27,427, 2007: 26,982, 2008: 27,262, 2009: 30,482, 2010: 34,854, 2011: 22,281, 2012: 23,198, 2013: 16,097, 2014: 16,096, 2015: 17,780, 2016: 19,373, 2017: 20,509, 2018: 21,561, 2019: 22,186, 2020: 23,333, 2021: 29,087, 2022: 26,720, 2023: 25,754, 2024: 27,832, 2025: 25,092, 2026: 3,246
- **EC:** 2021: 1,287, 2022: 3,443, 2023: 2,688, 2024: 3,245, 2025: 3,471, 2026: 2,361
- **Landed:** 2021: 874, 2022: 1,683, 2023: 1,281, 2024: 1,695, 2025: 1,884, 2026: 1,323
- **Private:** 2021: 6, 2022: 21, 2023: 31, 2024: 24, 2025: 22, 2026: 13

Candidate chronological splits (fit preprocessing only on each training window):

| Category | Train | Validation | Test |
| --- | --- | --- | --- |
| Hdb | through 2022-12 (889,599) | 2023-01 through 2024-12 (53,586) | 2025-01 onward (28,338) |
| Ec | through 2023-12 (7,418) | 2024-01 through 2024-12 (3,245) | 2025-01 onward (5,832) |
| Landed | through 2023-12 (3,838) | 2024-01 through 2024-12 (1,695) | 2025-01 onward (3,207) |
| Private | through 2023-12 (58) | 2024-01 through 2024-12 (24) | 2025-01 onward (35) |

For HDB, the long history may warrant a rolling/expanding-window backtest and/or a recent-era training window because market regimes and label definitions changed. Private counts are shown only for completeness; the split is too small and concentrated to support meaningful evaluation.

## Reference files

- `HDB Street Names.xlsx`: one sheet/column; useful as a street-name reference but contains blanks, duplicates, and incomplete historical transaction coverage as quantified above.
- `rpi-table.pdf`: local text extraction succeeded. It contains quarterly HDB Resale Price Index data from **1990-Q1 through 2026-Q2**, with 2009-Q1 = 100 and quarter-over-quarter changes. It can later be converted to a validated quarter/index table and joined to HDB by transaction quarter. The extracted text interleaves year labels with rows, so conversion needs sequence/spot checks.
- `Median resale prices for registered resale applications.pdf`: 77 pages, quarterly town × flat-type median tables from **2007-Q2 through 2026-Q2**, with suppressed low-count cells. Useful for EDA/benchmarking; same-quarter values must not become predictors unless lagged.
- `Important Project Info.docx`: a requirements/wishlist document, not data. It requests rental, geospatial, economic, and market data and suggests model/UI outputs. It does **not** supply rental records.
- No CSV/XLSX/PDF/DOCX supplied here contains transaction-level rental data; rental prediction/yield is out of scope.

## Recommended preprocessing sequence (do not execute yet)

1. Create category-specific canonical schemas while retaining `source_file` and raw categorical fields.
2. Normalize HDB flat-model/type spelling; parse all storey ranges; standardize supplied remaining lease and create a clearly named approximate value for older rows.
3. Parse URA money/area/count fields, keep one area unit, parse tenure, convert `-` to missing, and permanently exclude PSF/PSM/nett price from model feature lists.
4. Define single-unit versus portfolio landed segments and document outlier flags without deleting source records.
5. Build cached unique-location tables for later geocoding, starting with HDB block+street and URA project+street+district; preserve unresolved/ambiguous matches.
6. Convert and validate RPI by quarter, then join using only information available at prediction time.
7. Materialize chronological splits before fitting encoders/imputers; establish naive time/location baselines before advanced models.
8. Exclude the supplied private file from a general private model until substantially broader transaction coverage is provided.
