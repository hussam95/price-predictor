# Time-aware baseline modeling summary

All models use the existing chronological `data_split` assignments. Validation selected configurations; test was used only for shortlisted MRT pairs and final landed policy comparisons.

## HDB

Selected baseline: **hist_gradient_boosting / with_mrt**.
- Validation: MAE $62,799, RMSE $82,419, R² 0.797 on 53,586 rows (MAE 11.02% of median target).
- Test: MAE $120,336, RMSE $138,261, R² 0.541 on 28,338 rows (MAE 19.16% of median target).
- MRT validation MAE changed by $-971 (-1.52%); RMSE changed by $-2,769 (-3.25%).
- MRT test MAE changed by $17 (+0.01%); RMSE changed by $-2,604 (-1.85%).
- Decision: **retain MRT distance** based on validation MAE.

## EC

Selected baseline: **ridge / without_mrt**.
- Validation: MAE $86,251, RMSE $106,616, R² 0.855 on 3,245 rows (MAE 5.92% of median target).
- Test: MAE $130,502, RMSE $160,182, R² 0.769 on 5,832 rows (MAE 7.91% of median target).
- MRT validation MAE changed by $3,484 (+4.04%); RMSE changed by $5,240 (+4.92%).
- MRT test MAE changed by $8,297 (+6.36%); RMSE changed by $7,520 (+4.69%).
- Decision: **do not retain MRT distance** based on validation MAE.

## Landed

Selected baseline: **hist_gradient_boosting**, policy **exclude_multi_unit**; MRT is prohibited.
- Validation: MAE $797,671, RMSE $1,446,522, R² 0.885 on 1,694 rows (MAE 18.01% of median target).
- Test: MAE $1,205,642, RMSE $2,944,452, R² 0.709 on 3,203 rows (MAE 23.64% of median target).
- Multi-unit rows by split: {'train': 13, 'validation': 1, 'test': 4}. Luxury single-property transactions were retained.

- Excluding multi-unit training rows changed single-unit validation MAE by $-4,553 (-0.57%) and test MAE by $9,073 (+0.76%). The exclusion policy is retained because validation selected it and portfolio sales do not match individual-property inference; the test difference is small and adverse.

Policy comparison on single-unit evaluation rows:

| Split | Training policy | Rows | MAE | RMSE | R² |
| --- | --- | ---: | ---: | ---: | ---: |
| validation | all_transactions | 1,694 | $802,224 | $1,441,302 | 0.886 |
| validation | exclude_multi_unit | 1,694 | $797,671 | $1,446,522 | 0.885 |
| test | all_transactions | 3,203 | $1,196,569 | $2,908,155 | 0.716 |
| test | exclude_multi_unit | 3,203 | $1,205,642 | $2,944,452 | 0.709 |

## Features and preprocessing

- **HDB numeric:** floor_area_sqm, storey_midpoint, lease_commence_year, property_age_at_transaction, transaction_year, transaction_month, nearest_mrt_distance_m, mrt_distance_missing.
- **HDB categorical:** town, flat_type, flat_model.
- **EC numeric:** area_sqm, number_of_units, floor_midpoint, lease_duration_years, property_age_at_transaction, transaction_year, transaction_month.
- **EC categorical:** project_name, sale_type, area_type, property_type, tenure_category, postal_district, market_segment.
- **LANDED numeric:** area_sqm, number_of_units, lease_duration_years, property_age_at_transaction, transaction_year, transaction_month, is_multi_unit_transaction.
- **LANDED categorical:** sale_type, area_type, property_type, tenure_category, postal_district, market_segment.
- Numeric missing values use training-fitted median imputation. Linear models standardize numeric values and one-hot encode categories; HistGradientBoosting uses training-fitted ordinal category mappings and native categorical splits.
- Excluded: targets and price derivatives, diagnostic target-outlier flags, coordinates, station/exit names, reference-only MRT distance, HDB block/street/location key, and landed project/street. EC project is retained because its observed cardinality is modest and it identifies the development.
- `transaction_year` and `transaction_month` are known at inference: a future UI must ask for the intended valuation/transaction month (defaulting explicitly to the current month), then derive both values. RPI is not used.

## Empirical error ranges

These are empirical absolute errors, not formal confidence intervals.

| Category | Split | Rows | Median | P80 | P90 | P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HDB | validation | 53,586 | $50,271 | $94,589 | $128,151 | $163,298 |
| HDB | test | 28,338 | $107,999 | $162,184 | $202,684 | $242,812 |
| EC | validation | 3,245 | $75,121 | $132,739 | $165,948 | $201,552 |
| EC | test | 5,832 | $119,734 | $192,901 | $244,593 | $291,300 |
| LANDED | validation | 1,694 | $434,186 | $1,086,275 | $1,909,671 | $2,833,587 |
| LANDED | test | 3,203 | $741,422 | $1,651,339 | $2,476,091 | $3,341,832 |

## Feature importance and error analysis

- **HDB permutation importance:** floor_area_sqm ($32,637 MAE increase), town ($10,692 MAE increase), lease_commence_year ($6,060 MAE increase), flat_type ($4,465 MAE increase), storey_midpoint ($1,991 MAE increase).
- **EC permutation importance:** area_sqm ($153,757 MAE increase), property_age_at_transaction ($119,733 MAE increase), transaction_year ($22,733 MAE increase), project_name ($11,944 MAE increase), postal_district ($10,922 MAE increase).
- **LANDED permutation importance:** area_sqm ($1,233,285 MAE increase), postal_district ($375,914 MAE increase), lease_duration_years ($286,055 MAE increase), tenure_category ($261,766 MAE increase), property_age_at_transaction ($207,167 MAE increase).
- **HDB error pattern:** EXECUTIVE flats had $182,871 MAE; BISHAN had $199,246 MAE. The worst recent month (2026-01) had $134,699 MAE and $134,581 mean underprediction, showing strong post-training temporal drift.
- **EC error pattern:** NORTHOAKS had $264,266 MAE; RESALE transactions had $147,810 MAE. The low price band was hardest at $245,081 MAE.
- **Landed error pattern:** DETACHED HOUSE had $3,289,770 MAE; CORE CENTRAL REGION had $2,638,456 MAE, and the high price band had $2,214,174 MAE. The four multi-unit test rows had $454,343 MAE and are too few for a stable estimate.
- Group-level errors are in `reports/modeling/error_analysis/`; minimum group sizes are HDB 100 (town 200), EC 30, and landed 30 except the explicitly flagged unit-policy diagnostic.
- Price-band thresholds are fitted from training-target quartiles and used only for held-out error analysis, never as model features.

## Reproducibility, usefulness, and limitations

- Python 3.13.1, scikit-learn 1.9.1, pandas 3.0.6, NumPy 2.5.3; random seed 42.
- Split plan: `{"ec": {"split_column": "data_split", "test": {"from": "2025-01", "rows": 5832, "through": "2026-09"}, "train": {"rows": 7418, "through": "2023-12"}, "validation": {"from": "2024-01", "rows": 3245, "through": "2024-12"}}, "hdb": {"split_column": "data_split", "test": {"from": "2025-01", "rows": 28338, "through": "2026-02"}, "train": {"rows": 889599, "through": "2022-12"}, "validation": {"from": "2023-01", "rows": 53586, "through": "2024-12"}}, "landed": {"split_column": "data_split", "test": {"from": "2025-01", "rows": 3207, "through": "2026-09"}, "train": {"rows": 3838, "through": "2023-12"}, "validation": {"from": "2024-01", "rows": 1695, "through": "2024-12"}}, "method": "Chronological month boundaries; preprocessing must be fit on training rows only.", "version": 1}`.
- Total measured model-fit time: 39.9 seconds. No grid search or external data was used.
- **Demonstration readiness:** EC is the strongest baseline (test MAE 7.91% of its median). HDB is usable only as a visibly caveated demo because test MAE rises to 19.16% under temporal drift. Landed is a coarse demo estimate only (23.64%); it is not yet reliable for high-end valuation.
- The outputs are not valuation advice; rare luxury assets, unseen developments, structural market shifts, and coarse landed location remain material limitations.
- Next milestone: limited model refinement and calibration using validation only, including temporal-trend treatment, rare/unseen-category diagnostics, and optional HDB RPI with explicit inference semantics; then freeze production pipelines.
