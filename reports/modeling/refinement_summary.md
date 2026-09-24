# Model refinement and production freeze

Configuration selection used TRAIN fitting and VALIDATION metrics only. Existing TEST results were not used for selection. After each configuration was fixed, it was refit on TRAIN+VALIDATION and evaluated once on TEST; no post-test changes were made.

## HDB drift diagnosis

- The baseline mean prediction bias was $-39,431 in 2023 and $-79,451 in 2024 (prediction minus actual).
- Actual median rose from $550,000 in 2023 to $590,000 in 2024, while predicted median moved from $519,833 to $519,546. This is predominantly a market-level upward shift that a bounded tree time split cannot extrapolate.
- SENGKANG contributed the largest town share of validation absolute error (8.6%); 4 ROOM contributed the largest flat-type share (43.9%). Full year/quarter and group tables are provided separately.

## Validation experiments

| Category | Experiment | Selectable | MAE | RMSE | R² | Bias |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| HDB | A_current_baseline | True | $62,799 | $82,419 | 0.797 | $-60,217 |
| HDB | B_lagged_rpi | True | $52,309 | $74,276 | 0.835 | $-45,720 |
| HDB | C_lagged_rpi_no_mrt | True | $54,494 | $77,653 | 0.820 | $-47,211 |
| HDB | D_month_index_lagged_rpi | True | $50,829 | $71,634 | 0.846 | $-44,024 |
| HDB | E_rpi_scaled_target_with_mrt | True | $30,194 | $42,581 | 0.946 | $6,859 |
| HDB | F_rpi_scaled_target_no_mrt | True | $34,390 | $48,954 | 0.928 | $6,056 |
| HDB | diagnostic_contemporaneous_rpi | False | $51,399 | $72,835 | 0.841 | $-45,070 |
| EC | A_current_ridge | True | $86,251 | $106,616 | 0.855 | $45,974 |
| EC | B_time_index_ridge | True | $84,371 | $105,986 | 0.857 | $38,060 |
| EC | C_time_index_ridge_rare_project | True | $89,801 | $111,771 | 0.841 | $49,583 |
| EC | D_time_index_hist_gradient_boosting | True | $110,767 | $150,643 | 0.710 | $-98,543 |
| LANDED | A_current_hist_gradient_boosting | True | $797,671 | $1,446,522 | 0.885 | $-392,652 |
| LANDED | B_time_index_hist_gradient_boosting | True | $794,897 | $1,474,765 | 0.880 | $-361,283 |
| LANDED | C_time_index_log_target | True | $780,251 | $1,428,732 | 0.888 | $-348,539 |

## Final frozen models and one-time test

### HDB

- Frozen configuration: **E_rpi_scaled_target_with_mrt**.
- Test: MAE $32,723, RMSE $46,304, R² 0.949, MAE 5.21% of median, bias $11,800.
- Versus Milestone 4: MAE $-87,613, RMSE $-91,957, R² +0.407.

### EC

- Frozen configuration: **B_time_index_ridge**.
- Test: MAE $101,868, RMSE $138,187, R² 0.828, MAE 6.17% of median, bias $-2,849.
- Versus Milestone 4: MAE $-28,634, RMSE $-21,995, R² +0.059.

### LANDED

- Frozen configuration: **C_time_index_log_target**.
- Test: MAE $943,058, RMSE $2,659,602, R² 0.762, MAE 18.49% of median, bias $-516,171.
- Versus Milestone 4: MAE $-262,584, RMSE $-284,850, R² +0.054.

## RPI and MRT policy

- `hdb_rpi.csv` contains 146 validated official quarters from 1990-Q1 through 2026-Q2. Contemporaneous RPI was diagnostic-only and could not be selected.
- Production HDB uses lagged RPI: **yes**. It uses the latest official quarter strictly before the valuation quarter.
- For a future month beyond available data, the app uses the latest older official RPI and displays its source quarter; it never extrapolates or fabricates RPI.
- Deterministic examples: `{"2026-03-01": {"is_latest_available_fallback": false, "lagged_rpi": 203.6, "rpi_source_quarter": "2025-Q4"}, "2026-07-01": {"is_latest_available_fallback": false, "lagged_rpi": 202.8, "rpi_source_quarter": "2026-Q2"}, "2027-01-01": {"is_latest_available_fallback": true, "lagged_rpi": 202.8, "rpi_source_quarter": "2026-Q2"}}`.
- Final HDB MRT usage: **retained**, selected using the matched validation configurations after temporal/RPI refinement.

## Rare/unseen categories

- HDB `flat_model`: 0 unseen validation rows; 195 rows in train-defined rare categories.
- EC `project_name`: 845 unseen validation rows; 190 rows in train-defined rare categories.
- EC `market_segment`: 0 unseen validation rows; 9 rows in train-defined rare categories.
- LANDED `postal_district`: 0 unseen validation rows; 21 rows in train-defined rare categories.
- All frozen preprocessors handle unseen categories without crashing. The EC rare-project experiment groups categories using TRAIN frequencies only; whether it was selected is shown above.

## Historical prediction error ranges

Ranges use absolute errors from the single final chronological test evaluation. The app should select a fixed broad band using the model prediction, then use that band's P80/P90/P95 value; fall back to the global value when needed. These are historical prediction error ranges, not confidence intervals.

- **HDB global:** P50 $23,007, P80 $50,709, P90 $72,726, P95 $97,758. Predicted-price bands are stored in `models/error_calibration.json`.
- **EC global:** P50 $76,897, P80 $160,358, P90 $213,176, P95 $265,428. Predicted-price bands are stored in `models/error_calibration.json`.
- **LANDED global:** P50 $476,514, P80 $1,279,123, P90 $2,071,129, P95 $2,967,655. Predicted-price bands are stored in `models/error_calibration.json`.

## Limitations

- RPI availability is quarter-level and publication timing is represented conservatively by requiring a strictly prior quarter; the local file must be deliberately updated to use newer official releases.
- HDB remains exposed to structural shifts not captured by RPI. EC unseen projects rely on non-project features or train-derived rare handling. Landed remains heterogeneous and coarse-location constrained.
- Multi-unit landed sales are excluded; legitimate luxury single-property sales remain. Models are demonstration tools, not professional valuations.
