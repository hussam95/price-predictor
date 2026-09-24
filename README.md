# Singapore Residential Property Price Predictor

A fully local Streamlit application for estimating HDB resale, Executive Condominium (EC), and landed-property transaction prices. Production models are already trained and frozen; normal app usage does not retrain models, geocode properties, download MRT data, or call OneMap.

## Run locally

Python 3.11 or newer is recommended. Keep the supplied `models/` and `data/processed/` artifacts in place.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

For Windows Command Prompt, activate with `.venv\Scripts\activate.bat`.

The browser app contains four views: the predictor, local market summaries, frozen-model performance, and methodology. Runtime prediction needs no internet connection, API key, or OneMap token.

## What users enter

- **HDB:** valuation month, town, street, block, flat type, flat model, floor area, storey range, and lease commencement year. Lagged official HDB RPI and eligible MRT distance are resolved locally.
- **EC:** valuation month, project, area, floor level, sale type, area type, property type, tenure details, postal district, and market segment. MRT may be displayed as context but is not used by the model.
- **Landed:** valuation month, land area, property type, sale type, area type, tenure details, postal district, and market segment. The model always represents one individual property and does not use MRT.

Displayed ranges use the P80 historical absolute error for the applicable predicted-price band. They are not confidence intervals or guarantees.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests are deterministic, make no network requests, and do not retrain models.

## Optional development workflows

These are not needed to run the app:

```bash
# Rebuild compact dropdown, address-lookup, and market-summary artifacts
python -m src.inference.build_ui_artifacts

# Canonical preprocessing (rewrites processed transaction datasets)
python -m src.preprocess_data

# Location enrichment from an already populated local geocoding cache
python -m src.enrich_location_data
```

Geocoding and model-training workflows are intentionally separate from the application. See the reports under `reports/` for audit, preprocessing, location validation, and frozen-model methodology.

## Scope and limitations

The supplied private-residential data was too narrow for a general condo/apartment model, and rental transaction data was not supplied. Predictions are estimates, not professional valuations or financial advice. Landed estimates have materially wider errors because individual homes are heterogeneous and exact landed locations are unavailable.
