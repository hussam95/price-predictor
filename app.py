"""Local Streamlit entry point for Singapore residential price estimates."""

from __future__ import annotations

import streamlit as st

from src.inference.predictor import InferenceError
from src.inference.ui import app_predictor, app_references, page_intro, show_prediction, valuation_month_input
from src.inference.validation import InputValidationError


st.set_page_config(page_title="Singapore Property Price Predictor", page_icon="🏠", layout="wide")
page_intro(
    "Singapore Residential Property Price Predictor",
    "Estimate an HDB resale, Executive Condominium, or landed-house value using chronologically tested local models.",
)

try:
    refs = app_references()
    predictor = app_predictor()
except Exception as error:
    st.error(f"The local model or reference artifacts could not be loaded: {error}")
    st.stop()

category_label = st.radio(
    "Choose property type",
    ["HDB Flat", "Executive Condominium", "Landed House"],
    horizontal=True,
)


def hdb_inputs() -> dict:
    options = refs["hdb"]
    left, right = st.columns(2)
    with left:
        valuation = valuation_month_input("hdb_date")
        town = st.selectbox("Town", options["towns"])
        streets = options["town_streets"][town]
        street = st.selectbox("Street", streets)
        blocks = options["street_blocks"][f"{town}|{street}"]
        block = st.selectbox("Block", blocks)
        flat_type = st.selectbox("Flat type", options["flat_types"])
    with right:
        flat_model = st.selectbox("Flat model", options["flat_models"])
        storey_range = st.selectbox("Storey range", options["storey_ranges"])
        area_range = options["training_ranges"]["floor_area_sqm"]
        floor_area = st.number_input(
            "Floor area (sqm)", min_value=1.0, value=float(area_range["median"]), step=1.0
        )
        lease_range = options["training_ranges"]["lease_commence_year"]
        lease_year = st.number_input(
            "Lease commencement year",
            min_value=1900,
            max_value=2200,
            value=int(lease_range["median"]),
            step=1,
        )
    return {
        "valuation_month": valuation,
        "town": town,
        "street_name": street,
        "block": block,
        "flat_type": flat_type,
        "flat_model": flat_model,
        "storey_range": storey_range,
        "floor_area_sqm": floor_area,
        "lease_commence_year": lease_year,
    }


def ec_inputs() -> dict:
    options = refs["ec"]
    left, right = st.columns(2)
    with left:
        valuation = valuation_month_input("ec_date")
        project = st.selectbox("Project", options["projects"])
        context = options["project_context"].get(project, {})
        property_types = context.get("property_types") or options["property_types"]
        property_type = st.selectbox("Property type", property_types)
        area_range = options["training_ranges"]["area_sqm"]
        area = st.number_input("Area (sqm)", min_value=1.0, value=float(area_range["median"]), step=1.0)
        floor_level = st.selectbox("Floor level", options["floor_levels"])
        sale_type = st.selectbox("Type of sale", options["sale_types"])
    with right:
        area_type = st.selectbox("Type of area", options["area_types"])
        tenure_values = context.get("tenure_categories") or options["tenure_categories"]
        tenure = st.selectbox("Tenure", tenure_values)
        duration_values = context.get("lease_durations") or [99]
        duration = st.number_input("Lease duration (years)", min_value=1, value=int(duration_values[0]), step=1)
        commence_values = context.get("lease_commence_years") or [2020]
        commencement = st.number_input(
            "Lease commencement year", min_value=1800, max_value=2200, value=int(commence_values[0]), step=1
        )
        districts = context.get("postal_districts") or options["postal_districts"]
        district = st.selectbox("Postal district", districts)
        segment = st.selectbox("Market segment", options["market_segments"])
    return {
        "valuation_month": valuation,
        "project_name": project,
        "property_type": property_type,
        "area_sqm": area,
        "floor_level": floor_level,
        "sale_type": sale_type,
        "area_type": area_type,
        "tenure_category": tenure,
        "lease_duration_years": duration,
        "lease_commence_year": commencement,
        "postal_district": district,
        "market_segment": segment,
    }


def landed_inputs() -> dict:
    options = refs["landed"]
    left, right = st.columns(2)
    with left:
        valuation = valuation_month_input("landed_date")
        property_type = st.selectbox("Property type", options["property_types"])
        area_range = options["training_ranges"]["area_sqm"]
        area = st.number_input("Land area (sqm)", min_value=1.0, value=float(area_range["median"]), step=1.0)
        sale_type = st.selectbox("Type of sale", options["sale_types"])
        area_type = st.selectbox("Type of area", options["area_types"])
    with right:
        tenure = st.selectbox("Tenure", options["tenure_categories"])
        duration = None
        commencement = None
        if tenure == "LEASEHOLD":
            duration = st.number_input("Lease duration (years)", min_value=1, value=99, step=1)
            commencement = st.number_input(
                "Lease commencement year", min_value=1800, max_value=2200, value=1980, step=1
            )
        district = st.selectbox("Postal district", options["postal_districts"])
        segment = st.selectbox("Market segment", options["market_segments"])
        st.text_input("Number of properties", value="1", disabled=True, help="The model covers one individual property only.")
    return {
        "valuation_month": valuation,
        "property_type": property_type,
        "area_sqm": area,
        "sale_type": sale_type,
        "area_type": area_type,
        "tenure_category": tenure,
        "lease_duration_years": duration,
        "lease_commence_year": commencement,
        "postal_district": district,
        "market_segment": segment,
        "number_of_units": 1,
    }


if category_label == "HDB Flat":
    request = hdb_inputs()
    category = "hdb"
elif category_label == "Executive Condominium":
    request = ec_inputs()
    category = "ec"
else:
    request = landed_inputs()
    category = "landed"

if st.button("Estimate property value", type="primary", use_container_width=True):
    try:
        with st.spinner("Calculating estimate…"):
            result = predictor.predict(category, request)
        show_prediction(result)
    except (InputValidationError, InferenceError) as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Prediction could not be completed from the local artifacts: {error}")

st.divider()
st.caption("For demonstration only. Estimates are not professional valuations, offers, or financial advice.")
