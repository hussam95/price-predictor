"""Executive Condominium baseline feature definitions."""

from .common import FeatureSet

BASE_FEATURES = FeatureSet(
    numeric=(
        "area_sqm", "number_of_units", "floor_midpoint", "lease_duration_years",
        "property_age_at_transaction", "transaction_year", "transaction_month",
    ),
    categorical=(
        "project_name", "sale_type", "area_type", "property_type",
        "tenure_category", "postal_district", "market_segment",
    ),
)

MRT_FEATURES = FeatureSet(
    numeric=(*BASE_FEATURES.numeric, "nearest_mrt_distance_m", "mrt_distance_missing"),
    categorical=BASE_FEATURES.categorical,
)

ERROR_GROUPS = {"project_name": 30, "sale_type": 30, "market_segment": 30, "price_band": 30, "mrt_distance_band": 30}

