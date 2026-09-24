"""HDB baseline feature definitions."""

from .common import FeatureSet

BASE_FEATURES = FeatureSet(
    numeric=(
        "floor_area_sqm", "storey_midpoint", "lease_commence_year",
        "property_age_at_transaction", "transaction_year", "transaction_month",
    ),
    categorical=("town", "flat_type", "flat_model"),
)

MRT_FEATURES = FeatureSet(
    numeric=(*BASE_FEATURES.numeric, "nearest_mrt_distance_m", "mrt_distance_missing"),
    categorical=BASE_FEATURES.categorical,
)

ERROR_GROUPS = {"flat_type": 100, "town": 200, "price_band": 100, "mrt_distance_band": 100, "recent_period": 100}

