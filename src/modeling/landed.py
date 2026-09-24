"""Landed-property baseline feature definitions."""

from .common import FeatureSet

FEATURES = FeatureSet(
    numeric=(
        "area_sqm", "number_of_units", "lease_duration_years",
        "property_age_at_transaction", "transaction_year", "transaction_month",
        "is_multi_unit_transaction",
    ),
    categorical=(
        "sale_type", "area_type", "property_type", "tenure_category",
        "postal_district", "market_segment",
    ),
)

ERROR_GROUPS = {
    "property_type": 30, "postal_district": 30, "market_segment": 30,
    "price_band": 30,
}
