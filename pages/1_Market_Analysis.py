"""Pre-aggregated local market views."""

import pandas as pd
import streamlit as st

from src.inference.ui import app_market_summary, money, page_intro


st.set_page_config(page_title="Market Analysis", page_icon="📈", layout="wide")
page_intro("Market Analysis", "Explore compact summaries precomputed from the supplied transaction data.")

market = app_market_summary()
category = st.selectbox("Market", ["HDB", "Executive Condominium", "Landed"])


def monthly_chart(records):
    frame = pd.DataFrame(records)
    frame["transaction_date"] = pd.to_datetime(frame["transaction_date"])
    st.line_chart(frame.set_index("transaction_date")["median_price"], y_label="Median price (S$)")


def table_chart(records, label):
    frame = pd.DataFrame(records).rename(columns={frame_column: label})
    st.bar_chart(frame.set_index(label)["median_price"], y_label="Median price (S$)")
    display = frame.rename(columns={"median_price": "Median price", "transactions": "Transactions"})
    display["Median price"] = display["Median price"].map(money)
    st.dataframe(display, hide_index=True, use_container_width=True)


if category == "HDB":
    data = market["hdb"]
    st.subheader("Median resale price over time")
    monthly_chart(data["monthly"])
    tab1, tab2, tab3 = st.tabs(["Towns", "Flat types", "HDB RPI"])
    with tab1:
        frame_column = "town"
        table_chart(data["towns"], "Town")
    with tab2:
        frame_column = "flat_type"
        table_chart(data["flat_types"], "Flat type")
    with tab3:
        rpi = pd.DataFrame(data["rpi"])
        rpi["period"] = rpi["year"].astype(str) + " Q" + rpi["quarter"].astype(str)
        st.line_chart(rpi.set_index("period")["rpi"], y_label="HDB Resale Price Index")
elif category == "Executive Condominium":
    data = market["ec"]
    st.subheader("Median transaction price over time")
    monthly_chart(data["monthly"])
    tab1, tab2, tab3 = st.tabs(["Projects", "Sale types", "Market segments"])
    with tab1:
        frame_column = "project_name"
        table_chart(data["projects"], "Project")
    with tab2:
        frame_column = "sale_type"
        table_chart(data["sale_types"], "Sale type")
    with tab3:
        frame_column = "market_segment"
        table_chart(data["market_segments"], "Market segment")
else:
    data = market["landed"]
    st.subheader("Median transaction price over time")
    monthly_chart(data["monthly"])
    tab1, tab2, tab3 = st.tabs(["Property types", "Districts", "Market segments"])
    with tab1:
        frame_column = "property_type"
        table_chart(data["property_types"], "Property type")
    with tab2:
        frame_column = "postal_district"
        table_chart(data["districts"], "Postal district")
    with tab3:
        frame_column = "market_segment"
        table_chart(data["market_segments"], "Market segment")

st.caption("Charts describe the supplied historical data; they are not forecasts.")
