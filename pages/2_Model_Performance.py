"""Plain-language frozen model performance."""

import streamlit as st

from src.inference.ui import page_intro


st.set_page_config(page_title="Model Performance", page_icon="🧪", layout="wide")
page_intro("Model Performance", "Results from fixed chronological test periods containing newer, unseen transactions.")

rows = [
    ("HDB", "S$32,723", "S$46,304", "0.949"),
    ("Executive Condominium", "S$101,868", "S$138,187", "0.828"),
    ("Landed", "S$943,058", "S$2,659,602", "0.762"),
]
for name, mae, rmse, r2 in rows:
    st.subheader(name)
    a, b, c = st.columns(3)
    a.metric("MAE", mae)
    b.metric("RMSE", rmse)
    c.metric("R²", r2)

st.markdown(
    """
- **MAE** is the average absolute dollar error.
- **RMSE** gives larger errors more weight.
- **R²** is the share of observed price variation captured by the model; it is not an accuracy percentage.

All testing was chronological: models fitted older transactions and the test sets contained newer transactions. HDB uses a strictly lagged official HDB Resale Price Index and validated MRT distance. EC excludes MRT because it worsened validation performance. Landed excludes MRT because source locations were too imprecise, and its estimates are less precise because landed homes are highly heterogeneous.
"""
)
