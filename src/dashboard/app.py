"""
Streamlit multipage entry point for the Crypto Prediction Dashboard.

Run with::

    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import streamlit as st

from src.utils.config import load_config

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit command)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Crypto Prediction Dashboard",
    page_icon="\U0001F4C8",  # chart emoji
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Auto-refresh support
# ---------------------------------------------------------------------------

cfg = load_config()

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

pages = {
    "Real-Time Market": "realtime",
    "Price Prediction": "prediction",
    "Anomaly Detection": "anomaly",
    "Model Comparison": "comparison",
}

st.sidebar.title("Crypto Prediction Dashboard")

selected = st.sidebar.radio("Navigation", list(pages.keys()))

st.sidebar.divider()

# ---------------------------------------------------------------------------
# Route to the selected page
# ---------------------------------------------------------------------------

if selected == "Real-Time Market":
    from src.dashboard.pages.realtime import render
elif selected == "Price Prediction":
    from src.dashboard.pages.prediction import render
elif selected == "Anomaly Detection":
    from src.dashboard.pages.anomaly import render
elif selected == "Model Comparison":
    from src.dashboard.pages.comparison import render
else:
    from src.dashboard.pages.realtime import render

render()
