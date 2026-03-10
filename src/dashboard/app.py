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
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Metric card styling */
    div[data-testid="stMetric"] {
        background-color: #1a1e2e;
        border: 1px solid #2d3348;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label {
        color: #9e9e9e;
        font-size: 0.85rem;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: 600;
    }

    /* Section headers */
    h1, h2, h3 {
        color: #e0e0e0;
    }
    h2 {
        border-bottom: 2px solid #26a69a;
        padding-bottom: 6px;
    }

    /* Sidebar border */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #2d3348;
    }

    /* Sidebar footer */
    .sidebar-footer {
        position: fixed;
        bottom: 0;
        padding: 12px 16px;
        font-size: 0.75rem;
        color: #6b7280;
        border-top: 1px solid #2d3348;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Auto-refresh support
# ---------------------------------------------------------------------------

cfg = load_config()

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

pages = {
    "Market Overview": "realtime",
    "Training Results": "prediction",
    "Anomaly Detection": "anomaly",
    "Model Comparison": "comparison",
}

st.sidebar.title("Crypto Prediction Dashboard")

selected = st.sidebar.radio("Navigation", list(pages.keys()))

st.sidebar.divider()

# ---------------------------------------------------------------------------
# Route to the selected page
# ---------------------------------------------------------------------------

if selected == "Market Overview":
    from src.dashboard.pages.realtime import render
elif selected == "Training Results":
    from src.dashboard.pages.prediction import render
elif selected == "Anomaly Detection":
    from src.dashboard.pages.anomaly import render
elif selected == "Model Comparison":
    from src.dashboard.pages.comparison import render
else:
    from src.dashboard.pages.realtime import render

render()

# ---------------------------------------------------------------------------
# Sidebar footer
# ---------------------------------------------------------------------------

st.sidebar.markdown(
    '<div class="sidebar-footer">GCAP3123 Crypto Prediction System<br>'
    "Multi-model forecasting & anomaly detection</div>",
    unsafe_allow_html=True,
)
