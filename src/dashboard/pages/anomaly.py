"""
Anomaly Detection page.

Displays autoencoder-based anomaly detection summary statistics and a price
chart for context. The anomaly results JSON contains only summary stats
(no per-point arrays), so we show KPI cards.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.components.charts import _apply_defaults
from src.dashboard.components.sidebar import render_sidebar
from src.utils.constants import (
    METRICS_DIR,
    PROCESSED_DIR,
    PROCESSED_DATA_PATTERN,
    ANOMALY_RESULTS_PATTERN,
    TARGET_COL,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=120, show_spinner="Loading processed data ...")
def _load_processed(symbol: str, timeframe: str) -> pd.DataFrame | None:
    filename = PROCESSED_DATA_PATTERN.format(
        symbol=symbol.replace("/", "_"),
        timeframe=timeframe,
    )
    path = PROCESSED_DIR / filename
    if not path.exists():
        return None
    from src.utils.io import load_dataframe
    return load_dataframe(path)


@st.cache_data(ttl=300, show_spinner="Loading anomaly results ...")
def _load_anomaly_results(
    symbol: str, timeframe: str, lookback: int, horizon: int,
) -> dict[str, Any] | None:
    """Load anomaly detection results JSON produced by the anomaly pipeline."""
    filename = ANOMALY_RESULTS_PATTERN.format(
        symbol=symbol.replace("/", "_"),
        timeframe=timeframe,
        lookback=lookback,
        horizon=horizon,
    )
    for parent in [METRICS_DIR, METRICS_DIR.parent]:
        path = parent / filename
        if path.exists():
            with open(path, "r") as fh:
                return json.load(fh)
    return None


# ---------------------------------------------------------------------------
# Page renderer
# ---------------------------------------------------------------------------


def render() -> None:
    """Main entry point for the Anomaly Detection page."""

    st.header("Anomaly Detection")
    st.caption("LSTM-Autoencoder anomaly detection results. Threshold = mean + 3*std of reconstruction errors.")

    selections = render_sidebar(page="anomaly")
    symbol: str = selections["symbol"]
    timeframe: str = selections["timeframe"]
    lookback: int = selections["lookback_window"]
    horizon: int = selections["forecast_horizon"]

    # ---- Load anomaly results ----
    results = _load_anomaly_results(symbol, timeframe, lookback, horizon)
    if results is None:
        st.warning(
            f"No anomaly detection results found for **{symbol}** ({timeframe}, lb={lookback}, h={horizon}).  \n"
            "Run the anomaly detection pipeline first:\n\n"
            "```bash\npython scripts/run_anomaly_detection.py\n```"
        )
    else:
        # ---- Summary KPI cards ----
        n_test = results.get("n_test_samples", 0)
        n_anomalies = results.get("n_anomalies", 0)
        anomaly_ratio = results.get("anomaly_ratio", 0)
        threshold = results.get("threshold", None)
        mean_error = results.get("mean_error", None)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Test Samples", f"{n_test:,}")
        c2.metric("Anomalies Found", f"{n_anomalies:,}")
        c3.metric("Anomaly Ratio", f"{anomaly_ratio:.2%}")
        if threshold is not None:
            c4.metric("Threshold", f"{threshold:.6f}")
        else:
            c4.metric("Threshold", "N/A")
        if mean_error is not None:
            c5.metric("Mean Recon. Error", f"{mean_error:.6f}")
        else:
            c5.metric("Mean Recon. Error", "N/A")

    # ---- Price chart for context ----
    df = _load_processed(symbol, timeframe)
    if df is not None:
        st.markdown("#### Price Context")
        if isinstance(df.index, pd.DatetimeIndex):
            timestamps = df.index
        elif "timestamp" in df.columns:
            timestamps = pd.to_datetime(df["timestamp"])
        else:
            timestamps = pd.RangeIndex(len(df))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps, y=df[TARGET_COL],
            mode="lines", name="Close Price",
            line=dict(color="#26a69a", width=1.5),
        ))
        fig.update_layout(height=450, yaxis_title="Price")
        fig = _apply_defaults(fig, f"{symbol} ({timeframe}) — Close Price")
        st.plotly_chart(fig, use_container_width=True)
    elif results is None:
        st.info(
            f"No processed data for **{symbol}** ({timeframe}).  "
            "Run the data pipeline first:\n\n"
            "```bash\npython scripts/run_pipeline.py --quick\n```"
        )
