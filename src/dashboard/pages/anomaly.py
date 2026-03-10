"""
Anomaly Detection page.

Displays autoencoder-based anomaly detection results: a price timeline with
anomaly markers, reconstruction error chart, threshold line, alert cards,
and summary statistics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.components.charts import create_anomaly_chart, _apply_defaults
from src.dashboard.components.sidebar import render_sidebar
from src.utils.constants import (
    METRICS_DIR,
    PROCESSED_DIR,
    PROCESSED_DATA_PATTERN,
    TARGET_COL,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ANOMALY_RESULTS_PATTERN = "anomaly_{symbol}_{timeframe}_results.json"


@st.cache_data(ttl=120, show_spinner="Loading processed data ...")
def _load_processed(symbol: str, timeframe: str) -> pd.DataFrame | None:
    filename = PROCESSED_DATA_PATTERN.format(
        symbol=symbol.replace("/", ""),
        timeframe=timeframe,
    )
    path = PROCESSED_DIR / filename
    if not path.exists():
        return None
    from src.utils.io import load_dataframe
    return load_dataframe(path)


@st.cache_data(ttl=300, show_spinner="Loading anomaly results ...")
def _load_anomaly_results(symbol: str, timeframe: str) -> dict[str, Any] | None:
    """Load anomaly detection results JSON produced by the anomaly pipeline.

    Expected keys: ``anomaly_flags``, ``reconstruction_errors``, ``threshold``,
    ``timestamps`` (optional).
    """
    filename = _ANOMALY_RESULTS_PATTERN.format(
        symbol=symbol.replace("/", ""),
        timeframe=timeframe,
    )
    # Try both METRICS_DIR and RESULTS_DIR parent
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

    selections = render_sidebar()
    symbol: str = selections["symbol"]
    timeframe: str = selections["timeframe"]

    # ---- Load data ----
    df = _load_processed(symbol, timeframe)
    if df is None:
        st.info(
            f"No processed data for **{symbol}** ({timeframe}).  "
            "Run the data pipeline first:\n\n"
            "```bash\npython -m src.data.fetch\npython -m src.data.process\n```"
        )
        return

    # ---- Load anomaly results ----
    results = _load_anomaly_results(symbol, timeframe)
    if results is None:
        st.warning(
            f"No anomaly detection results found for **{symbol}** ({timeframe}).  \n"
            "Run the anomaly detection pipeline first:\n\n"
            "```bash\npython -m src.training.train --model anomaly "
            f"--symbol {symbol} --timeframe {timeframe}\n```"
        )
        _show_placeholder(df)
        return

    # ---- Parse results ----
    anomaly_flags = np.array(results.get("anomaly_flags", []))
    recon_errors = np.array(results.get("reconstruction_errors", []))
    threshold = results.get("threshold", None)

    # Align lengths – use tail of df to match result arrays
    n = min(len(anomaly_flags), len(recon_errors), len(df))
    if n == 0:
        st.warning("Anomaly results are empty.")
        _show_placeholder(df)
        return

    df_aligned = df.iloc[-n:].copy()
    anomaly_flags = anomaly_flags[-n:]
    recon_errors = recon_errors[-n:]

    timestamps = (
        df_aligned.index
        if isinstance(df_aligned.index, pd.DatetimeIndex)
        else pd.RangeIndex(n)
    )

    # ---- Summary statistics ----
    total_anomalies = int(anomaly_flags.sum())
    total_points = len(anomaly_flags)
    anomaly_rate = (total_anomalies / total_points * 100) if total_points else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Data Points", f"{total_points:,}")
    col2.metric("Anomalies Detected", f"{total_anomalies:,}")
    col3.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
    if threshold is not None:
        col4.metric("Threshold", f"{threshold:.6f}")
    else:
        col4.metric("Threshold", "N/A")

    # ---- Anomaly chart (price + markers) ----
    st.subheader("Price with Anomaly Markers")
    fig = create_anomaly_chart(
        df_aligned,
        anomaly_flags,
        recon_errors,
        timestamps,
        title=f"Anomaly Detection - {symbol} ({timeframe})",
    )

    # Add threshold line to the reconstruction error subplot
    if threshold is not None:
        import plotly.graph_objects as go

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=[threshold] * len(timestamps),
                mode="lines",
                name="Threshold",
                line=dict(color="#ef5350", width=1.5, dash="dash"),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)

    # ---- Recent anomaly alert cards ----
    st.subheader("Recent Anomaly Alerts")
    _render_anomaly_alerts(df_aligned, anomaly_flags, recon_errors, timestamps, threshold)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _show_placeholder(df: pd.DataFrame) -> None:
    """Show a simple price chart when anomaly results are unavailable."""
    st.subheader("Historical Close Price")
    timestamps = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.RangeIndex(len(df))
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timestamps, y=df[TARGET_COL], mode="lines", name="Close"))
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)


def _render_anomaly_alerts(
    df: pd.DataFrame,
    anomaly_flags: np.ndarray,
    recon_errors: np.ndarray,
    timestamps,
    threshold,
    max_alerts: int = 10,
) -> None:
    """Show the most recent anomalies as styled alert cards."""
    mask = anomaly_flags.astype(bool)
    if not mask.any():
        st.success("No anomalies detected in the current data window.")
        return

    # Gather anomaly details
    anom_indices = np.where(mask)[0]
    # Show the most recent ones first
    anom_indices = anom_indices[::-1][:max_alerts]

    for idx in anom_indices:
        ts = timestamps[idx] if hasattr(timestamps, "__getitem__") else f"Index {idx}"
        price = df.iloc[idx][TARGET_COL]
        error = recon_errors[idx]
        severity = "High" if (threshold and error > threshold * 1.5) else "Medium"

        color = "#ef5350" if severity == "High" else "#ffa726"

        st.markdown(
            f"""
            <div style="border-left: 4px solid {color}; padding: 8px 12px; margin-bottom: 8px;
                        background-color: rgba(255,255,255,0.03); border-radius: 4px;">
                <strong>{ts}</strong> &nbsp;|&nbsp;
                Price: <code>${price:,.2f}</code> &nbsp;|&nbsp;
                Recon Error: <code>{error:.6f}</code> &nbsp;|&nbsp;
                Severity: <span style="color:{color}; font-weight:bold;">{severity}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
