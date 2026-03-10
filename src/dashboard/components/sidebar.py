"""
Shared sidebar controls rendered on every dashboard page.

Usage::

    from src.dashboard.components.sidebar import render_sidebar

    selections = render_sidebar()
    symbol = selections["symbol"]
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import streamlit as st

from src.utils.config import load_config

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_sidebar() -> dict[str, Any]:
    """Render sidebar widgets and return a dict of all user selections.

    Returns
    -------
    dict
        Keys: ``symbol``, ``model``, ``timeframe``, ``lookback_window``,
        ``forecast_horizon``, ``date_start``, ``date_end``, ``auto_refresh``.
    """
    cfg = load_config()
    dash = cfg.dashboard

    st.sidebar.header("Controls")

    # -- Symbol --
    symbol = st.sidebar.selectbox(
        "Symbol",
        options=dash.available_symbols,
        index=dash.available_symbols.index(dash.defaults.symbol)
        if dash.defaults.symbol in dash.available_symbols
        else 0,
    )

    # -- Model --
    model = st.sidebar.selectbox(
        "Model",
        options=dash.available_models,
        index=dash.available_models.index(dash.defaults.model)
        if dash.defaults.model in dash.available_models
        else 0,
    )

    # -- Timeframe --
    timeframe_options = ["1h", "4h"]
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        options=timeframe_options,
        index=timeframe_options.index(dash.defaults.timeframe)
        if dash.defaults.timeframe in timeframe_options
        else 0,
    )

    # -- Lookback window --
    lookback_options = [24, 48, 96]
    lookback_window = st.sidebar.select_slider(
        "Lookback Window (hours)",
        options=lookback_options,
        value=dash.defaults.lookback_window
        if dash.defaults.lookback_window in lookback_options
        else lookback_options[1],
    )

    # -- Forecast horizon --
    horizon_options = [1, 4, 12, 24]
    forecast_horizon = st.sidebar.select_slider(
        "Forecast Horizon (hours)",
        options=horizon_options,
        value=dash.defaults.forecast_horizon
        if dash.defaults.forecast_horizon in horizon_options
        else horizon_options[1],
    )

    # -- Date range --
    st.sidebar.subheader("Date Range")
    default_end = date.today()
    default_start = default_end - timedelta(days=90)

    date_start = st.sidebar.date_input("Start Date", value=default_start)
    date_end = st.sidebar.date_input("End Date", value=default_end)

    # -- Auto-refresh --
    auto_refresh = st.sidebar.toggle(
        "Auto-refresh",
        value=False,
        help=f"Refresh every {dash.refresh_interval_seconds}s",
    )

    return {
        "symbol": symbol,
        "model": model,
        "timeframe": timeframe,
        "lookback_window": lookback_window,
        "forecast_horizon": forecast_horizon,
        "date_start": date_start,
        "date_end": date_end,
        "auto_refresh": auto_refresh,
    }
