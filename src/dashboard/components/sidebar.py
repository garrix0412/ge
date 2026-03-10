"""
Shared sidebar controls rendered on every dashboard page.

Usage::

    from src.dashboard.components.sidebar import render_sidebar

    selections = render_sidebar(page="market")
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


def render_sidebar(
    page: str = "all",
    data_date_range: tuple[date, date] | None = None,
) -> dict[str, Any]:
    """Render sidebar widgets and return a dict of all user selections.

    Parameters
    ----------
    page
        Context hint controlling which controls appear:
        ``"market"`` – Symbol, Timeframe, Date Range only
        ``"prediction"`` – Symbol, Model, Timeframe, Lookback, Horizon
        ``"anomaly"`` – Symbol, Timeframe, Lookback, Horizon
        ``"comparison"`` – minimal (comparison has its own filters)
        ``"all"`` – everything
    data_date_range
        Optional (min_date, max_date) tuple from actual data to clamp the
        date range picker. If *None*, defaults to the last 90 days.

    Returns
    -------
    dict
        Keys: ``symbol``, ``model``, ``timeframe``, ``lookback_window``,
        ``forecast_horizon``, ``date_start``, ``date_end``.
    """
    cfg = load_config()
    dash = cfg.dashboard

    st.sidebar.header("Controls")

    # -- Symbol (always shown except comparison) --
    symbol = dash.defaults.symbol
    if page != "comparison":
        symbol = st.sidebar.selectbox(
            "Symbol",
            options=dash.available_symbols,
            index=dash.available_symbols.index(dash.defaults.symbol)
            if dash.defaults.symbol in dash.available_symbols
            else 0,
        )

    # -- Model (only for prediction and "all") --
    model = dash.defaults.model
    if page in ("prediction", "all"):
        model = st.sidebar.selectbox(
            "Model",
            options=dash.available_models,
            index=dash.available_models.index(dash.defaults.model)
            if dash.defaults.model in dash.available_models
            else 0,
        )

    # -- Timeframe --
    timeframe_options = ["1h", "4h"]
    timeframe = dash.defaults.timeframe
    if page != "comparison":
        timeframe = st.sidebar.selectbox(
            "Timeframe",
            options=timeframe_options,
            index=timeframe_options.index(dash.defaults.timeframe)
            if dash.defaults.timeframe in timeframe_options
            else 0,
        )

    # -- Lookback window (prediction, anomaly, all) --
    lookback_options = [24, 48, 96]
    lookback_window = dash.defaults.lookback_window
    if lookback_window not in lookback_options:
        lookback_window = lookback_options[1]
    if page in ("prediction", "anomaly", "all"):
        lookback_window = st.sidebar.select_slider(
            "Lookback Window (hours)",
            options=lookback_options,
            value=lookback_window,
        )

    # -- Forecast horizon (prediction, anomaly, all) --
    horizon_options = [1, 4, 12, 24]
    forecast_horizon = dash.defaults.forecast_horizon
    if forecast_horizon not in horizon_options:
        forecast_horizon = horizon_options[1]
    if page in ("prediction", "anomaly", "all"):
        forecast_horizon = st.sidebar.select_slider(
            "Forecast Horizon (hours)",
            options=horizon_options,
            value=forecast_horizon,
        )

    # -- Date range (market and "all" only) --
    date_start = date(2024, 1, 1)
    date_end = date(2024, 12, 31)
    if page in ("market", "all"):
        st.sidebar.subheader("Date Range")
        if data_date_range is not None:
            default_start, default_end = data_date_range
        else:
            default_end = date(2024, 12, 31)
            default_start = default_end - timedelta(days=90)

        date_start = st.sidebar.date_input("Start Date", value=default_start)
        date_end = st.sidebar.date_input("End Date", value=default_end)

    return {
        "symbol": symbol,
        "model": model,
        "timeframe": timeframe,
        "lookback_window": lookback_window,
        "forecast_horizon": forecast_horizon,
        "date_start": date_start,
        "date_end": date_end,
    }
