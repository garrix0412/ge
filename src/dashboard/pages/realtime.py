"""
Market Overview page.

Displays candlestick OHLCV data, technical indicators (EMA, Bollinger Bands),
RSI, MACD sub-plots, and key statistics cards.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.dashboard.components.charts import _apply_defaults
from src.dashboard.components.sidebar import render_sidebar
from src.utils.constants import PROCESSED_DIR, PROCESSED_DATA_PATTERN

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=120, show_spinner="Loading market data ...")
def _load_market_data(symbol: str, timeframe: str) -> pd.DataFrame | None:
    """Load processed parquet data for *symbol* / *timeframe*."""
    filename = PROCESSED_DATA_PATTERN.format(
        symbol=symbol.replace("/", "_"),
        timeframe=timeframe,
    )
    path = PROCESSED_DIR / filename
    if not path.exists():
        return None

    from src.utils.io import load_dataframe

    return load_dataframe(path)


def _get_data_date_range(df: pd.DataFrame) -> tuple[date, date]:
    """Return (min_date, max_date) from the dataframe index."""
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    else:
        idx = pd.DatetimeIndex(pd.to_datetime(df.get("timestamp", df.index)))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.min().date(), idx.max().date()


def _filter_by_dates(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Narrow *df* to the selected date window."""
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    else:
        idx = pd.DatetimeIndex(pd.to_datetime(df.get("timestamp", df.index)))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end) + pd.Timedelta(days=1))
    return df.loc[mask]


def _build_indicator_chart(
    df: pd.DataFrame,
    show_ema: bool,
    show_bb: bool,
) -> go.Figure:
    """Candlestick + optional EMA / Bollinger Band overlays, with RSI and MACD subplots."""
    if isinstance(df.index, pd.DatetimeIndex):
        timestamps = df.index
    elif "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"])
    else:
        timestamps = df.index

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.45, 0.15, 0.2, 0.2],
        subplot_titles=["", "Volume", "RSI", "MACD"],
    )

    # -- Candlestick --
    fig.add_trace(
        go.Candlestick(
            x=timestamps,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1,
        col=1,
    )

    # -- EMA overlays --
    if show_ema:
        for col, color in [("ema_7", "#42a5f5"), ("ema_25", "#ffa726"), ("ema_99", "#ab47bc")]:
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(x=timestamps, y=df[col], mode="lines", name=col.upper(), line=dict(color=color, width=1)),
                    row=1,
                    col=1,
                )

    # -- Bollinger Bands --
    if show_bb:
        for col in ["bb_upper", "bb_middle", "bb_lower"]:
            if col in df.columns:
                dash_style = "dot" if col != "bb_middle" else "dash"
                fig.add_trace(
                    go.Scatter(x=timestamps, y=df[col], mode="lines", name=col.replace("bb_", "BB ").title(), line=dict(width=1, dash=dash_style, color="#78909c")),
                    row=1,
                    col=1,
                )

    # -- Volume --
    colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["close"], df["open"])]
    fig.add_trace(
        go.Bar(x=timestamps, y=df["volume"], marker_color=colors, name="Volume", showlegend=False),
        row=2,
        col=1,
    )

    # -- RSI --
    if "rsi" in df.columns:
        fig.add_trace(
            go.Scatter(x=timestamps, y=df["rsi"], mode="lines", name="RSI", line=dict(color="#42a5f5", width=1.5)),
            row=3,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)

    # -- MACD --
    if "macd" in df.columns:
        fig.add_trace(
            go.Scatter(x=timestamps, y=df["macd"], mode="lines", name="MACD", line=dict(color="#42a5f5", width=1.5)),
            row=4,
            col=1,
        )
    if "macd_signal" in df.columns:
        fig.add_trace(
            go.Scatter(x=timestamps, y=df["macd_signal"], mode="lines", name="Signal", line=dict(color="#ffa726", width=1.5)),
            row=4,
            col=1,
        )
    if "macd_hist" in df.columns:
        hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["macd_hist"]]
        fig.add_trace(
            go.Bar(x=timestamps, y=df["macd_hist"], marker_color=hist_colors, name="Histogram", showlegend=False),
            row=4,
            col=1,
        )

    fig.update_layout(xaxis_rangeslider_visible=False, height=900)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)

    return _apply_defaults(fig, "")


# ---------------------------------------------------------------------------
# Page renderer
# ---------------------------------------------------------------------------


def render() -> None:
    """Main entry point for the Market Overview page."""

    st.header("Market Overview")
    st.caption("Live OHLCV data with technical indicators (EMA, Bollinger Bands, RSI, MACD).")

    # Load data first to determine date range before rendering sidebar
    # We need a default symbol/timeframe to bootstrap
    from src.utils.config import load_config

    cfg = load_config()
    _default_symbol = cfg.dashboard.defaults.symbol
    _default_tf = cfg.dashboard.defaults.timeframe

    # Pre-load to get date bounds
    df_probe = _load_market_data(_default_symbol, _default_tf)
    data_range = None
    if df_probe is not None:
        data_range = _get_data_date_range(df_probe)

    selections = render_sidebar(page="market", data_date_range=data_range)

    symbol: str = selections["symbol"]
    timeframe: str = selections["timeframe"]

    df = _load_market_data(symbol, timeframe)

    if df is None:
        st.info(
            f"No processed data found for **{symbol}** ({timeframe}).  "
            "Run the data pipeline first:\n\n"
            "```bash\npython scripts/run_pipeline.py --quick\n```"
        )
        return

    # Clamp date filter to actual data bounds
    actual_range = _get_data_date_range(df)
    sel_start = max(selections["date_start"], actual_range[0])
    sel_end = min(selections["date_end"], actual_range[1])

    df = _filter_by_dates(df, sel_start, sel_end)

    if df.empty:
        st.warning("No data in the selected date range.")
        return

    # ---- Key statistics cards ----
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    price = latest["close"]
    change = ((price - prev["close"]) / prev["close"]) * 100 if prev["close"] else 0.0
    vol = latest["volume"]
    period_high = df["high"].max()
    period_low = df["low"].min()

    # RSI status
    rsi_val = latest.get("rsi", None)
    if rsi_val is not None:
        if rsi_val > 70:
            rsi_status = "Overbought"
        elif rsi_val < 30:
            rsi_status = "Oversold"
        else:
            rsi_status = "Neutral"

    col1, col2, col3 = st.columns(3)
    col1.metric("Symbol", symbol)
    col2.metric("Current Price", f"${price:,.2f}", delta=f"{change:+.2f}%")
    col3.metric("Volume", f"{vol:,.0f}")
    col4, col5, col6 = st.columns(3)
    col4.metric("Period High", f"${period_high:,.2f}")
    col5.metric("Period Low", f"${period_low:,.2f}")
    if rsi_val is not None:
        col6.metric(f"RSI ({rsi_status})", f"{rsi_val:.1f}")
    else:
        col6.metric("Data Points", f"{len(df):,}")

    # ---- Indicator toggles ----
    tcol1, tcol2 = st.columns(2)
    show_ema = tcol1.checkbox("Show EMA (7 / 25 / 99)", value=True)
    show_bb = tcol2.checkbox("Show Bollinger Bands", value=True)

    # ---- Main chart ----
    chart = _build_indicator_chart(df, show_ema=show_ema, show_bb=show_bb)
    st.plotly_chart(chart, use_container_width=True)
