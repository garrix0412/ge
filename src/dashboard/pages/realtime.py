"""
Real-Time Market page.

Displays candlestick OHLCV data, technical indicators (EMA, Bollinger Bands),
RSI, MACD sub-plots, and key statistics cards.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.dashboard.components.charts import _apply_defaults, create_candlestick
from src.dashboard.components.sidebar import render_sidebar
from src.utils.constants import PROCESSED_DIR, PROCESSED_DATA_PATTERN

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=120, show_spinner="Loading market data ...")
def _load_market_data(symbol: str, timeframe: str) -> pd.DataFrame | None:
    """Load processed parquet data for *symbol* / *timeframe*.

    Returns ``None`` when the file does not exist.
    """
    filename = PROCESSED_DATA_PATTERN.format(
        symbol=symbol.replace("/", ""),
        timeframe=timeframe,
    )
    path = PROCESSED_DIR / filename
    if not path.exists():
        return None

    from src.utils.io import load_dataframe

    df = load_dataframe(path)
    return df


def _filter_by_dates(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Narrow *df* to the selected date window."""
    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.get("timestamp", df.index))
    mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end) + pd.Timedelta(days=1))
    return df.loc[mask]


def _build_indicator_chart(
    df: pd.DataFrame,
    show_ema: bool,
    show_bb: bool,
) -> go.Figure:
    """Candlestick + optional EMA / Bollinger Band overlays, with RSI and MACD subplots."""
    timestamps = df.index if isinstance(df.index, pd.DatetimeIndex) else df.get("timestamp", df.index)

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
    """Main entry point for the Real-Time Market page."""

    st.header("Real-Time Market")

    selections = render_sidebar()

    symbol: str = selections["symbol"]
    timeframe: str = selections["timeframe"]

    df = _load_market_data(symbol, timeframe)

    if df is None:
        st.info(
            f"No processed data found for **{symbol}** ({timeframe}).  "
            "Run the data pipeline first:\n\n"
            "```bash\npython -m src.data.fetch\npython -m src.data.process\n```"
        )
        return

    # Filter by date range
    df = _filter_by_dates(df, selections["date_start"], selections["date_end"])

    if df.empty:
        st.warning("No data in the selected date range.")
        return

    # ---- Key statistics cards ----
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    price = latest["close"]
    change_24h = ((price - prev["close"]) / prev["close"]) * 100 if prev["close"] else 0.0
    vol = latest["volume"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Symbol", symbol)
    col2.metric("Current Price", f"${price:,.2f}", delta=f"{change_24h:+.2f}%")
    col3.metric("24h Volume", f"{vol:,.0f}")
    col4.metric("Data Points", f"{len(df):,}")

    # ---- Indicator toggles ----
    tcol1, tcol2 = st.columns(2)
    show_ema = tcol1.checkbox("Show EMA (7 / 25 / 99)", value=True)
    show_bb = tcol2.checkbox("Show Bollinger Bands", value=True)

    # ---- Main chart ----
    chart = _build_indicator_chart(df, show_ema=show_ema, show_bb=show_bb)
    st.plotly_chart(chart, use_container_width=True)

    # ---- Auto-refresh ----
    if selections["auto_refresh"]:
        from src.utils.config import load_config

        cfg = load_config()
        st.toast(f"Auto-refresh every {cfg.dashboard.refresh_interval_seconds}s")
        import time

        time.sleep(cfg.dashboard.refresh_interval_seconds)
        st.rerun()
