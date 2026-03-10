"""
Price Prediction page.

Loads a model checkpoint, runs inference on processed data, and displays
actual vs predicted prices, confidence intervals, metrics, and a recent
predictions table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.components.charts import create_prediction_chart
from src.dashboard.components.sidebar import render_sidebar
from src.utils.constants import (
    CHECKPOINTS_DIR,
    METRICS_DIR,
    PROCESSED_DIR,
    SCALERS_DIR,
    FEATURE_COLUMNS,
    TARGET_COL,
    MODEL_FILE_PATTERN,
    METRICS_FILE_PATTERN,
    PROCESSED_DATA_PATTERN,
    SCALER_FILE_PATTERN,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


@st.cache_data(ttl=300, show_spinner="Loading metrics ...")
def _load_model_metrics(model_name: str, symbol: str, timeframe: str) -> dict[str, Any] | None:
    filename = METRICS_FILE_PATTERN.format(
        model_name=model_name,
        symbol=symbol.replace("/", ""),
        timeframe=timeframe,
    )
    path = METRICS_DIR / filename
    if not path.exists():
        return None
    from src.utils.io import load_metrics
    return load_metrics(path)


@st.cache_resource(show_spinner="Loading model checkpoint ...")
def _load_checkpoint(model_name: str, symbol: str, timeframe: str):
    """Attempt to load a serialized full checkpoint (``torch.load``).

    Returns the checkpoint dict or ``None`` if the file is missing.
    """
    filename = MODEL_FILE_PATTERN.format(
        model_name=model_name,
        symbol=symbol.replace("/", ""),
        timeframe=timeframe,
    )
    path = CHECKPOINTS_DIR / filename
    if not path.exists():
        return None
    try:
        import torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return checkpoint
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading scaler ...")
def _load_scaler(symbol: str, timeframe: str):
    filename = SCALER_FILE_PATTERN.format(
        symbol=symbol.replace("/", ""),
        timeframe=timeframe,
    )
    path = SCALERS_DIR / filename
    if not path.exists():
        return None
    from src.utils.io import load_scaler
    return load_scaler(path)


def _run_inference(checkpoint, scaler, df: pd.DataFrame, lookback: int, horizon: int):
    """Run model inference and return predicted values.

    This is a best-effort utility. Because the exact model architecture is
    resolved at training time, we attempt a simple forward pass.  If anything
    goes wrong we return ``None`` so the page can show a graceful message.
    """
    try:
        import torch

        model = checkpoint if hasattr(checkpoint, "forward") else None
        if model is None:
            return None, None, None

        model.eval()

        # Prepare features
        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        data = df[feature_cols].dropna().values

        if scaler is not None:
            data = scaler.transform(data)

        if len(data) < lookback:
            return None, None, None

        # Create input sequence from the last `lookback` rows
        x = torch.tensor(data[-lookback:], dtype=torch.float32).unsqueeze(0)  # (1, seq, feat)

        with torch.no_grad():
            output = model(x)

        predicted = output.squeeze().cpu().numpy()

        # If the model returns multiple horizons, slice to requested horizon
        if isinstance(predicted, np.ndarray) and predicted.ndim >= 1:
            predicted = predicted[:horizon] if len(predicted) >= horizon else predicted

        return predicted, None, None  # predicted, upper, lower
    except Exception:
        return None, None, None


# ---------------------------------------------------------------------------
# Page renderer
# ---------------------------------------------------------------------------


def render() -> None:
    """Main entry point for the Price Prediction page."""

    st.header("Price Prediction")

    selections = render_sidebar()
    symbol: str = selections["symbol"]
    model_name: str = selections["model"]
    timeframe: str = selections["timeframe"]
    lookback: int = selections["lookback_window"]
    horizon: int = selections["forecast_horizon"]

    # ---- Load processed data ----
    df = _load_processed(symbol, timeframe)
    if df is None:
        st.info(
            f"No processed data for **{symbol}** ({timeframe}).  "
            "Run the data pipeline first:\n\n"
            "```bash\npython -m src.data.fetch\npython -m src.data.process\n```"
        )
        return

    # ---- Load checkpoint ----
    checkpoint = _load_checkpoint(model_name, symbol, timeframe)
    if checkpoint is None:
        st.warning(
            f"No model checkpoint found for **{model_name}** on {symbol} ({timeframe}).  \n"
            "Train the model first:\n\n"
            f"```bash\npython -m src.training.train --model {model_name} --symbol {symbol} --timeframe {timeframe}\n```"
        )
        _show_placeholder(df, symbol, model_name)
        return

    # ---- Load scaler ----
    scaler = _load_scaler(symbol, timeframe)

    # ---- Run inference ----
    predicted, conf_upper, conf_lower = _run_inference(checkpoint, scaler, df, lookback, horizon)

    if predicted is None:
        st.warning(
            "Could not run inference with the loaded checkpoint. "
            "The checkpoint may be a state_dict rather than a full model. "
            "Showing historical metrics only."
        )
        _show_placeholder(df, symbol, model_name)
        return

    # ---- Display results ----
    _show_predictions(df, predicted, conf_upper, conf_lower, symbol, model_name, timeframe, lookback, horizon)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _show_placeholder(df: pd.DataFrame, symbol: str, model_name: str) -> None:
    """Show metrics and a simple price chart when inference is unavailable."""

    st.subheader("Historical Close Price")
    timestamps = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.get("timestamp", df.index))
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timestamps, y=df[TARGET_COL], mode="lines", name="Close"))
    fig.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Show saved metrics if available
    for tf in ["1h", "4h"]:
        metrics = _load_model_metrics(model_name, symbol, tf)
        if metrics:
            st.subheader(f"Saved Metrics ({model_name} / {tf})")
            _render_metrics_cards(metrics)
            break


def _show_predictions(
    df: pd.DataFrame,
    predicted: np.ndarray,
    conf_upper,
    conf_lower,
    symbol: str,
    model_name: str,
    timeframe: str,
    lookback: int,
    horizon: int,
) -> None:
    """Render the full prediction view."""

    # Build timestamps for forecasted steps
    last_ts = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else pd.Timestamp.now()
    freq = "1h" if timeframe == "1h" else "4h"
    future_ts = pd.date_range(start=last_ts, periods=len(predicted) + 1, freq=freq)[1:]

    # ---- Metrics cards (from saved JSON) ----
    metrics = _load_model_metrics(model_name, symbol, timeframe)
    if metrics:
        st.subheader("Model Performance")
        _render_metrics_cards(metrics)

    # ---- Prediction chart (historical tail + forecast) ----
    st.subheader("Actual vs Predicted")

    tail_len = min(lookback * 2, len(df))
    tail_df = df.iloc[-tail_len:]
    actual_ts = tail_df.index if isinstance(tail_df.index, pd.DatetimeIndex) else pd.RangeIndex(len(tail_df))
    actual_vals = tail_df[TARGET_COL].values

    # Combine historical actual with forecast placeholder
    combined_ts = list(actual_ts) + list(future_ts)
    combined_actual = list(actual_vals) + [np.nan] * len(predicted)
    combined_pred = [np.nan] * len(actual_vals) + list(predicted)

    fig = create_prediction_chart(
        actual=combined_actual,
        predicted=combined_pred,
        timestamps=combined_ts,
        title=f"{model_name.upper()} Forecast - {symbol} ({timeframe})",
        confidence_upper=conf_upper,
        confidence_lower=conf_lower,
    )
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)

    # ---- Forecast table ----
    st.subheader(f"Next {len(predicted)} Hour(s) Forecast")
    forecast_df = pd.DataFrame({
        "Timestamp": future_ts,
        "Predicted Price": predicted,
    })
    if conf_upper is not None:
        forecast_df["Upper Bound"] = conf_upper
    if conf_lower is not None:
        forecast_df["Lower Bound"] = conf_lower
    st.dataframe(forecast_df, use_container_width=True)


def _render_metrics_cards(metrics: dict[str, Any]) -> None:
    """Render metrics as Streamlit metric cards in columns."""
    display_metrics = {
        "MAE": metrics.get("mae"),
        "RMSE": metrics.get("rmse"),
        "MAPE": metrics.get("mape"),
        "Directional Acc.": metrics.get("directional_accuracy") or metrics.get("direction_accuracy"),
    }
    # Filter out None values
    display_metrics = {k: v for k, v in display_metrics.items() if v is not None}

    if not display_metrics:
        st.caption("No standard metrics (MAE / RMSE / MAPE / Dir. Acc.) found in the saved file.")
        return

    cols = st.columns(len(display_metrics))
    for col, (name, value) in zip(cols, display_metrics.items()):
        fmt = f"{value:.4f}" if isinstance(value, float) else str(value)
        col.metric(name, fmt)
