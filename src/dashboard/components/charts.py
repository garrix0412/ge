"""
Reusable Plotly chart builder functions for the dashboard.

All charts use a dark theme (``plotly_dark``) and consistent styling.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Shared layout defaults
# ---------------------------------------------------------------------------

_LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    margin=dict(l=40, r=20, t=50, b=30),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    font=dict(family="Inter, sans-serif", size=12),
)


def _apply_defaults(fig: go.Figure, title: str) -> go.Figure:
    """Apply shared layout defaults and an optional title."""
    fig.update_layout(**_LAYOUT_DEFAULTS)
    if title:
        fig.update_layout(title=dict(text=title, x=0.5, xanchor="center"))
    return fig


# ---------------------------------------------------------------------------
# Candlestick + Volume
# ---------------------------------------------------------------------------

def create_candlestick(df: pd.DataFrame, title: str = "") -> go.Figure:
    """OHLCV candlestick chart with a volume subplot underneath.

    Parameters
    ----------
    df:
        DataFrame with columns ``open``, ``high``, ``low``, ``close``,
        ``volume`` and a datetime index (or ``timestamp`` column).
    title:
        Chart title.

    Returns
    -------
    go.Figure
    """
    timestamps = df.index if isinstance(df.index, pd.DatetimeIndex) else df.get("timestamp", df.index)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

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

    colors = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(df["close"], df["open"])
    ]

    fig.add_trace(
        go.Bar(
            x=timestamps,
            y=df["volume"],
            marker_color=colors,
            name="Volume",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return _apply_defaults(fig, title)


# ---------------------------------------------------------------------------
# Multi-line chart
# ---------------------------------------------------------------------------

def create_line_chart(df: pd.DataFrame, columns: list[str], title: str = "") -> go.Figure:
    """Multi-line chart for selected *columns* of *df*.

    Parameters
    ----------
    df:
        DataFrame with a datetime index (or ``timestamp`` column).
    columns:
        Column names to plot.
    title:
        Chart title.

    Returns
    -------
    go.Figure
    """
    timestamps = df.index if isinstance(df.index, pd.DatetimeIndex) else df.get("timestamp", df.index)

    fig = go.Figure()
    for col in columns:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=timestamps, y=df[col], mode="lines", name=col))

    return _apply_defaults(fig, title)


# ---------------------------------------------------------------------------
# Prediction chart  (actual vs predicted with optional confidence band)
# ---------------------------------------------------------------------------

def create_prediction_chart(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
    timestamps: np.ndarray | pd.Series | pd.DatetimeIndex,
    title: str = "",
    confidence_upper: Optional[np.ndarray | pd.Series] = None,
    confidence_lower: Optional[np.ndarray | pd.Series] = None,
) -> go.Figure:
    """Actual vs predicted line chart with an optional confidence band.

    Parameters
    ----------
    actual, predicted:
        Arrays of actual and predicted values.
    timestamps:
        Corresponding datetime values.
    title:
        Chart title.
    confidence_upper, confidence_lower:
        Upper and lower bounds for a confidence / prediction interval.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    # Confidence band (filled area between upper and lower)
    if confidence_upper is not None and confidence_lower is not None:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=confidence_upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                name="Upper Bound",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=confidence_lower,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(99, 110, 250, 0.15)",
                name="Confidence Band",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=actual,
            mode="lines",
            name="Actual",
            line=dict(color="#26a69a", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=predicted,
            mode="lines",
            name="Predicted",
            line=dict(color="#ffa726", width=2, dash="dash"),
        )
    )

    fig.update_yaxes(title_text="Price")
    return _apply_defaults(fig, title)


# ---------------------------------------------------------------------------
# Anomaly chart
# ---------------------------------------------------------------------------

def create_anomaly_chart(
    df: pd.DataFrame,
    anomaly_flags: np.ndarray | pd.Series,
    reconstruction_errors: np.ndarray | pd.Series,
    timestamps: np.ndarray | pd.Series | pd.DatetimeIndex,
    title: str = "",
) -> go.Figure:
    """Price line with anomaly points highlighted in red.

    A secondary subplot shows reconstruction errors over time.

    Parameters
    ----------
    df:
        DataFrame containing at least a ``close`` column.
    anomaly_flags:
        Boolean (or 0/1) array – True where an anomaly was detected.
    reconstruction_errors:
        Reconstruction error values (e.g. from an autoencoder).
    timestamps:
        Datetime values aligned with the arrays above.
    title:
        Chart title.

    Returns
    -------
    go.Figure
    """
    anomaly_mask = np.asarray(anomaly_flags).astype(bool)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.6, 0.4],
        subplot_titles=["Price & Anomalies", "Reconstruction Error"],
    )

    # Price line
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=df["close"],
            mode="lines",
            name="Close Price",
            line=dict(color="#26a69a", width=1.5),
        ),
        row=1,
        col=1,
    )

    # Anomaly markers
    if anomaly_mask.any():
        anom_ts = np.asarray(timestamps)[anomaly_mask]
        anom_price = np.asarray(df["close"])[anomaly_mask]
        fig.add_trace(
            go.Scatter(
                x=anom_ts,
                y=anom_price,
                mode="markers",
                name="Anomaly",
                marker=dict(color="#ef5350", size=8, symbol="x"),
            ),
            row=1,
            col=1,
        )

    # Reconstruction error
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=reconstruction_errors,
            mode="lines",
            name="Recon Error",
            line=dict(color="#ffa726", width=1.5),
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Error", row=2, col=1)

    return _apply_defaults(fig, title)


# ---------------------------------------------------------------------------
# Metrics bar chart
# ---------------------------------------------------------------------------

def create_metrics_bar_chart(metrics_dict: dict[str, dict[str, float]], title: str = "") -> go.Figure:
    """Grouped bar chart comparing models by each metric.

    Parameters
    ----------
    metrics_dict:
        ``{model_name: {metric_name: value, ...}, ...}``
    title:
        Chart title.

    Returns
    -------
    go.Figure
    """
    if not metrics_dict:
        fig = go.Figure()
        fig.add_annotation(text="No metrics available", showarrow=False, font=dict(size=16))
        return _apply_defaults(fig, title)

    model_names = list(metrics_dict.keys())
    # Collect the union of metric names
    all_metrics: set[str] = set()
    for v in metrics_dict.values():
        all_metrics.update(v.keys())
    metric_names = sorted(all_metrics)

    fig = go.Figure()
    for metric in metric_names:
        values = [metrics_dict[m].get(metric, 0) for m in model_names]
        fig.add_trace(go.Bar(name=metric, x=model_names, y=values))

    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Value")
    fig.update_xaxes(title_text="Model")

    return _apply_defaults(fig, title)


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------

def create_correlation_heatmap(df: pd.DataFrame, title: str = "") -> go.Figure:
    """Feature correlation heatmap.

    Parameters
    ----------
    df:
        DataFrame whose numeric columns will be correlated.
    title:
        Chart title.

    Returns
    -------
    go.Figure
    """
    corr = df.select_dtypes(include="number").corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Corr"),
        )
    )
    fig.update_layout(height=max(400, 20 * len(corr.columns)))

    return _apply_defaults(fig, title)


# ---------------------------------------------------------------------------
# Attention heatmap
# ---------------------------------------------------------------------------

def create_attention_heatmap(
    weights: np.ndarray,
    feature_names: list[str],
    title: str = "",
) -> go.Figure:
    """Attention weight visualisation.

    Parameters
    ----------
    weights:
        2-D array of shape ``(time_steps, features)`` (or similar).
    feature_names:
        Names for the feature axis.
    title:
        Chart title.

    Returns
    -------
    go.Figure
    """
    weights = np.atleast_2d(weights)
    y_labels = [f"Step {i}" for i in range(weights.shape[0])]

    fig = go.Figure(
        data=go.Heatmap(
            z=weights,
            x=feature_names,
            y=y_labels,
            colorscale="Viridis",
            colorbar=dict(title="Weight"),
        )
    )
    fig.update_layout(height=max(350, 20 * weights.shape[0]))

    return _apply_defaults(fig, title)
