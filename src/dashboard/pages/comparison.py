"""
Model Comparison page.

Loads all metrics JSON files from ``results/metrics/``, shows a sortable
comparison table, grouped bar chart, radar chart, and highlights the best
model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.components.charts import create_metrics_bar_chart, _apply_defaults
from src.dashboard.components.sidebar import render_sidebar
from src.utils.constants import METRICS_DIR

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner="Scanning metrics directory ...")
def _load_all_metrics() -> pd.DataFrame:
    """Read every ``*_metrics.json`` file in METRICS_DIR into a DataFrame.

    Returns an empty DataFrame if the directory is missing or has no files.
    """
    if not METRICS_DIR.exists():
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for path in sorted(METRICS_DIR.glob("*_metrics.json")):
        try:
            with open(path, "r") as fh:
                data: dict[str, Any] = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue

        # Try to infer model / symbol / timeframe from the filename
        # Expected pattern: {model}_{symbol}_{timeframe}_metrics.json
        stem = path.stem.replace("_metrics", "")
        parts = stem.rsplit("_", 2)
        if len(parts) >= 3:
            model_name, symbol_part, tf = parts[-3], parts[-2], parts[-1]
        elif len(parts) == 2:
            model_name, symbol_part, tf = parts[0], parts[1], ""
        else:
            model_name, symbol_part, tf = stem, "", ""

        record = {
            "model": data.get("model", model_name),
            "symbol": data.get("symbol", symbol_part),
            "timeframe": data.get("timeframe", tf),
            "file": path.name,
        }

        # Flatten numeric metrics into the record
        for k, v in data.items():
            if isinstance(v, (int, float)):
                record[k] = v

        records.append(record)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def _metric_columns(df: pd.DataFrame) -> list[str]:
    """Return column names that contain numeric metric values."""
    skip = {"model", "symbol", "timeframe", "file"}
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


# ---------------------------------------------------------------------------
# Radar chart
# ---------------------------------------------------------------------------

def _create_radar_chart(df: pd.DataFrame, metric_cols: list[str], title: str = "") -> go.Figure:
    """Radar (polar) chart showing each model's metrics normalised to [0, 1].

    For error metrics (lower is better) the values are inverted so that
    larger area = better model.
    """
    if df.empty or not metric_cols:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data for radar chart", showarrow=False, font=dict(size=14))
        return _apply_defaults(fig, title)

    # Normalise each metric to [0, 1]; invert error-like metrics
    error_keywords = {"mae", "rmse", "mse", "mape", "loss", "error"}
    normed = df[metric_cols].copy()
    for col in metric_cols:
        cmin = normed[col].min()
        cmax = normed[col].max()
        rng = cmax - cmin if cmax != cmin else 1.0
        normed[col] = (normed[col] - cmin) / rng
        if any(kw in col.lower() for kw in error_keywords):
            normed[col] = 1.0 - normed[col]

    fig = go.Figure()
    for i, row in normed.iterrows():
        values = row[metric_cols].tolist()
        values.append(values[0])  # close the polygon
        angles = metric_cols + [metric_cols[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=angles,
                fill="toself",
                name=df.loc[i, "model"] if "model" in df.columns else f"Model {i}",
                opacity=0.6,
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
    )
    return _apply_defaults(fig, title)


# ---------------------------------------------------------------------------
# Page renderer
# ---------------------------------------------------------------------------


def render() -> None:
    """Main entry point for the Model Comparison page."""

    st.header("Model Comparison")

    selections = render_sidebar()

    # ---- Load all metrics ----
    all_metrics = _load_all_metrics()

    if all_metrics.empty:
        st.info(
            "No metrics files found in the results directory.  \n"
            "Train models first to generate metrics:\n\n"
            "```bash\npython -m src.training.train --model lstm\n"
            "python -m src.training.train --model gru\n```"
        )
        return

    # ---- Filters ----
    st.subheader("Filters")
    fcol1, fcol2, fcol3 = st.columns(3)

    available_symbols = sorted(all_metrics["symbol"].unique()) if "symbol" in all_metrics.columns else []
    available_tfs = sorted(all_metrics["timeframe"].unique()) if "timeframe" in all_metrics.columns else []

    selected_symbols = fcol1.multiselect(
        "Symbol", options=available_symbols, default=available_symbols
    )
    selected_tfs = fcol2.multiselect(
        "Timeframe", options=available_tfs, default=available_tfs
    )

    metric_cols = _metric_columns(all_metrics)
    sort_metric = fcol3.selectbox("Sort by", options=metric_cols if metric_cols else ["(none)"])

    # Apply filters
    filtered = all_metrics.copy()
    if selected_symbols:
        filtered = filtered[filtered["symbol"].isin(selected_symbols)]
    if selected_tfs:
        filtered = filtered[filtered["timeframe"].isin(selected_tfs)]

    if filtered.empty:
        st.warning("No metrics match the current filters.")
        return

    # Sort
    if sort_metric and sort_metric in filtered.columns:
        # Error metrics: lower is better -> ascending; accuracy: higher is better -> descending
        error_keywords = {"mae", "rmse", "mse", "mape", "loss", "error"}
        ascending = any(kw in sort_metric.lower() for kw in error_keywords)
        filtered = filtered.sort_values(sort_metric, ascending=ascending)

    # ---- Sortable comparison table ----
    st.subheader("Comparison Table")
    display_cols = ["model", "symbol", "timeframe"] + metric_cols
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=min(400, 35 * len(filtered) + 38),
    )

    # ---- Best model highlight ----
    st.subheader("Best Model")
    if sort_metric and sort_metric in filtered.columns:
        best_row = filtered.iloc[0]
        bcol1, bcol2, bcol3 = st.columns(3)
        bcol1.metric("Model", best_row.get("model", "N/A"))
        bcol2.metric("Symbol", best_row.get("symbol", "N/A"))
        if sort_metric in best_row:
            val = best_row[sort_metric]
            bcol3.metric(sort_metric.upper(), f"{val:.4f}" if isinstance(val, float) else str(val))

    # ---- Grouped bar chart ----
    st.subheader("Metrics Comparison (Bar Chart)")
    metrics_dict: dict[str, dict[str, float]] = {}
    for _, row in filtered.iterrows():
        label = f"{row.get('model', '?')} / {row.get('timeframe', '?')}"
        metrics_dict[label] = {m: row[m] for m in metric_cols if pd.notna(row.get(m))}

    bar_fig = create_metrics_bar_chart(metrics_dict, title="Model Metrics Comparison")
    bar_fig.update_layout(height=500)
    st.plotly_chart(bar_fig, use_container_width=True)

    # ---- Radar chart ----
    st.subheader("Radar Chart")
    if len(metric_cols) >= 3:
        radar_fig = _create_radar_chart(
            filtered.reset_index(drop=True),
            metric_cols,
            title="Model Strengths",
        )
        radar_fig.update_layout(height=550)
        st.plotly_chart(radar_fig, use_container_width=True)
    else:
        st.caption("At least 3 numeric metrics are needed for a radar chart.")
