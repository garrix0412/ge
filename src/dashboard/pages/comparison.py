"""
Model Comparison page.

Loads ``comparison_table.csv`` from the metrics directory, shows a filterable
comparison table, grouped bar chart of average metrics per model, and a radar
chart.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.components.charts import _apply_defaults
from src.dashboard.components.sidebar import render_sidebar
from src.utils.constants import METRICS_DIR

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_METRIC_COLS = [
    "reg_mae", "reg_rmse", "reg_mape", "reg_directional_accuracy",
    "cls_accuracy", "cls_f1_score", "cls_auc_roc",
]


@st.cache_data(ttl=300, show_spinner="Loading comparison table ...")
def _load_comparison_table() -> pd.DataFrame:
    """Read ``comparison_table.csv`` from METRICS_DIR.

    Returns an empty DataFrame if the file is missing.
    """
    path = METRICS_DIR / "comparison_table.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


def _available_metric_cols(df: pd.DataFrame) -> list[str]:
    """Return metric columns that actually exist and have non-null data."""
    return [c for c in _METRIC_COLS if c in df.columns and df[c].notna().any()]


# ---------------------------------------------------------------------------
# Radar chart
# ---------------------------------------------------------------------------

def _create_radar_chart(df: pd.DataFrame, metric_cols: list[str], title: str = "") -> go.Figure:
    """Radar chart — each model's average metrics normalised to [0, 1]."""
    if df.empty or not metric_cols:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data for radar chart", showarrow=False, font=dict(size=14))
        return _apply_defaults(fig, title)

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
        values.append(values[0])
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
    st.caption("Compare all 144 training configurations across models, symbols, timeframes, and hyperparameters.")

    render_sidebar(page="comparison")

    # ---- Load comparison table ----
    all_data = _load_comparison_table()

    if all_data.empty:
        st.info(
            "No comparison table found.  \n"
            "Run the full pipeline to generate it:\n\n"
            "```bash\npython scripts/run_pipeline.py\n```"
        )
        return

    # ---- Filters ----
    st.subheader("Filters")
    fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns(5)

    available_models = sorted(all_data["model"].dropna().unique()) if "model" in all_data.columns else []
    available_symbols = sorted(all_data["symbol"].dropna().unique()) if "symbol" in all_data.columns else []
    available_tfs = sorted(all_data["timeframe"].dropna().unique()) if "timeframe" in all_data.columns else []
    available_lbs = sorted(all_data["lookback"].dropna().unique()) if "lookback" in all_data.columns else []
    available_hs = sorted(all_data["horizon"].dropna().unique()) if "horizon" in all_data.columns else []

    selected_models = fcol1.multiselect("Model", options=available_models, default=available_models)
    selected_symbols = fcol2.multiselect("Symbol", options=available_symbols, default=available_symbols)
    selected_tfs = fcol3.multiselect("Timeframe", options=available_tfs, default=available_tfs)
    selected_lbs = fcol4.multiselect("Lookback", options=available_lbs, default=available_lbs)
    selected_hs = fcol5.multiselect("Horizon", options=available_hs, default=available_hs)

    # Apply filters
    filtered = all_data.copy()
    if selected_models:
        filtered = filtered[filtered["model"].isin(selected_models)]
    if selected_symbols:
        filtered = filtered[filtered["symbol"].isin(selected_symbols)]
    if selected_tfs:
        filtered = filtered[filtered["timeframe"].isin(selected_tfs)]
    if selected_lbs and "lookback" in filtered.columns:
        filtered = filtered[filtered["lookback"].isin(selected_lbs)]
    if selected_hs and "horizon" in filtered.columns:
        filtered = filtered[filtered["horizon"].isin(selected_hs)]

    if filtered.empty:
        st.warning("No results match the current filters.")
        return

    metric_cols = _available_metric_cols(filtered)

    # Sort control
    sort_col = st.selectbox("Sort by", options=metric_cols if metric_cols else ["(none)"])
    if sort_col and sort_col in filtered.columns:
        error_keywords = {"mae", "rmse", "mse", "mape", "loss", "error"}
        ascending = any(kw in sort_col.lower() for kw in error_keywords)
        filtered = filtered.sort_values(sort_col, ascending=ascending)

    # ---- Full comparison table ----
    st.subheader("Comparison Table")
    display_cols = ["model", "symbol", "timeframe", "lookback", "horizon"] + metric_cols
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=min(500, 35 * len(filtered) + 38),
    )

    # ---- Best model highlight ----
    st.subheader("Best Model")
    if sort_col and sort_col in filtered.columns:
        best_row = filtered.iloc[0]
        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
        bcol1.metric("Model", best_row.get("model", "N/A"))
        bcol2.metric("Symbol", best_row.get("symbol", "N/A"))
        bcol3.metric("Config", f"lb={int(best_row.get('lookback', 0))} h={int(best_row.get('horizon', 0))}")
        if sort_col in best_row:
            val = best_row[sort_col]
            bcol4.metric(sort_col, f"{val:.4f}" if isinstance(val, float) else str(val))

    # ---- Grouped bar chart: average metrics per model ----
    st.subheader("Average Metrics by Model")
    if metric_cols:
        avg_by_model = filtered.groupby("model")[metric_cols].mean().reset_index()
        fig = go.Figure()
        for metric in metric_cols:
            if metric in avg_by_model.columns:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=avg_by_model["model"],
                    y=avg_by_model[metric],
                ))
        fig.update_layout(barmode="group", height=500)
        fig.update_yaxes(title_text="Value")
        fig.update_xaxes(title_text="Model")
        fig = _apply_defaults(fig, "Average Metrics per Model")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Radar chart ----
    st.subheader("Radar Chart")
    radar_metrics = [c for c in metric_cols if filtered[c].notna().all()]
    if len(radar_metrics) >= 3:
        avg_for_radar = filtered.groupby("model")[radar_metrics].mean().reset_index()
        radar_fig = _create_radar_chart(
            avg_for_radar,
            radar_metrics,
            title="Model Strengths (averaged across configs)",
        )
        radar_fig.update_layout(height=550)
        st.plotly_chart(radar_fig, use_container_width=True)
    else:
        st.caption("At least 3 numeric metrics with data are needed for a radar chart.")
