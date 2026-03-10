"""
Training Results page.

Displays saved model metrics (regression + classification), training loss
curves, and a filtered table of other configurations for the same model.
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
    METRICS_FILE_PATTERN,
    HISTORY_FILE_PATTERN,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner="Loading metrics ...")
def _load_model_metrics(
    model_name: str, symbol: str, timeframe: str, lookback: int, horizon: int,
) -> dict[str, Any] | None:
    filename = METRICS_FILE_PATTERN.format(
        model_name=model_name,
        symbol=symbol.replace("/", "_"),
        timeframe=timeframe,
        lookback=lookback,
        horizon=horizon,
    )
    path = METRICS_DIR / filename
    if not path.exists():
        return None
    with open(path, "r") as fh:
        return json.load(fh)


@st.cache_data(ttl=300, show_spinner="Loading training history ...")
def _load_training_history(
    model_name: str, symbol: str, timeframe: str, lookback: int, horizon: int,
) -> dict[str, Any] | None:
    filename = HISTORY_FILE_PATTERN.format(
        model_name=model_name,
        symbol=symbol.replace("/", "_"),
        timeframe=timeframe,
        lookback=lookback,
        horizon=horizon,
    )
    path = METRICS_DIR / filename
    if not path.exists():
        return None
    with open(path, "r") as fh:
        return json.load(fh)


@st.cache_data(ttl=300, show_spinner="Loading comparison table ...")
def _load_comparison_table() -> pd.DataFrame | None:
    path = METRICS_DIR / "comparison_table.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _build_loss_chart(history: dict[str, Any], model_name: str) -> go.Figure:
    """Build a training loss curve from the history JSON."""
    fig = go.Figure()

    if "train_loss" in history:
        epochs = list(range(1, len(history["train_loss"]) + 1))
        fig.add_trace(go.Scatter(
            x=epochs, y=history["train_loss"],
            mode="lines+markers", name="Train Loss",
            line=dict(color="#42a5f5", width=2),
        ))
    if "val_loss" in history:
        epochs = list(range(1, len(history["val_loss"]) + 1))
        fig.add_trace(go.Scatter(
            x=epochs, y=history["val_loss"],
            mode="lines+markers", name="Val Loss",
            line=dict(color="#ffa726", width=2),
        ))
    if "train_aic" in history:
        fig.add_trace(go.Bar(
            x=["AIC"], y=history["train_aic"],
            name="Train AIC", marker_color="#42a5f5",
        ))
    if "reg_best_score" in history:
        labels, values, colors = [], [], []
        if history.get("reg_best_score"):
            labels.append("Reg Best Score")
            values.append(history["reg_best_score"][0])
            colors.append("#42a5f5")
        if history.get("cls_best_score"):
            labels.append("Cls Best Score")
            values.append(history["cls_best_score"][0])
            colors.append("#ffa726")
        if labels:
            fig.add_trace(go.Bar(x=labels, y=values, marker_color=colors, name="Best Score"))

    fig.update_layout(
        xaxis_title="Epoch" if "train_loss" in history else "",
        yaxis_title="Loss / Score",
        height=400,
    )
    return _apply_defaults(fig, f"{model_name.upper()} Training History")


# ---------------------------------------------------------------------------
# Page renderer
# ---------------------------------------------------------------------------


def render() -> None:
    """Main entry point for the Training Results page."""

    st.header("Training Results")
    st.caption("View saved model performance metrics and training history for each configuration.")

    selections = render_sidebar(page="prediction")
    symbol: str = selections["symbol"]
    model_name: str = selections["model"]
    timeframe: str = selections["timeframe"]
    lookback: int = selections["lookback_window"]
    horizon: int = selections["forecast_horizon"]

    config_label = f"**{model_name.upper()}** | {symbol} | {timeframe} | lb={lookback} h={horizon}"
    st.subheader(f"Configuration: {config_label}")

    # ---- Load metrics ----
    metrics = _load_model_metrics(model_name, symbol, timeframe, lookback, horizon)
    if metrics is None:
        st.warning(
            f"No metrics found for this configuration.  \n"
            "Train the model first:\n\n"
            f"```bash\npython scripts/run_training.py --model {model_name} "
            f"--symbol {symbol} --timeframe {timeframe} "
            f"--lookback {lookback} --horizon {horizon}\n```"
        )
    else:
        # -- Regression metrics --
        st.markdown("#### Regression Metrics")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("MAE", f"{metrics.get('reg_mae', 0):.4f}")
        r2.metric("RMSE", f"{metrics.get('reg_rmse', 0):.4f}")
        r3.metric("MAPE", f"{metrics.get('reg_mape', 0):.2f}%")
        r4.metric("Dir. Accuracy", f"{metrics.get('reg_directional_accuracy', 0):.2%}")

        # -- Classification metrics --
        cls_acc = metrics.get("cls_accuracy")
        cls_f1 = metrics.get("cls_f1_score")
        cls_auc = metrics.get("cls_auc_roc")
        has_cls = any(v is not None for v in [cls_acc, cls_f1, cls_auc])

        if has_cls:
            st.markdown("#### Classification Metrics")
            c1, c2, c3 = st.columns(3)
            if cls_acc is not None:
                c1.metric("Accuracy", f"{cls_acc:.4f}")
            else:
                c1.metric("Accuracy", "N/A")
            if cls_f1 is not None:
                c2.metric("F1 Score", f"{cls_f1:.4f}")
            else:
                c2.metric("F1 Score", "N/A")
            if cls_auc is not None:
                c3.metric("AUC-ROC", f"{cls_auc:.4f}")
            else:
                c3.metric("AUC-ROC", "N/A")

    # ---- Load training history ----
    history = _load_training_history(model_name, symbol, timeframe, lookback, horizon)
    if history is not None:
        st.markdown("#### Training History")
        loss_fig = _build_loss_chart(history, model_name)
        st.plotly_chart(loss_fig, use_container_width=True)

    # ---- Other configurations for same model ----
    comp_df = _load_comparison_table()
    if comp_df is not None and not comp_df.empty:
        st.markdown("#### Other Configurations")
        model_rows = comp_df[comp_df["model"] == model_name].copy()
        if not model_rows.empty:
            display_cols = [
                c for c in [
                    "symbol", "timeframe", "lookback", "horizon",
                    "reg_mae", "reg_rmse", "reg_mape",
                    "reg_directional_accuracy",
                    "cls_accuracy", "cls_f1_score", "cls_auc_roc",
                    "elapsed_seconds",
                ]
                if c in model_rows.columns
            ]
            st.dataframe(
                model_rows[display_cols].reset_index(drop=True),
                use_container_width=True,
                height=min(400, 35 * len(model_rows) + 38),
            )
