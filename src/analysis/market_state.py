"""
Market-regime classifier and per-state evaluation.

Classifies each time step into **bull**, **bear**, or **sideways** based on
EMA crossovers and rolling returns, then slices prediction metrics by regime
so the user can see where the model excels and where it struggles.

Usage::

    from src.analysis.market_state import MarketStateClassifier

    clf = MarketStateClassifier()
    states = clf.classify(df, window=20)
    report = clf.compute_metrics_by_state(y_true, y_pred, states)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.training.evaluator import Evaluator
from src.utils.constants import FIGURES_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MarketStateClassifier:
    """Classify market regimes and evaluate model performance per regime.

    Parameters
    ----------
    ema_short_period:
        Period for the short (fast) EMA.  Default ``10``.
    ema_long_period:
        Period for the long (slow) EMA.  Default ``50``.
    return_threshold:
        Minimum absolute rolling return (as a fraction, e.g. 0.02 = 2 %)
        required to label a period as trending rather than sideways.
    volatility_threshold:
        Maximum rolling standard deviation of returns that still counts as
        "low volatility" (sideways).  When volatility exceeds this value
        the period may still be labelled trending if the return threshold
        is also met.
    """

    def __init__(
        self,
        ema_short_period: int = 10,
        ema_long_period: int = 50,
        return_threshold: float = 0.02,
        volatility_threshold: float = 0.015,
    ) -> None:
        if ema_short_period >= ema_long_period:
            raise ValueError(
                f"ema_short_period ({ema_short_period}) must be less than "
                f"ema_long_period ({ema_long_period})."
            )
        self.ema_short_period = ema_short_period
        self.ema_long_period = ema_long_period
        self.return_threshold = return_threshold
        self.volatility_threshold = volatility_threshold

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(
        self,
        df: pd.DataFrame,
        window: int = 20,
        price_col: str = "close",
    ) -> pd.Series:
        """Assign a market-state label to every row of *df*.

        Parameters
        ----------
        df:
            DataFrame that must contain a ``close`` column (or the column
            named by *price_col*).
        window:
            Rolling window used for return and volatility calculations.
        price_col:
            Column name for the price series.

        Returns
        -------
        pd.Series
            A Series with the same index as *df* and values in
            ``{"bull", "bear", "sideways"}``.
        """
        if price_col not in df.columns:
            raise KeyError(f"Column '{price_col}' not found in DataFrame.")

        price = df[price_col].astype(float)

        # EMA crossover signals
        ema_short = price.ewm(span=self.ema_short_period, adjust=False).mean()
        ema_long = price.ewm(span=self.ema_long_period, adjust=False).mean()
        ema_bull = ema_short > ema_long
        ema_bear = ema_short < ema_long

        # Rolling return over *window* periods
        rolling_return = price.pct_change(periods=window)

        # Rolling volatility (standard deviation of 1-step returns)
        rolling_vol = price.pct_change().rolling(window=window).std()

        # Combine signals
        bull_mask = ema_bull & (rolling_return > self.return_threshold)
        bear_mask = ema_bear & (rolling_return < -self.return_threshold)

        states = pd.Series("sideways", index=df.index, name="market_state")
        states[bull_mask] = "bull"
        states[bear_mask] = "bear"

        dist = self.get_state_distribution(states)
        logger.info(
            "Market-state classification complete — bull: %d (%.1f%%), "
            "bear: %d (%.1f%%), sideways: %d (%.1f%%)",
            dist["bull"]["count"],
            dist["bull"]["percentage"],
            dist["bear"]["count"],
            dist["bear"]["percentage"],
            dist["sideways"]["count"],
            dist["sideways"]["percentage"],
        )

        return states

    # ------------------------------------------------------------------
    # Per-state evaluation
    # ------------------------------------------------------------------

    def compute_metrics_by_state(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        states: pd.Series,
    ) -> dict[str, dict[str, float]]:
        """Compute regression metrics separately for each market state.

        Parameters
        ----------
        y_true:
            Ground-truth values, shape ``(n,)``.
        y_pred:
            Predicted values, shape ``(n,)``.
        states:
            Market-state labels aligned with *y_true* / *y_pred*.

        Returns
        -------
        dict[str, dict[str, float]]
            ``{"bull": {"mae": …, "rmse": …, "mape": …}, "bear": {…}, "sideways": {…}}``
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        states_arr = np.asarray(states).ravel()

        if not (len(y_true) == len(y_pred) == len(states_arr)):
            raise ValueError(
                f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}, "
                f"states={len(states_arr)}."
            )

        result: dict[str, dict[str, float]] = {}
        for state in ("bull", "bear", "sideways"):
            mask = states_arr == state
            n = int(mask.sum())
            if n == 0:
                logger.warning("No samples in '%s' state — skipping metrics.", state)
                result[state] = {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "count": 0}
                continue

            yt = y_true[mask]
            yp = y_pred[mask]
            result[state] = {
                "mae": Evaluator.mae(yt, yp),
                "rmse": Evaluator.rmse(yt, yp),
                "mape": Evaluator.mape(yt, yp),
                "count": n,
            }
            logger.info(
                "State '%s' (%d samples) — MAE: %.6f, RMSE: %.6f, MAPE: %.2f%%",
                state,
                n,
                result[state]["mae"],
                result[state]["rmse"],
                result[state]["mape"],
            )

        return result

    # ------------------------------------------------------------------
    # Distribution
    # ------------------------------------------------------------------

    @staticmethod
    def get_state_distribution(states: pd.Series) -> dict[str, dict[str, Any]]:
        """Return counts and percentages for each market state.

        Parameters
        ----------
        states:
            Series of market-state labels.

        Returns
        -------
        dict[str, dict[str, Any]]
            ``{"bull": {"count": int, "percentage": float}, …}``
        """
        total = len(states)
        dist: dict[str, dict[str, Any]] = {}
        for label in ("bull", "bear", "sideways"):
            count = int((states == label).sum())
            pct = (count / total * 100.0) if total > 0 else 0.0
            dist[label] = {"count": count, "percentage": round(pct, 2)}
        return dist

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_states(
        self,
        df: pd.DataFrame,
        states: pd.Series,
        price_col: str = "close",
        save_path: Optional[str | Path] = None,
    ) -> go.Figure:
        """Plot price with colour-coded market-state background regions.

        Parameters
        ----------
        df:
            DataFrame with a price column.
        states:
            Market-state labels (same index as *df*).
        price_col:
            Column name for the price series.
        save_path:
            If given, the figure is saved as an HTML file.  When *None*
            the figure is saved to ``results/figures/market_states.html``.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        colour_map = {
            "bull": "rgba(0, 200, 0, 0.15)",
            "bear": "rgba(200, 0, 0, 0.15)",
            "sideways": "rgba(128, 128, 128, 0.10)",
        }

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.75, 0.25],
            subplot_titles=("Price with Market States", "State Timeline"),
        )

        # --- Price line ---
        x_vals = df.index if isinstance(df.index, pd.DatetimeIndex) else list(range(len(df)))
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=df[price_col],
                mode="lines",
                name="Close Price",
                line=dict(color="black", width=1),
            ),
            row=1,
            col=1,
        )

        # --- Shaded regions per state ---
        state_values = states.values
        x_arr = x_vals if isinstance(x_vals, pd.DatetimeIndex) else list(range(len(df)))

        for state, colour in colour_map.items():
            mask = state_values == state
            if not mask.any():
                continue
            # Build contiguous blocks for shading
            starts, ends = self._find_contiguous_blocks(mask)
            for s, e in zip(starts, ends):
                fig.add_vrect(
                    x0=x_arr[s],
                    x1=x_arr[min(e, len(x_arr) - 1)],
                    fillcolor=colour,
                    layer="below",
                    line_width=0,
                    row=1,
                    col=1,
                )

        # --- State timeline (categorical scatter) ---
        state_num = states.map({"bull": 1, "bear": -1, "sideways": 0}).values
        colours = [
            "green" if s == "bull" else "red" if s == "bear" else "gray"
            for s in state_values
        ]
        fig.add_trace(
            go.Scatter(
                x=x_arr,
                y=state_num,
                mode="markers",
                marker=dict(color=colours, size=2),
                name="State",
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            height=700,
            title_text="Market State Classification",
            showlegend=True,
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(
            title_text="State",
            tickvals=[-1, 0, 1],
            ticktext=["Bear", "Sideways", "Bull"],
            row=2,
            col=1,
        )

        # Save
        out_path = Path(save_path) if save_path else FIGURES_DIR / "market_states.html"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path))
        logger.info("Market-state figure saved -> %s", out_path)

        return fig

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_contiguous_blocks(mask: np.ndarray) -> tuple[list[int], list[int]]:
        """Return (start_indices, end_indices) for contiguous True regions."""
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            return [], []
        diff = np.diff(mask.astype(int))
        starts = list(np.where(diff == 1)[0] + 1)
        ends = list(np.where(diff == -1)[0] + 1)
        if mask[0]:
            starts.insert(0, 0)
        if mask[-1]:
            ends.append(len(mask))
        return starts, ends
