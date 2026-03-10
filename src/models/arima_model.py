"""
ARIMA model for univariate time-series forecasting.

Uses statsmodels ARIMA with automatic order selection via AIC grid search
over (p, d, q) ranges specified in the project configuration.
"""

from __future__ import annotations

import itertools
import pickle
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.models.base_model import BaseModel
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ARIMAModel(BaseModel):
    """Univariate ARIMA forecaster with AIC-based order selection.

    This model operates on the **close price** only.  If the input array
    is 2-D or 3-D (e.g. ``(n, lookback, features)``), it is flattened to
    a 1-D series before fitting.

    Parameters
    ----------
    horizon : int
        Number of future steps to forecast.
    config : optional
        Pre-loaded :class:`AppConfig`.  When *None* the default YAML
        configuration is loaded automatically.
    """

    name: str = "arima"

    def __init__(self, horizon: int = 1, config: Any = None, **kwargs: Any) -> None:
        self.horizon = horizon
        cfg = config or load_config()
        arima_cfg = cfg.model.arima

        self.max_p: int = arima_cfg.max_p
        self.max_d: int = arima_cfg.max_d
        self.max_q: int = arima_cfg.max_q
        self.seasonal: bool = arima_cfg.seasonal

        self.order: Optional[tuple[int, int, int]] = None
        self._fitted_model: Any = None  # statsmodels ARIMAResults
        self._train_series: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_series(X: np.ndarray) -> np.ndarray:
        """Collapse any multi-dimensional input to a flat 1-D array."""
        arr = np.asarray(X, dtype=np.float64).squeeze()
        if arr.ndim > 1:
            # Take the last column (assumed to be close price) of flattened view
            arr = arr.reshape(-1, arr.shape[-1])[:, -1]
        return arr.ravel()

    def _grid_search(self, series: np.ndarray) -> tuple[int, int, int]:
        """Find the ARIMA order that minimises AIC."""
        from statsmodels.tsa.arima.model import ARIMA  # noqa: WPS433

        best_aic = np.inf
        best_order: tuple[int, int, int] = (1, 0, 0)

        p_range = range(0, self.max_p + 1)
        d_range = range(0, self.max_d + 1)
        q_range = range(0, self.max_q + 1)

        total = (self.max_p + 1) * (self.max_d + 1) * (self.max_q + 1)
        logger.info("ARIMA grid search: %d combinations (p≤%d, d≤%d, q≤%d)",
                     total, self.max_p, self.max_d, self.max_q)

        for p, d, q in itertools.product(p_range, d_range, q_range):
            if p == 0 and q == 0:
                continue  # degenerate model
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(series, order=(p, d, q))
                    result = model.fit()
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_order = (p, d, q)
            except Exception:  # noqa: BLE001
                # Convergence failure or singular matrix — skip silently.
                continue

        logger.info("Best ARIMA order: %s  (AIC=%.2f)", best_order, best_aic)
        return best_order

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        """Fit the ARIMA model on the training close-price series.

        ``y_train`` is used as the primary series.  If ``y_train`` is 2-D
        (multi-horizon targets), only the first column is used.

        Returns
        -------
        dict
            ``{"train_aic": [aic_value]}`` as a minimal history.
        """
        from statsmodels.tsa.arima.model import ARIMA  # noqa: WPS433

        series = self._to_series(y_train)
        self._train_series = series

        # Auto-select order via AIC grid search
        self.order = self._grid_search(series)

        # Fit final model with best order
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(series, order=self.order)
            self._fitted_model = model.fit()

        aic = float(self._fitted_model.aic)
        logger.info("ARIMA(%d,%d,%d) fitted — AIC=%.2f, n_obs=%d",
                     *self.order, aic, len(series))

        return {"train_aic": [aic]}

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Forecast the next ``horizon`` steps.

        Parameters
        ----------
        X : np.ndarray
            Ignored for pure forecasting (ARIMA is autoregressive).
            If provided, its length can hint at how many rolling forecasts
            to produce.

        Returns
        -------
        np.ndarray
            Shape ``(n_forecasts, horizon)`` or ``(horizon,)`` when a single
            forecast is requested.
        """
        if self._fitted_model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        horizon = kwargs.get("horizon", self.horizon)

        # Number of forecast origins
        n = 1 if X is None else max(1, len(np.asarray(X)))

        forecasts: list[np.ndarray] = []
        for i in range(n):
            fc = self._fitted_model.forecast(steps=horizon)
            forecasts.append(np.asarray(fc, dtype=np.float64))

        result = np.stack(forecasts, axis=0)  # (n, horizon)
        return result.squeeze()

    def save(self, path: str | Path) -> None:
        """Pickle the fitted ARIMA result to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "order": self.order,
            "fitted_model": self._fitted_model,
            "train_series": self._train_series,
            "horizon": self.horizon,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)

        logger.info("ARIMAModel saved -> %s", path)

    def load(self, path: str | Path) -> None:
        """Restore a previously saved ARIMA model from *path*."""
        path = Path(path)
        with open(path, "rb") as fh:
            payload = pickle.load(fh)  # noqa: S301

        self.order = payload["order"]
        self._fitted_model = payload["fitted_model"]
        self._train_series = payload["train_series"]
        self.horizon = payload["horizon"]

        logger.info("ARIMAModel loaded <- %s  (order=%s)", path, self.order)
