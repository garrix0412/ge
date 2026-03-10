"""
XGBoost model with dual-head (regression + classification) support.

Flattens 3-D sequence inputs to 2-D feature matrices and trains separate
XGBRegressor and XGBClassifier instances.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np

from src.models.base_model import BaseModel
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost wrapper with separate regression and classification heads.

    Parameters
    ----------
    horizon : int
        Number of future time-steps to predict.
    config : optional
        Pre-loaded :class:`AppConfig`.  Loaded from YAML when *None*.
    """

    name: str = "xgboost"

    def __init__(self, horizon: int = 1, config: Any = None, **kwargs: Any) -> None:
        from xgboost import XGBClassifier, XGBRegressor  # noqa: WPS433

        self.horizon = horizon
        cfg = config or load_config()
        xgb_cfg = cfg.model.xgboost
        common_cfg = cfg.model.common

        self._xgb_params: dict[str, Any] = {
            "n_estimators": xgb_cfg.n_estimators,
            "max_depth": xgb_cfg.max_depth,
            "learning_rate": xgb_cfg.learning_rate,
            "subsample": xgb_cfg.subsample,
            "colsample_bytree": xgb_cfg.colsample_bytree,
            "random_state": common_cfg.seed,
            "n_jobs": -1,
            "verbosity": 0,
        }
        self._early_stopping_rounds: int = xgb_cfg.early_stopping_rounds

        # Regression model (one per horizon step, or single MultiOutput)
        self._reg_model: Optional[XGBRegressor] = None
        # Classification model
        self._cls_model: Optional[XGBClassifier] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        """Flatten 3-D sequences ``(n, lookback, features)`` to 2-D ``(n, lookback*features)``."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3:
            n, lookback, features = X.shape
            return X.reshape(n, lookback * features)
        if X.ndim == 1:
            return X.reshape(-1, 1)
        return X

    @staticmethod
    def _prepare_cls_targets(y: np.ndarray) -> np.ndarray:
        """Convert regression targets to binary classification labels.

        Positive returns (>= 0) map to 1, negative to 0.  If *y* is already
        binary ``{0, 1}`` it is returned unchanged.
        """
        y = np.asarray(y)
        unique = np.unique(y)
        if set(unique).issubset({0, 1, 0.0, 1.0}):
            return y.astype(np.int32)
        # Interpret as price / return: positive -> 1
        return (y >= 0).astype(np.int32)

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
        """Train regression and classification XGBoost models.

        Parameters
        ----------
        X_train : np.ndarray
            Shape ``(n, lookback, features)`` or ``(n, n_features)``.
        y_train : np.ndarray
            Regression targets.  Classification labels are derived
            automatically (positive return -> 1).
        X_val, y_val : optional
            Validation data used for early stopping.

        Returns
        -------
        dict
            ``{"reg_best_score": [...], "cls_best_score": [...]}``.
        """
        from xgboost import XGBClassifier, XGBRegressor  # noqa: WPS433

        X_train_2d = self._flatten(X_train)
        y_reg = np.asarray(y_train, dtype=np.float32)
        if y_reg.ndim == 1:
            y_reg = y_reg.ravel()

        # Use explicit classification labels if provided, else derive from regression targets
        y_train_cls = kwargs.get("y_train_cls", None)
        if y_train_cls is not None:
            y_cls = np.asarray(y_train_cls, dtype=np.int32)
        else:
            y_cls = self._prepare_cls_targets(y_train)
        if y_cls.ndim > 1:
            y_cls = y_cls[:, 0]  # classify first horizon step

        eval_set_reg: list[tuple[np.ndarray, np.ndarray]] = []
        eval_set_cls: list[tuple[np.ndarray, np.ndarray]] = []
        if X_val is not None and y_val is not None:
            X_val_2d = self._flatten(X_val)
            y_val_reg = np.asarray(y_val, dtype=np.float32)
            if y_val_reg.ndim == 1:
                y_val_reg = y_val_reg.ravel()
            y_val_cls = self._prepare_cls_targets(y_val)
            if y_val_cls.ndim > 1:
                y_val_cls = y_val_cls[:, 0]
            eval_set_reg.append((X_val_2d, y_val_reg))
            eval_set_cls.append((X_val_2d, y_val_cls))

        # ---- Regression ----
        logger.info("Training XGBRegressor (n_train=%d, n_features=%d) ...",
                     X_train_2d.shape[0], X_train_2d.shape[1])

        fit_params_reg: dict[str, Any] = {"verbose": False}
        if eval_set_reg:
            fit_params_reg["eval_set"] = eval_set_reg

        self._reg_model = XGBRegressor(
            **self._xgb_params,
            early_stopping_rounds=self._early_stopping_rounds if eval_set_reg else None,
        )
        self._reg_model.fit(X_train_2d, y_reg, **fit_params_reg)

        # ---- Classification ----
        n_classes = len(np.unique(y_cls))
        if n_classes < 2:
            logger.warning(
                "Skipping classifier training — only %d class(es) in labels.", n_classes,
            )
            self._cls_model = None
        else:
            logger.info("Training XGBClassifier ...")
            fit_params_cls: dict[str, Any] = {"verbose": False}
            if eval_set_cls:
                fit_params_cls["eval_set"] = eval_set_cls

            self._cls_model = XGBClassifier(
                **self._xgb_params,
                objective="binary:logistic",
                eval_metric="logloss",
                early_stopping_rounds=self._early_stopping_rounds if eval_set_cls else None,
            )
            self._cls_model.fit(X_train_2d, y_cls, **fit_params_cls)

        history: dict[str, list[float]] = {
            "reg_best_score": [float(self._reg_model.best_score)] if hasattr(self._reg_model, "best_score") and self._reg_model.best_score else [],
            "cls_best_score": [float(self._cls_model.best_score)] if hasattr(self._cls_model, "best_score") and self._cls_model.best_score else [],
        }

        logger.info("XGBoost training complete.")
        return history

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Return regression predictions.

        Parameters
        ----------
        X : np.ndarray
            Input array — 2-D or 3-D.

        Returns
        -------
        np.ndarray
            Predicted values with shape ``(n_samples,)`` or
            ``(n_samples, horizon)``.
        """
        if self._reg_model is None:
            raise RuntimeError("Regression model not fitted. Call fit() first.")
        X_2d = self._flatten(X)
        return self._reg_model.predict(X_2d)

    def predict_cls(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Return classification probabilities for the positive class.

        Parameters
        ----------
        X : np.ndarray
            Input array — 2-D or 3-D.

        Returns
        -------
        np.ndarray
            Probability of class 1, shape ``(n_samples,)``.
        """
        if self._cls_model is None:
            raise RuntimeError("Classification model not fitted. Call fit() first.")
        X_2d = self._flatten(X)
        proba = self._cls_model.predict_proba(X_2d)
        # predict_proba returns (n, 2); take positive-class column
        return proba[:, 1] if proba.ndim == 2 else proba

    def save(self, path: str | Path) -> None:
        """Save both models via joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "reg_model": self._reg_model,
            "cls_model": self._cls_model,
            "horizon": self.horizon,
            "xgb_params": self._xgb_params,
        }
        joblib.dump(payload, path)
        logger.info("XGBoostModel saved -> %s", path)

    def load(self, path: str | Path) -> None:
        """Load both models via joblib."""
        path = Path(path)
        payload = joblib.load(path)

        self._reg_model = payload["reg_model"]
        self._cls_model = payload["cls_model"]
        self.horizon = payload["horizon"]
        self._xgb_params = payload["xgb_params"]

        logger.info("XGBoostModel loaded <- %s", path)
