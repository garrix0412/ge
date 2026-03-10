"""
Model registry with lazy imports and factory function.

Centralises model discovery so that the rest of the codebase can
instantiate any model by name without importing heavyweight
dependencies up front.

Usage::

    from src.models.registry import get_model, list_models

    model = get_model("lstm", input_size=22, lookback=24, horizon=1)
    print(list_models())
"""

from __future__ import annotations

from typing import Any

from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy-import helpers
# ---------------------------------------------------------------------------
# Each helper returns the model **class** only when called, so that modules
# like ``xgboost``, ``statsmodels``, or heavy torch sub-packages are not
# loaded unless actually requested.


def _get_arima_class() -> type:
    from src.models.arima_model import ARIMAModel
    return ARIMAModel


def _get_xgboost_class() -> type:
    from src.models.xgboost_model import XGBoostModel
    return XGBoostModel


def _get_lstm_class() -> type:
    from src.models.lstm_model import LSTMModel
    return LSTMModel


def _get_gru_class() -> type:
    from src.models.gru_model import GRUModel
    return GRUModel


def _get_transformer_class() -> type:
    from src.models.transformer_model import TransformerModel
    return TransformerModel


def _get_tft_class() -> type:
    from src.models.tft_model import TFTModel
    return TFTModel


def _get_anomaly_class() -> type:
    from src.models.anomaly_autoencoder import AnomalyAutoencoder
    return AnomalyAutoencoder


# ---------------------------------------------------------------------------
# Registry mapping
# ---------------------------------------------------------------------------
# Maps a short model name to a *callable* that returns the class.
# This avoids importing every model at module-load time.

_MODEL_LOADERS: dict[str, callable] = {
    "arima": _get_arima_class,
    "xgboost": _get_xgboost_class,
    "lstm": _get_lstm_class,
    "gru": _get_gru_class,
    "transformer": _get_transformer_class,
    "tft": _get_tft_class,
    "anomaly": _get_anomaly_class,
}

# Eagerly-resolved cache (populated on first use per key).
MODEL_REGISTRY: dict[str, type] = {}


def _resolve(name: str) -> type:
    """Resolve a model name to its class, caching the result."""
    if name not in MODEL_REGISTRY:
        loader = _MODEL_LOADERS.get(name)
        if loader is None:
            raise KeyError(
                f"Unknown model '{name}'. Available: {list(_MODEL_LOADERS.keys())}"
            )
        MODEL_REGISTRY[name] = loader()
    return MODEL_REGISTRY[name]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_model(name: str, **kwargs: Any) -> BaseModel:
    """Instantiate a model by its registry name.

    Parameters
    ----------
    name:
        Short model identifier (e.g. ``"lstm"``, ``"xgboost"``).
    **kwargs:
        Forwarded to the model constructor (e.g. ``input_size``,
        ``lookback``, ``horizon``, ``config``).

    Returns
    -------
    BaseModel
        A freshly constructed model instance.

    Raises
    ------
    KeyError
        If *name* is not in the registry.
    """
    model_cls = _resolve(name)
    logger.info("Creating model '%s' (%s) with kwargs: %s", name, model_cls.__name__, list(kwargs.keys()))
    return model_cls(**kwargs)


def list_models() -> list[str]:
    """Return the names of all registered models.

    Returns
    -------
    list[str]
        Sorted list of available model names.
    """
    return sorted(_MODEL_LOADERS.keys())
