"""
Save / load utilities for models, scalers, metrics, and dataframes.

Every ``save_*`` helper calls :func:`ensure_dir` automatically so the caller
never has to create directories manually.

Usage::

    from src.utils.io import save_model, load_model, save_metrics
    save_model(model, "results/checkpoints/lstm_BTC_1h.pt")
    model = load_model("results/checkpoints/lstm_BTC_1h.pt")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import joblib
import pandas as pd
import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Directory helper
# ---------------------------------------------------------------------------

def ensure_dir(path: PathLike) -> Path:
    """Create *path* (and parents) if it does not already exist.

    If *path* looks like a file (has a suffix), the **parent** directory is
    created instead.

    Returns
    -------
    Path
        The resolved directory path that was ensured to exist.
    """
    p = Path(path)
    target = p.parent if p.suffix else p
    target.mkdir(parents=True, exist_ok=True)
    return target


# ---------------------------------------------------------------------------
# PyTorch models  (.pt)
# ---------------------------------------------------------------------------

def save_model(model: nn.Module, path: PathLike) -> None:
    """Serialise a PyTorch model's ``state_dict`` to *path*."""
    p = Path(path)
    ensure_dir(p)
    torch.save(model.state_dict(), p)
    logger.info("Model saved -> %s", p)


def load_model(model: nn.Module, path: PathLike, device: str = "cpu") -> nn.Module:
    """Load a ``state_dict`` from *path* into *model* and return it.

    Parameters
    ----------
    model:
        An **uninitialised** (randomly weighted) model instance whose
        architecture matches the saved state dict.
    path:
        File produced by :func:`save_model`.
    device:
        Map location for ``torch.load`` (e.g. ``"cpu"`` or ``"cuda:0"``).

    Returns
    -------
    nn.Module
        The same *model* with loaded weights, set to eval mode.
    """
    p = Path(path)
    state_dict = torch.load(p, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Model loaded <- %s", p)
    return model


# ---------------------------------------------------------------------------
# Scalers  (.joblib)
# ---------------------------------------------------------------------------

def save_scaler(scaler: Any, path: PathLike) -> None:
    """Persist a scikit-learn scaler (or any object) via joblib."""
    p = Path(path)
    ensure_dir(p)
    joblib.dump(scaler, p)
    logger.info("Scaler saved -> %s", p)


def load_scaler(path: PathLike) -> Any:
    """Load a scaler previously saved with :func:`save_scaler`."""
    p = Path(path)
    scaler = joblib.load(p)
    logger.info("Scaler loaded <- %s", p)
    return scaler


# ---------------------------------------------------------------------------
# Metrics  (.json)
# ---------------------------------------------------------------------------

def save_metrics(metrics: dict[str, Any], path: PathLike) -> None:
    """Write a metrics dictionary as pretty-printed JSON."""
    p = Path(path)
    ensure_dir(p)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    logger.info("Metrics saved -> %s", p)


def load_metrics(path: PathLike) -> dict[str, Any]:
    """Read a JSON metrics file back into a Python dict."""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as fh:
        metrics: dict[str, Any] = json.load(fh)
    logger.info("Metrics loaded <- %s", p)
    return metrics


# ---------------------------------------------------------------------------
# DataFrames  (.parquet)
# ---------------------------------------------------------------------------

def save_dataframe(df: pd.DataFrame, path: PathLike) -> None:
    """Write a DataFrame to Parquet format."""
    p = Path(path)
    ensure_dir(p)
    df.to_parquet(p, index=True)
    logger.info("DataFrame saved -> %s  (%d rows)", p, len(df))


def load_dataframe(path: PathLike) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame."""
    p = Path(path)
    df = pd.read_parquet(p)
    logger.info("DataFrame loaded <- %s  (%d rows)", p, len(df))
    return df
