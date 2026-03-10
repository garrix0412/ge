"""
Abstract base classes for all prediction models.

Every model in the project (ARIMA, XGBoost, LSTM, GRU, Transformer, TFT,
Anomaly Autoencoder) implements :class:`BaseModel` so that the training
infrastructure, evaluation, and dashboard can treat them uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """Uniform interface shared by every model in the project.

    Subclasses must implement :meth:`fit`, :meth:`predict`,
    :meth:`save`, and :meth:`load`.
    """

    name: str = "base"

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        """Train the model.

        Parameters
        ----------
        X_train : shape ``(n_samples, lookback, n_features)`` or ``(n_samples, n_features)``
        y_train : shape ``(n_samples,)`` or ``(n_samples, horizon)``
        X_val, y_val : optional validation arrays

        Returns
        -------
        dict
            Training history, e.g. ``{"train_loss": [...], "val_loss": [...]}``.
        """

    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Generate predictions.

        Parameters
        ----------
        X : input array with the same shape convention as ``fit``.

        Returns
        -------
        np.ndarray
            Predictions, shape ``(n_samples,)`` or ``(n_samples, horizon)``.
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist model to disk."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Restore model from disk."""


class BaseTorchModel(BaseModel):
    """Base class for all PyTorch-based models.

    Provides common utilities: device management, parameter counting,
    ``save`` / ``load`` via ``state_dict``, and MC-Dropout inference
    for uncertainty estimation.
    """

    name: str = "torch_base"

    def __init__(self) -> None:
        self.device: torch.device = torch.device("cpu")
        self.model: Optional[nn.Module] = None

    def to(self, device: str | torch.device) -> BaseTorchModel:
        """Move the underlying ``nn.Module`` to *device*."""
        self.device = torch.device(device)
        if self.model is not None:
            self.model.to(self.device)
        return self

    def auto_device(self) -> BaseTorchModel:
        """Select the best available device (CUDA > MPS > CPU)."""
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        logger.info("Auto-selected device: %s", device)
        return self.to(device)

    @property
    def num_parameters(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.model is None:
            raise RuntimeError("No model to save.")
        torch.save(self.model.state_dict(), path)
        logger.info("Model saved -> %s (%d params)", path, self.num_parameters)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if self.model is None:
            raise RuntimeError("Build the model architecture before loading weights.")
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        logger.info("Model loaded <- %s", path)

    # ------------------------------------------------------------------
    # MC-Dropout uncertainty estimation
    # ------------------------------------------------------------------

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        n_samples: int = 50,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run *n_samples* stochastic forward passes with dropout enabled.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(mean_prediction, std_prediction)`` — each of shape
            ``(n_samples_data, horizon)``.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        self.model.train()  # enable dropout
        x_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)

        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.model(x_tensor)
                # If model returns tuple (regression, classification), take regression
                if isinstance(out, tuple):
                    out = out[0]
                preds.append(out.cpu().numpy())

        self.model.eval()

        preds_array = np.stack(preds, axis=0)  # (n_samples, n_data, horizon)
        mean = preds_array.mean(axis=0)
        std = preds_array.std(axis=0)
        return mean, std
