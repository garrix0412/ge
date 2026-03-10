"""
GRU model with dual-head (regression + classification) output.

Mirrors the LSTM architecture but replaces ``nn.LSTM`` with ``nn.GRU``.
Implements the full training loop inside :meth:`fit` with Adam optimiser,
dual-head loss (weighted MSE + BCE), early stopping, and learning-rate
scheduling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_model import BaseTorchModel
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ======================================================================
# Inner nn.Module
# ======================================================================

class _GRUNetwork(nn.Module):
    """GRU encoder with dual regression / classification heads.

    Parameters
    ----------
    input_size : int
        Number of input features per time-step.
    hidden_size : int
        GRU hidden dimension.
    num_layers : int
        Number of stacked GRU layers.
    dropout : float
        Dropout rate applied between GRU layers (>= 2 layers).
    horizon : int
        Forecast length (number of future steps).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        horizon: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Regression head — predicts continuous values for each horizon step
        self.reg_head = nn.Linear(hidden_size, horizon)

        # Classification head — predicts probability of price going up
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size, horizon),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, lookback, features)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(reg_out, cls_out)`` each of shape ``(batch, horizon)``.
        """
        # gru_out: (batch, lookback, hidden)
        gru_out, _ = self.gru(x)

        # Use the last time-step hidden state
        last_hidden = gru_out[:, -1, :]  # (batch, hidden)

        reg_out = self.reg_head(last_hidden)
        cls_out = self.cls_head(last_hidden)

        return reg_out, cls_out


# ======================================================================
# Public model wrapper
# ======================================================================

class GRUModel(BaseTorchModel):
    """GRU-based forecaster with dual regression / classification heads.

    Parameters
    ----------
    input_size : int
        Number of input features per time-step.
    horizon : int
        Number of future steps to forecast.
    config : optional
        Pre-loaded :class:`AppConfig`.
    """

    name: str = "gru"

    def __init__(
        self,
        input_size: int,
        horizon: int = 1,
        config: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        cfg = config or load_config()
        gru_cfg = cfg.model.gru
        common_cfg = cfg.model.common
        dual_cfg = cfg.model.dual_head

        self.input_size = input_size
        self.horizon = horizon
        self.hidden_size: int = gru_cfg.hidden_size
        self.num_layers: int = gru_cfg.num_layers
        self.dropout: float = gru_cfg.dropout

        # Training hyper-parameters
        self.lr: float = common_cfg.learning_rate
        self.batch_size: int = common_cfg.batch_size
        self.max_epochs: int = common_cfg.max_epochs
        self.patience: int = common_cfg.patience
        self.seed: int = common_cfg.seed
        self.num_workers: int = common_cfg.num_workers
        self.alpha: float = dual_cfg.alpha  # regression loss weight

        # Build the network
        self.model = _GRUNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            horizon=self.horizon,
        )

        self.auto_device()
        self._set_seed()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_seed(self) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    @staticmethod
    def _prepare_cls_targets(y: np.ndarray) -> np.ndarray:
        """Derive binary classification targets from regression targets."""
        y = np.asarray(y, dtype=np.float32)
        unique = np.unique(y)
        if set(unique.tolist()).issubset({0.0, 1.0}):
            return y
        return (y >= 0).astype(np.float32)

    def _build_loader(
        self,
        X: np.ndarray,
        y_reg: np.ndarray,
        y_cls: np.ndarray,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create a :class:`DataLoader` from numpy arrays."""
        X_t = torch.as_tensor(X, dtype=torch.float32)
        y_reg_t = torch.as_tensor(y_reg, dtype=torch.float32)
        y_cls_t = torch.as_tensor(y_cls, dtype=torch.float32)

        # Ensure target shapes are 2-D: (n, horizon)
        if y_reg_t.ndim == 1:
            y_reg_t = y_reg_t.unsqueeze(1)
        if y_cls_t.ndim == 1:
            y_cls_t = y_cls_t.unsqueeze(1)

        dataset = TensorDataset(X_t, y_reg_t, y_cls_t)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,  # safe default for portability
            pin_memory=(self.device.type == "cuda"),
        )

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
        """Train the GRU with a dual-head loss.

        Loss = alpha * MSE(reg) + (1 - alpha) * BCE(cls)

        Parameters
        ----------
        X_train : np.ndarray
            Shape ``(n, lookback, features)``.
        y_train : np.ndarray
            Regression targets, shape ``(n,)`` or ``(n, horizon)``.
        X_val, y_val : optional
            Validation arrays for early stopping.

        Returns
        -------
        dict
            ``{"train_loss": [...], "val_loss": [...]}``.
        """
        assert self.model is not None

        y_cls_train = self._prepare_cls_targets(y_train)
        train_loader = self._build_loader(X_train, y_train, y_cls_train, shuffle=True)

        val_loader: Optional[DataLoader] = None
        if X_val is not None and y_val is not None:
            y_cls_val = self._prepare_cls_targets(y_val)
            val_loader = self._build_loader(X_val, y_val, y_cls_val, shuffle=False)

        # Optimiser and scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(1, self.patience // 3),
        )

        mse_loss = nn.MSELoss()
        bce_loss = nn.BCELoss()

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_state: Optional[dict[str, Any]] = None
        epochs_no_improve = 0

        logger.info("Starting GRU training — %d epochs max, patience=%d, device=%s",
                     self.max_epochs, self.patience, self.device)

        for epoch in range(1, self.max_epochs + 1):
            # ---- Training ----
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for X_b, y_reg_b, y_cls_b in train_loader:
                X_b = X_b.to(self.device)
                y_reg_b = y_reg_b.to(self.device)
                y_cls_b = y_cls_b.to(self.device)

                reg_pred, cls_pred = self.model(X_b)

                # Align shapes
                if reg_pred.shape != y_reg_b.shape:
                    y_reg_b = y_reg_b[:, : reg_pred.shape[1]]
                if cls_pred.shape != y_cls_b.shape:
                    y_cls_b = y_cls_b[:, : cls_pred.shape[1]]

                loss = (
                    self.alpha * mse_loss(reg_pred, y_reg_b)
                    + (1 - self.alpha) * bce_loss(cls_pred, y_cls_b)
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_train)

            # ---- Validation ----
            avg_val = float("nan")
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_batches = 0
                with torch.no_grad():
                    for X_b, y_reg_b, y_cls_b in val_loader:
                        X_b = X_b.to(self.device)
                        y_reg_b = y_reg_b.to(self.device)
                        y_cls_b = y_cls_b.to(self.device)

                        reg_pred, cls_pred = self.model(X_b)

                        if reg_pred.shape != y_reg_b.shape:
                            y_reg_b = y_reg_b[:, : reg_pred.shape[1]]
                        if cls_pred.shape != y_cls_b.shape:
                            y_cls_b = y_cls_b[:, : cls_pred.shape[1]]

                        loss = (
                            self.alpha * mse_loss(reg_pred, y_reg_b)
                            + (1 - self.alpha) * bce_loss(cls_pred, y_cls_b)
                        )
                        val_loss += loss.item()
                        val_batches += 1

                avg_val = val_loss / max(val_batches, 1)
                history["val_loss"].append(avg_val)

                scheduler.step(avg_val)

                # Early stopping
                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.patience:
                    logger.info("Early stopping at epoch %d (best val_loss=%.6f)",
                                epoch, best_val_loss)
                    break

            if epoch % 10 == 0 or epoch == 1:
                logger.info("Epoch %03d/%03d — train_loss=%.6f  val_loss=%.6f",
                            epoch, self.max_epochs, avg_train, avg_val)

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        self.model.eval()
        logger.info("GRU training complete — %d epochs, best_val_loss=%.6f",
                     len(history["train_loss"]), best_val_loss)
        return history

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Return regression predictions as a numpy array.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n, lookback, features)``.

        Returns
        -------
        np.ndarray
            Shape ``(n, horizon)`` or ``(n,)`` when ``horizon == 1``.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        self.model.eval()
        x_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            reg_out, _ = self.model(x_tensor)

        result = reg_out.cpu().numpy()
        return result.squeeze() if self.horizon == 1 else result

    def predict_proba(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Return classification (sigmoid) predictions as a numpy array.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n, lookback, features)``.

        Returns
        -------
        np.ndarray
            Probabilities, shape ``(n, horizon)`` or ``(n,)``.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        self.model.eval()
        x_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, cls_out = self.model(x_tensor)

        result = cls_out.cpu().numpy()
        return result.squeeze() if self.horizon == 1 else result
