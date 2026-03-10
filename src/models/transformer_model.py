"""
Transformer-based dual-head model for crypto price prediction.

Uses a standard Transformer encoder with sinusoidal positional encoding,
causal masking, and dual regression + classification heads.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_model import BaseTorchModel
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class _TransformerNetwork(nn.Module):
    """Transformer encoder with dual regression/classification heads.

    Parameters
    ----------
    input_size : int
        Number of input features per time step.
    horizon : int
        Forecast horizon (number of future steps to predict).
    d_model : int
        Dimensionality of the transformer model.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of transformer encoder layers.
    dim_feedforward : int
        Hidden dimension of the feedforward network inside each layer.
    dropout : float
        Dropout probability applied throughout.
    max_len : int
        Maximum sequence length for positional encoding.
    """

    def __init__(
        self,
        input_size: int,
        horizon: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.horizon = horizon

        # ---- Input projection ----
        self.input_projection = nn.Linear(input_size, d_model)

        # ---- Sinusoidal positional encoding (fixed, not learnable) ----
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer so it moves with the model but is not a parameter
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        # ---- Transformer encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # ---- Dropout ----
        self.dropout = nn.Dropout(dropout)

        # ---- Dual heads ----
        self.regression_head = nn.Linear(d_model, horizon)
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, horizon),
            nn.Sigmoid(),
        )

    def _generate_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Upper-triangular causal mask to prevent attending to future positions.

        Returns a ``(seq_len, seq_len)`` float mask where ``-inf`` blocks
        attention and ``0.0`` allows it.
        """
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf")),
            diagonal=1,
        )
        return mask

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, seq_len, input_size)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(reg_out, cls_out)`` each of shape ``(batch, horizon)``.
        """
        seq_len = x.size(1)

        # Project input features to d_model
        x = self.input_projection(x) * math.sqrt(self.d_model)

        # Add positional encoding
        x = x + self.pe[:, :seq_len, :]
        x = self.dropout(x)

        # Causal mask
        causal_mask = self._generate_causal_mask(seq_len).to(x.device)

        # Transformer encoder
        encoded = self.transformer_encoder(x, mask=causal_mask)

        # Use last position output as sequence representation
        last_output = encoded[:, -1, :]  # (batch, d_model)

        reg_out = self.regression_head(last_output)       # (batch, horizon)
        cls_out = self.classification_head(last_output)    # (batch, horizon)

        return reg_out, cls_out

    def get_attention_weights(
        self, x: torch.Tensor
    ) -> list[torch.Tensor]:
        """Extract attention weights from each encoder layer.

        This requires a forward pass with hooks. Returns a list of tensors,
        one per encoder layer, each of shape ``(batch, nhead, seq_len, seq_len)``.
        """
        attention_weights: list[torch.Tensor] = []

        def _hook_fn(
            _module: nn.Module,
            _input: Any,
            output: Any,
        ) -> None:
            # nn.MultiheadAttention returns (attn_output, attn_weights)
            # when called internally, but TransformerEncoderLayer only
            # exposes the output.  We hook into the self_attn sub-module.
            pass  # pragma: no cover

        hooks = []
        for layer in self.transformer_encoder.layers:
            original_forward = layer.self_attn.forward

            def _make_hook(orig_fwd: Any) -> Any:
                def _wrapper(*args: Any, **kwargs: Any) -> Any:
                    kwargs["need_weights"] = True
                    kwargs["average_attn_weights"] = False
                    out = orig_fwd(*args, **kwargs)
                    attention_weights.append(out[1].detach())
                    return out
                return _wrapper

            layer.self_attn.forward = _make_hook(original_forward)  # type: ignore[assignment]
            hooks.append((layer.self_attn, original_forward))

        # Run forward pass
        seq_len = x.size(1)
        projected = self.input_projection(x) * math.sqrt(self.d_model)
        projected = projected + self.pe[:, :seq_len, :]
        projected = self.dropout(projected)
        causal_mask = self._generate_causal_mask(seq_len).to(x.device)
        self.transformer_encoder(projected, mask=causal_mask)

        # Restore original forward methods
        for self_attn, orig_fwd in hooks:
            self_attn.forward = orig_fwd  # type: ignore[assignment]

        return attention_weights


class TransformerModel(BaseTorchModel):
    """Dual-head Transformer model for crypto price prediction.

    Combines MSE regression loss with BCE classification loss using a
    configurable weighting factor ``alpha``.

    Parameters
    ----------
    input_size : int
        Number of input features per time step.
    horizon : int
        Forecast horizon (number of future steps to predict).
    lookback : int, optional
        Lookback window size (informational; not used directly).
    config : optional
        Pre-loaded :class:`AppConfig`.  If *None*, config is loaded
        from the default YAML files.
    """

    name: str = "transformer"

    def __init__(
        self,
        input_size: int,
        horizon: int = 1,
        lookback: int = 48,
        config: Any = None,
    ) -> None:
        super().__init__()
        self.cfg = config or load_config()
        self.input_size = input_size
        self.horizon = horizon
        self.lookback = lookback
        self._history: dict[str, list[float]] = {}

        # Build the model immediately so that num_parameters works
        self._build_model(input_size, horizon)
        self.auto_device()
        self._set_seed()

    def _set_seed(self) -> None:
        """Set random seeds for reproducibility."""
        seed = self.cfg.model.common.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_model(self, input_size: int, horizon: int) -> None:
        """Instantiate the Transformer network and move it to the device."""
        tcfg = self.cfg.model.transformer
        self.model = _TransformerNetwork(
            input_size=input_size,
            horizon=horizon,
            d_model=tcfg.d_model,
            nhead=tcfg.nhead,
            num_layers=tcfg.num_layers,
            dim_feedforward=tcfg.dim_feedforward,
            dropout=tcfg.dropout,
        )
        self.model.to(self.device)
        logger.info(
            "TransformerNetwork built: %d parameters", self.num_parameters
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        """Train the Transformer model.

        Parameters
        ----------
        X_train : np.ndarray
            Shape ``(n_samples, lookback, n_features)``.
        y_train : np.ndarray
            Shape ``(n_samples, horizon)`` for regression targets.  An
            optional ``y_cls_train`` may be passed via *kwargs* for
            classification targets; otherwise they are derived as
            ``(y_train > 0).astype(float)`` (price-up indicator).
        X_val, y_val : optional
            Validation arrays.

        Returns
        -------
        dict
            Keys: ``train_loss``, ``val_loss`` (and component losses).
        """
        # ----- Seed -----
        self._set_seed()

        assert self.model is not None

        # ----- Determine shapes -----
        horizon = y_train.shape[-1] if y_train.ndim > 1 else 1
        y_train = y_train.reshape(-1, horizon)

        # ----- Prepare classification labels -----
        y_cls_train = kwargs.get("y_cls_train", None)
        if y_cls_train is None:
            y_cls_train = (y_train > 0).astype(np.float32)
        y_cls_train = y_cls_train.reshape(-1, horizon)

        # ----- Create data loaders -----
        batch_size = self.cfg.model.common.batch_size
        train_ds = TensorDataset(
            torch.as_tensor(X_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.float32),
            torch.as_tensor(y_cls_train, dtype=torch.float32),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

        val_loader: Optional[DataLoader] = None
        if X_val is not None and y_val is not None:
            y_val = y_val.reshape(-1, horizon)
            y_cls_val = kwargs.get("y_cls_val", None)
            if y_cls_val is None:
                y_cls_val = (y_val > 0).astype(np.float32)
            y_cls_val = y_cls_val.reshape(-1, horizon)
            val_ds = TensorDataset(
                torch.as_tensor(X_val, dtype=torch.float32),
                torch.as_tensor(y_val, dtype=torch.float32),
                torch.as_tensor(y_cls_val, dtype=torch.float32),
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )

        # ----- Optimiser & scheduler -----
        lr = self.cfg.model.common.learning_rate
        max_epochs = self.cfg.model.common.max_epochs
        patience = self.cfg.model.common.patience
        alpha = self.cfg.model.dual_head.alpha

        optimizer = Adam(self.model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

        mse_loss_fn = nn.MSELoss()
        bce_loss_fn = nn.BCELoss()

        # ----- Training loop -----
        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_mse": [],
            "train_bce": [],
            "val_loss": [],
            "val_mse": [],
            "val_bce": [],
        }

        best_val_loss = float("inf")
        best_state_dict: Optional[dict[str, Any]] = None
        epochs_without_improvement = 0

        for epoch in range(1, max_epochs + 1):
            # --- Train ---
            self.model.train()
            epoch_loss = 0.0
            epoch_mse = 0.0
            epoch_bce = 0.0
            n_batches = 0

            for xb, yb_reg, yb_cls in train_loader:
                xb = xb.to(self.device)
                yb_reg = yb_reg.to(self.device)
                yb_cls = yb_cls.to(self.device)

                optimizer.zero_grad()
                reg_out, cls_out = self.model(xb)

                mse = mse_loss_fn(reg_out, yb_reg)
                bce = bce_loss_fn(cls_out, yb_cls)
                loss = alpha * mse + (1 - alpha) * bce

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_mse += mse.item()
                epoch_bce += bce.item()
                n_batches += 1

            scheduler.step()

            avg_train_loss = epoch_loss / n_batches
            avg_train_mse = epoch_mse / n_batches
            avg_train_bce = epoch_bce / n_batches
            history["train_loss"].append(avg_train_loss)
            history["train_mse"].append(avg_train_mse)
            history["train_bce"].append(avg_train_bce)

            # --- Validation ---
            if val_loader is not None:
                self.model.eval()
                val_loss_sum = 0.0
                val_mse_sum = 0.0
                val_bce_sum = 0.0
                val_batches = 0

                with torch.no_grad():
                    for xb, yb_reg, yb_cls in val_loader:
                        xb = xb.to(self.device)
                        yb_reg = yb_reg.to(self.device)
                        yb_cls = yb_cls.to(self.device)

                        reg_out, cls_out = self.model(xb)
                        mse = mse_loss_fn(reg_out, yb_reg)
                        bce = bce_loss_fn(cls_out, yb_cls)
                        loss = alpha * mse + (1 - alpha) * bce

                        val_loss_sum += loss.item()
                        val_mse_sum += mse.item()
                        val_bce_sum += bce.item()
                        val_batches += 1

                avg_val_loss = val_loss_sum / val_batches
                avg_val_mse = val_mse_sum / val_batches
                avg_val_bce = val_bce_sum / val_batches
            else:
                avg_val_loss = avg_train_loss
                avg_val_mse = avg_train_mse
                avg_val_bce = avg_train_bce

            history["val_loss"].append(avg_val_loss)
            history["val_mse"].append(avg_val_mse)
            history["val_bce"].append(avg_val_bce)

            logger.info(
                "Epoch %3d/%d | train_loss=%.6f | val_loss=%.6f | lr=%.2e",
                epoch,
                max_epochs,
                avg_train_loss,
                avg_val_loss,
                scheduler.get_last_lr()[0],
            )

            # --- Early stopping ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state_dict = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d)",
                        epoch,
                        patience,
                    )
                    break

        # Restore best weights
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
        self.model.eval()

        self._history = history
        logger.info(
            "Training complete. Best val_loss=%.6f", best_val_loss
        )
        return history

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Generate regression predictions.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_samples, lookback, n_features)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, horizon)``.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call fit() first.")

        self.model.eval()
        x_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            reg_out, _ = self.model(x_tensor)

        return reg_out.cpu().numpy()

    def get_attention_weights(self, X: np.ndarray) -> list[np.ndarray]:
        """Return attention weights from each encoder layer.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_samples, lookback, n_features)``.

        Returns
        -------
        list[np.ndarray]
            One array per layer, each of shape
            ``(n_samples, nhead, seq_len, seq_len)``.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call fit() first.")

        self.model.eval()
        x_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)

        assert isinstance(self.model, _TransformerNetwork)
        weights = self.model.get_attention_weights(x_tensor)
        return [w.cpu().numpy() for w in weights]
