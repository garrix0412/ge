"""
Temporal Fusion Transformer (TFT) for crypto price prediction.

Implements variable selection, gated residual networks, LSTM temporal
processing, multi-head attention, and quantile regression with an
auxiliary classification head.
"""

from __future__ import annotations

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


# =========================================================================
# Building blocks
# =========================================================================


class GatedLinearUnit(nn.Module):
    """GLU activation: ``sigmoid(Wx + b) * (Vx + c)``.

    Parameters
    ----------
    input_size : int
        Dimensionality of the input.
    output_size : int
        Dimensionality of the output.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.fc1(x)) * self.fc2(x)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN).

    Applies a two-layer ELU network, followed by a GLU gate, layer
    normalisation, and a skip connection.  An optional *context* vector
    can condition the hidden layer.

    Parameters
    ----------
    input_size : int
        Feature dimensionality of the primary input.
    hidden_size : int
        Hidden dimensionality.
    output_size : int
        Output dimensionality.
    context_size : int, optional
        If provided, a linear projection of the context vector is added
        to the first hidden layer.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        context_size: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.context_projection = (
            nn.Linear(context_size, hidden_size, bias=False)
            if context_size is not None
            else None
        )
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gate = GatedLinearUnit(hidden_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)

        # Skip connection: project if input_size != output_size
        self.skip_projection = (
            nn.Linear(input_size, output_size)
            if input_size != output_size
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        if self.skip_projection is not None:
            residual = self.skip_projection(residual)

        hidden = self.fc1(x)
        if self.context_projection is not None and context is not None:
            hidden = hidden + self.context_projection(context)
        hidden = self.elu(hidden)

        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        hidden = self.gate(hidden)

        return self.layer_norm(hidden + residual)


class VariableSelectionNetwork(nn.Module):
    """Softmax-weighted sum of per-variable GRN transformations.

    Given input ``x`` with shape ``(batch, seq_len, n_features)``, this
    module treats each feature as a separate ``(batch, seq_len, 1)``
    variable, transforms it via a dedicated GRN, then computes softmax
    weights and returns the weighted sum.

    Parameters
    ----------
    n_features : int
        Number of input features.
    hidden_size : int
        GRN hidden / output dimensionality.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size

        # One GRN per input variable
        self.variable_grns = nn.ModuleList(
            [
                GatedResidualNetwork(
                    input_size=1,
                    hidden_size=hidden_size,
                    output_size=hidden_size,
                    dropout=dropout,
                )
                for _ in range(n_features)
            ]
        )

        # Softmax weight GRN (flattened input -> n_features logits)
        self.weight_grn = GatedResidualNetwork(
            input_size=n_features,
            hidden_size=hidden_size,
            output_size=n_features,
            dropout=dropout,
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, seq_len, n_features)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(selected_features, weights)`` where ``selected_features``
            has shape ``(batch, seq_len, hidden_size)`` and ``weights``
            has shape ``(batch, seq_len, n_features)``.
        """
        # Compute variable selection weights
        weights = self.softmax(self.weight_grn(x))  # (batch, seq_len, n_features)

        # Transform each variable independently
        transformed = []
        for i, grn in enumerate(self.variable_grns):
            var_input = x[:, :, i : i + 1]  # (batch, seq_len, 1)
            transformed.append(grn(var_input))  # (batch, seq_len, hidden_size)

        # Stack: (batch, seq_len, n_features, hidden_size)
        transformed_stack = torch.stack(transformed, dim=2)

        # Weighted sum over features
        weights_expanded = weights.unsqueeze(-1)  # (batch, seq_len, n_features, 1)
        selected = (transformed_stack * weights_expanded).sum(dim=2)  # (batch, seq_len, hidden_size)

        return selected, weights


# =========================================================================
# Main TFT network
# =========================================================================


class _TFTNetwork(nn.Module):
    """Temporal Fusion Transformer network.

    Parameters
    ----------
    input_size : int
        Number of input features per time step.
    horizon : int
        Forecast horizon.
    hidden_size : int
        Hidden dimensionality used throughout.
    num_attention_heads : int
        Number of heads in the multi-head attention layer.
    dropout : float
        Dropout probability.
    n_quantiles : int
        Number of quantile outputs.
    """

    def __init__(
        self,
        input_size: int,
        horizon: int,
        hidden_size: int = 128,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        n_quantiles: int = 3,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.n_quantiles = n_quantiles

        # ---- Variable selection ----
        self.variable_selection = VariableSelectionNetwork(
            n_features=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        # ---- LSTM encoder ----
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        # ---- Gated skip connection after LSTM ----
        self.lstm_gate = GatedLinearUnit(hidden_size, hidden_size)
        self.lstm_layer_norm = nn.LayerNorm(hidden_size)

        # ---- Multi-head attention ----
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ---- Post-attention gated residual ----
        self.attention_gate = GatedLinearUnit(hidden_size, hidden_size)
        self.attention_layer_norm = nn.LayerNorm(hidden_size)

        # ---- Position-wise GRN ----
        self.positionwise_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )

        # ---- Output heads ----
        self.dropout_layer = nn.Dropout(dropout)

        # Quantile output
        self.quantile_head = nn.Linear(hidden_size, horizon * n_quantiles)

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, horizon),
            nn.Sigmoid(),
        )

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
            ``(quantile_output, cls_output)`` where ``quantile_output``
            has shape ``(batch, horizon, n_quantiles)`` and ``cls_output``
            has shape ``(batch, horizon)``.
        """
        batch_size = x.size(0)

        # ---- Variable selection ----
        selected, _vsn_weights = self.variable_selection(x)  # (batch, seq_len, hidden_size)

        # ---- LSTM temporal encoding ----
        lstm_out, _ = self.lstm_encoder(selected)  # (batch, seq_len, hidden_size)

        # Gated skip connection around LSTM
        lstm_gated = self.lstm_gate(lstm_out)
        temporal = self.lstm_layer_norm(lstm_gated + selected)  # (batch, seq_len, hidden_size)

        # ---- Multi-head self-attention ----
        attn_out, _attn_weights = self.multihead_attention(
            query=temporal,
            key=temporal,
            value=temporal,
            need_weights=True,
            average_attn_weights=False,
        )

        # Gated residual after attention
        attn_gated = self.attention_gate(attn_out)
        enriched = self.attention_layer_norm(attn_gated + temporal)

        # ---- Position-wise feedforward ----
        output = self.positionwise_grn(enriched)  # (batch, seq_len, hidden_size)

        # Use last time step as summary
        last_output = output[:, -1, :]  # (batch, hidden_size)
        last_output = self.dropout_layer(last_output)

        # ---- Quantile output ----
        quantile_flat = self.quantile_head(last_output)  # (batch, horizon * n_quantiles)
        quantile_output = quantile_flat.view(batch_size, self.horizon, self.n_quantiles)

        # ---- Classification output ----
        cls_output = self.classification_head(last_output)  # (batch, horizon)

        return quantile_output, cls_output

    def get_attention_weights(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Return temporal attention weights.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, num_heads, seq_len, seq_len)``.
        """
        selected, _ = self.variable_selection(x)
        lstm_out, _ = self.lstm_encoder(selected)
        lstm_gated = self.lstm_gate(lstm_out)
        temporal = self.lstm_layer_norm(lstm_gated + selected)

        _, attn_weights = self.multihead_attention(
            query=temporal,
            key=temporal,
            value=temporal,
            need_weights=True,
            average_attn_weights=False,
        )
        return attn_weights

    def get_variable_selection_weights(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Return variable selection weights.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, seq_len, n_features)``.
        """
        _, weights = self.variable_selection(x)
        return weights


# =========================================================================
# TFTModel
# =========================================================================


class TFTModel(BaseTorchModel):
    """Temporal Fusion Transformer with quantile regression and
    dual-head classification loss for crypto price prediction.

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

    name: str = "tft"

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
        self._quantiles: list[float] = list(self.cfg.model.tft.quantiles)
        self._history: dict[str, list[float]] = {}

        # Build the model immediately so num_parameters works
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
        """Instantiate the TFT network and move it to the device."""
        tft_cfg = self.cfg.model.tft
        self.model = _TFTNetwork(
            input_size=input_size,
            horizon=horizon,
            hidden_size=tft_cfg.hidden_size,
            num_attention_heads=tft_cfg.num_attention_heads,
            dropout=tft_cfg.dropout,
            n_quantiles=len(self._quantiles),
        )
        self.model.to(self.device)
        logger.info("TFTNetwork built: %d parameters", self.num_parameters)

    @staticmethod
    def _quantile_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        quantiles: list[float],
    ) -> torch.Tensor:
        """Compute combined quantile (pinball) loss.

        Parameters
        ----------
        predictions : torch.Tensor
            Shape ``(batch, horizon, n_quantiles)``.
        targets : torch.Tensor
            Shape ``(batch, horizon)``.
        quantiles : list[float]
            Quantile levels, e.g. ``[0.1, 0.5, 0.9]``.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        targets = targets.unsqueeze(-1)  # (batch, horizon, 1)
        errors = targets - predictions   # (batch, horizon, n_quantiles)

        losses = []
        for i, q in enumerate(quantiles):
            e = errors[:, :, i]
            loss_q = torch.max(q * e, (q - 1.0) * e)
            losses.append(loss_q)

        return torch.stack(losses, dim=-1).mean()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        """Train the TFT model.

        Parameters
        ----------
        X_train : np.ndarray
            Shape ``(n_samples, lookback, n_features)``.
        y_train : np.ndarray
            Shape ``(n_samples, horizon)`` regression targets.
        X_val, y_val : optional
            Validation data.

        Returns
        -------
        dict
            Training history.
        """
        # ----- Seed -----
        self._set_seed()

        assert self.model is not None

        # ----- Shapes -----
        horizon = y_train.shape[-1] if y_train.ndim > 1 else 1
        y_train = y_train.reshape(-1, horizon)

        # ----- Classification labels -----
        y_cls_train = kwargs.get("y_cls_train", None)
        if y_cls_train is None:
            y_cls_train = (y_train > 0).astype(np.float32)
        y_cls_train = y_cls_train.reshape(-1, horizon)

        # ----- Data loaders -----
        batch_size = self.cfg.model.common.batch_size
        train_ds = TensorDataset(
            torch.as_tensor(X_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.float32),
            torch.as_tensor(y_cls_train, dtype=torch.float32),
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0
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
                val_ds, batch_size=batch_size, shuffle=False, num_workers=0
            )

        # ----- Optimiser & scheduler -----
        lr = self.cfg.model.common.learning_rate
        max_epochs = self.cfg.model.common.max_epochs
        patience = self.cfg.model.common.patience
        alpha = self.cfg.model.dual_head.alpha

        optimizer = Adam(self.model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
        bce_loss_fn = nn.BCELoss()

        # ----- Training loop -----
        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_quantile": [],
            "train_bce": [],
            "val_loss": [],
            "val_quantile": [],
            "val_bce": [],
        }

        best_val_loss = float("inf")
        best_state_dict: Optional[dict[str, Any]] = None
        epochs_without_improvement = 0

        for epoch in range(1, max_epochs + 1):
            # --- Train ---
            self.model.train()
            epoch_loss = 0.0
            epoch_ql = 0.0
            epoch_bce = 0.0
            n_batches = 0

            for xb, yb_reg, yb_cls in train_loader:
                xb = xb.to(self.device)
                yb_reg = yb_reg.to(self.device)
                yb_cls = yb_cls.to(self.device)

                optimizer.zero_grad()
                quantile_out, cls_out = self.model(xb)

                ql = self._quantile_loss(quantile_out, yb_reg, self._quantiles)
                bce = bce_loss_fn(cls_out, yb_cls)
                loss = alpha * ql + (1 - alpha) * bce

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_ql += ql.item()
                epoch_bce += bce.item()
                n_batches += 1

            scheduler.step()

            avg_train_loss = epoch_loss / n_batches
            avg_train_ql = epoch_ql / n_batches
            avg_train_bce = epoch_bce / n_batches
            history["train_loss"].append(avg_train_loss)
            history["train_quantile"].append(avg_train_ql)
            history["train_bce"].append(avg_train_bce)

            # --- Validation ---
            if val_loader is not None:
                self.model.eval()
                val_loss_sum = 0.0
                val_ql_sum = 0.0
                val_bce_sum = 0.0
                val_batches = 0

                with torch.no_grad():
                    for xb, yb_reg, yb_cls in val_loader:
                        xb = xb.to(self.device)
                        yb_reg = yb_reg.to(self.device)
                        yb_cls = yb_cls.to(self.device)

                        quantile_out, cls_out = self.model(xb)
                        ql = self._quantile_loss(
                            quantile_out, yb_reg, self._quantiles
                        )
                        bce = bce_loss_fn(cls_out, yb_cls)
                        loss = alpha * ql + (1 - alpha) * bce

                        val_loss_sum += loss.item()
                        val_ql_sum += ql.item()
                        val_bce_sum += bce.item()
                        val_batches += 1

                avg_val_loss = val_loss_sum / val_batches
                avg_val_ql = val_ql_sum / val_batches
                avg_val_bce = val_bce_sum / val_batches
            else:
                avg_val_loss = avg_train_loss
                avg_val_ql = avg_train_ql
                avg_val_bce = avg_train_bce

            history["val_loss"].append(avg_val_loss)
            history["val_quantile"].append(avg_val_ql)
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
        logger.info("Training complete. Best val_loss=%.6f", best_val_loss)
        return history

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Return median (q=0.5) predictions.

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
            quantile_out, _ = self.model(x_tensor)

        # Find the index of the median quantile (0.5)
        try:
            median_idx = self._quantiles.index(0.5)
        except ValueError:
            # Fallback: use the middle quantile
            median_idx = len(self._quantiles) // 2

        return quantile_out[:, :, median_idx].cpu().numpy()

    def predict_quantiles(self, X: np.ndarray) -> np.ndarray:
        """Return all quantile predictions for confidence intervals.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_samples, lookback, n_features)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, horizon, n_quantiles)``.  With the default
            quantiles ``[0.1, 0.5, 0.9]``, the last axis corresponds to
            (q10, q50, q90).
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call fit() first.")

        self.model.eval()
        x_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            quantile_out, _ = self.model(x_tensor)

        return quantile_out.cpu().numpy()

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Return variable selection weights averaged over a recent batch.

        These are stored during the last forward pass.  For a fresh
        computation, call :meth:`get_feature_importance_from_data`.

        Returns
        -------
        np.ndarray or None
            Mean variable selection weights of shape ``(n_features,)``.
        """
        logger.warning(
            "get_feature_importance() requires data; "
            "use get_feature_importance_from_data(X) instead."
        )
        return None

    def get_feature_importance_from_data(
        self, X: np.ndarray
    ) -> np.ndarray:
        """Compute variable selection weights for the given data.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_samples, lookback, n_features)``.

        Returns
        -------
        np.ndarray
            Mean variable selection weights, shape ``(n_features,)``.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call fit() first.")

        self.model.eval()
        x_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)

        assert isinstance(self.model, _TFTNetwork)
        with torch.no_grad():
            weights = self.model.get_variable_selection_weights(x_tensor)

        # Average over batch and time -> (n_features,)
        return weights.mean(dim=(0, 1)).cpu().numpy()

    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """Return temporal attention weights.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_samples, lookback, n_features)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, num_heads, seq_len, seq_len)``.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call fit() first.")

        self.model.eval()
        x_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)

        assert isinstance(self.model, _TFTNetwork)
        with torch.no_grad():
            attn_weights = self.model.get_attention_weights(x_tensor)

        return attn_weights.cpu().numpy()
