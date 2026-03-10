"""
LSTM Autoencoder for anomaly detection in crypto price data.

Learns to reconstruct normal sequences; samples with high reconstruction
error are flagged as anomalies.  The threshold is calibrated as
``mean + threshold_sigma * std`` of reconstruction errors on the
training data.
"""

from __future__ import annotations

from pathlib import Path
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


class _LSTMAutoencoder(nn.Module):
    """LSTM-based autoencoder for sequence reconstruction.

    Architecture
    ------------
    1. **Encoder**: multi-layer LSTM that compresses the input sequence
       into a fixed-size hidden state.
    2. **Latent projection**: bottleneck ``hidden_size -> latent_dim ->
       hidden_size`` to enforce information compression.
    3. **Decoder**: multi-layer LSTM that reconstructs the original
       sequence from the latent representation.
    4. **Output projection**: linear layer that maps decoder hidden
       states back to the input feature space.

    Parameters
    ----------
    input_size : int
        Number of features per time step.
    hidden_size : int
        LSTM hidden dimensionality.
    latent_dim : int
        Bottleneck dimensionality.
    num_layers : int
        Number of stacked LSTM layers in both encoder and decoder.
    dropout : float
        Dropout probability between LSTM layers (only applied when
        ``num_layers > 1``).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # ---- Encoder ----
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ---- Latent bottleneck ----
        self.to_latent = nn.Linear(hidden_size, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_size)

        # ---- Decoder ----
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ---- Output projection ----
        self.output_projection = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and reconstruct the input sequence.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, seq_len, input_size)``.

        Returns
        -------
        torch.Tensor
            Reconstructed sequence with the same shape as input:
            ``(batch, seq_len, input_size)``.
        """
        batch_size, seq_len, _ = x.shape

        # ---- Encode ----
        _, (h_n, c_n) = self.encoder(x)
        # h_n shape: (num_layers, batch, hidden_size)

        # ---- Latent projection (use last layer hidden state) ----
        latent = self.to_latent(h_n[-1])          # (batch, latent_dim)
        decoded_hidden = self.from_latent(latent)  # (batch, hidden_size)

        # ---- Prepare decoder input ----
        # Repeat the latent representation across the sequence length
        decoder_input = decoded_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        # (batch, seq_len, hidden_size)

        # Initialise decoder hidden state from latent
        h_dec = decoded_hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_dec = torch.zeros_like(h_dec)

        # ---- Decode ----
        decoder_out, _ = self.decoder(decoder_input, (h_dec, c_dec))
        # (batch, seq_len, hidden_size)

        # ---- Output projection ----
        reconstructed = self.output_projection(decoder_out)
        # (batch, seq_len, input_size)

        return reconstructed


class AnomalyAutoencoder(BaseTorchModel):
    """LSTM Autoencoder for unsupervised anomaly detection.

    After training, samples whose reconstruction error exceeds
    ``mean + threshold_sigma * std`` (computed on the training set) are
    flagged as anomalies.

    Parameters
    ----------
    input_size : int
        Number of input features per time step.
    horizon : int, optional
        Forecast horizon (unused; kept for API compatibility with
        the model registry).
    lookback : int, optional
        Lookback window size (informational; not used directly).
    config : optional
        Pre-loaded :class:`AppConfig`.  If *None*, config is loaded
        from the default YAML files.
    """

    name: str = "anomaly"

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
        self.threshold: float = 0.0
        self._train_error_mean: float = 0.0
        self._train_error_std: float = 0.0
        self._history: dict[str, list[float]] = {}

        # Build the model immediately so num_parameters works
        self._build_model(input_size)
        self.auto_device()
        self._set_seed()

    def _set_seed(self) -> None:
        """Set random seeds for reproducibility."""
        seed = self.cfg.model.common.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_model(self, input_size: int) -> None:
        """Instantiate the LSTM autoencoder and move it to the device."""
        acfg = self.cfg.model.anomaly
        self.model = _LSTMAutoencoder(
            input_size=input_size,
            hidden_size=acfg.hidden_size,
            latent_dim=acfg.latent_dim,
            num_layers=acfg.num_layers,
            dropout=0.1,
        )
        self.model.to(self.device)
        logger.info(
            "LSTMAutoencoder built: %d parameters", self.num_parameters
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        """Train the autoencoder with MSE reconstruction loss.

        This is an unsupervised method: ``y_train`` is ignored.  The
        autoencoder learns to reconstruct ``X_train``.  After training,
        the anomaly threshold is computed from the reconstruction errors
        on the training data.

        Parameters
        ----------
        X_train : np.ndarray
            Shape ``(n_samples, seq_len, n_features)``.
        y_train : np.ndarray
            Ignored (kept for API compatibility).
        X_val, y_val : optional
            Validation data (``y_val`` is ignored).

        Returns
        -------
        dict
            Training history with keys ``train_loss`` and ``val_loss``.
        """
        # ----- Seed -----
        self._set_seed()

        assert self.model is not None

        # ----- Data loaders -----
        batch_size = self.cfg.model.common.batch_size
        train_ds = TensorDataset(
            torch.as_tensor(X_train, dtype=torch.float32),
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0
        )

        val_loader: Optional[DataLoader] = None
        if X_val is not None:
            val_ds = TensorDataset(
                torch.as_tensor(X_val, dtype=torch.float32),
            )
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False, num_workers=0
            )

        # ----- Optimiser & scheduler -----
        lr = self.cfg.model.common.learning_rate
        max_epochs = self.cfg.model.common.max_epochs
        patience = self.cfg.model.common.patience

        optimizer = Adam(self.model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
        mse_loss_fn = nn.MSELoss()

        # ----- Training loop -----
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

        best_val_loss = float("inf")
        best_state_dict: Optional[dict[str, Any]] = None
        epochs_without_improvement = 0

        for epoch in range(1, max_epochs + 1):
            # --- Train ---
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for (xb,) in train_loader:
                xb = xb.to(self.device)

                optimizer.zero_grad()
                reconstructed = self.model(xb)
                loss = mse_loss_fn(reconstructed, xb)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_train_loss = epoch_loss / n_batches
            history["train_loss"].append(avg_train_loss)

            # --- Validation ---
            if val_loader is not None:
                self.model.eval()
                val_loss_sum = 0.0
                val_batches = 0

                with torch.no_grad():
                    for (xb,) in val_loader:
                        xb = xb.to(self.device)
                        reconstructed = self.model(xb)
                        loss = mse_loss_fn(reconstructed, xb)
                        val_loss_sum += loss.item()
                        val_batches += 1

                avg_val_loss = val_loss_sum / val_batches
            else:
                avg_val_loss = avg_train_loss

            history["val_loss"].append(avg_val_loss)

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

        # ----- Compute anomaly threshold on training data -----
        train_errors = self.get_reconstruction_error(X_train)
        self._train_error_mean = float(np.mean(train_errors))
        self._train_error_std = float(np.std(train_errors))
        sigma = self.cfg.model.anomaly.threshold_sigma
        self.threshold = self._train_error_mean + sigma * self._train_error_std

        logger.info(
            "Anomaly threshold computed: %.6f (mean=%.6f, std=%.6f, sigma=%.1f)",
            self.threshold,
            self._train_error_mean,
            self._train_error_std,
            sigma,
        )

        self._history = history
        logger.info("Training complete. Best val_loss=%.6f", best_val_loss)
        return history

    def get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample mean reconstruction error.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_samples, seq_len, n_features)``.

        Returns
        -------
        np.ndarray
            Per-sample MSE, shape ``(n_samples,)``.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call fit() first.")

        self.model.eval()
        x_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            reconstructed = self.model(x_tensor)

        # Per-sample MSE: mean over seq_len and features
        errors = (x_tensor - reconstructed).pow(2).mean(dim=(1, 2))
        return errors.cpu().numpy()

    def detect_anomalies(self, X: np.ndarray) -> dict[str, Any]:
        """Detect anomalies based on reconstruction error.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_samples, seq_len, n_features)``.

        Returns
        -------
        dict
            Keys:
            - ``reconstruction_errors``: ``np.ndarray`` of shape ``(n_samples,)``
            - ``anomaly_flags``: ``np.ndarray`` of bool, shape ``(n_samples,)``
            - ``threshold``: ``float``
        """
        errors = self.get_reconstruction_error(X)
        anomaly_flags = errors > self.threshold

        logger.info(
            "Anomaly detection: %d / %d samples flagged (threshold=%.6f)",
            int(anomaly_flags.sum()),
            len(anomaly_flags),
            self.threshold,
        )

        return {
            "reconstruction_errors": errors,
            "anomaly_flags": anomaly_flags,
            "threshold": self.threshold,
        }

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict anomaly flags.

        Convenience wrapper around :meth:`detect_anomalies` that returns
        the reconstruction errors as the primary output (consistent with
        the ``BaseModel`` interface).

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_samples, seq_len, n_features)``.

        Returns
        -------
        np.ndarray
            Reconstruction errors, shape ``(n_samples,)``.
        """
        return self.get_reconstruction_error(X)

    # ------------------------------------------------------------------
    # Persistence (overrides to include threshold)
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save model state dict and anomaly threshold.

        The threshold metadata is stored alongside the model weights in
        a single checkpoint file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.model is None:
            raise RuntimeError("No model to save.")

        checkpoint = {
            "state_dict": self.model.state_dict(),
            "threshold": self.threshold,
            "train_error_mean": self._train_error_mean,
            "train_error_std": self._train_error_std,
        }
        torch.save(checkpoint, path)
        logger.info(
            "AnomalyAutoencoder saved -> %s (%d params, threshold=%.6f)",
            path,
            self.num_parameters,
            self.threshold,
        )

    def load(self, path: str | Path) -> None:
        """Load model state dict and anomaly threshold."""
        path = Path(path)
        if self.model is None:
            raise RuntimeError(
                "Build the model architecture before loading weights."
            )

        checkpoint = torch.load(
            path, map_location=self.device, weights_only=False
        )

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
            self.threshold = checkpoint.get("threshold", 0.0)
            self._train_error_mean = checkpoint.get("train_error_mean", 0.0)
            self._train_error_std = checkpoint.get("train_error_std", 0.0)
        else:
            # Fallback: plain state_dict (legacy format)
            self.model.load_state_dict(checkpoint)
            logger.warning(
                "Loaded legacy checkpoint without threshold metadata."
            )

        self.model.eval()
        logger.info(
            "AnomalyAutoencoder loaded <- %s (threshold=%.6f)",
            path,
            self.threshold,
        )
