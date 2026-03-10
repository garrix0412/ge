"""
Walk-forward (rolling / expanding window) validation for time-series models.

Ensures strict temporal ordering: training data always precedes validation
data, and no future information leaks into the model.

Usage::

    from src.training.walk_forward import WalkForwardValidator

    wfv = WalkForwardValidator(n_splits=5, min_train_size=500)
    results = wfv.validate(
        model_class=LSTMModel,
        data=df,
        feature_cols=FEATURE_COLUMNS,
        target_col="close",
        label_col="direction",
        lookback=24,
        horizon=1,
    )
"""

from __future__ import annotations

from typing import Any, Generator, Optional

import numpy as np
import pandas as pd

from src.data.dataset import create_dataloaders
from src.data.preprocessor import DataPreprocessor
from src.models.base_model import BaseModel
from src.training.evaluator import Evaluator
from src.training.trainer import Trainer
from src.utils.config import AppConfig, load_config
from src.utils.constants import DIRECTION_COL, FEATURE_COLUMNS, TARGET_COL
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WalkForwardValidator:
    """Walk-forward (rolling / expanding window) cross-validation.

    Parameters
    ----------
    n_splits:
        Number of train/validation folds.
    min_train_size:
        Minimum number of rows in the training window.  This guarantees
        that early folds have enough data to learn from.
    expanding:
        If *True* (default), the training window grows with each fold
        (anchored at the beginning of the dataset).  If *False*, a
        fixed-size rolling window is used instead.
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_size: int = 500,
        expanding: bool = True,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if min_train_size < 1:
            raise ValueError(f"min_train_size must be >= 1, got {min_train_size}")

        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.expanding = expanding

        logger.info(
            "WalkForwardValidator — n_splits=%d, min_train_size=%d, expanding=%s",
            n_splits,
            min_train_size,
            expanding,
        )

    # ------------------------------------------------------------------
    # Split generator
    # ------------------------------------------------------------------

    def split(
        self,
        data: pd.DataFrame | np.ndarray,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Yield ``(train_indices, val_indices)`` for each fold.

        Guarantees:
        - Strict temporal ordering: ``train_indices.max() < val_indices.min()``
        - No overlap between train and validation sets.
        - Training window respects ``min_train_size``.

        Parameters
        ----------
        data:
            The full dataset (only its length is used).

        Yields
        ------
        tuple[np.ndarray, np.ndarray]
            Integer index arrays for training and validation rows.
        """
        n = len(data)

        # Reserve enough room so that every fold has a validation segment.
        # Total size after min_train is divided into n_splits validation
        # segments.
        remaining = n - self.min_train_size
        if remaining <= 0:
            raise ValueError(
                f"Dataset length ({n}) must exceed min_train_size "
                f"({self.min_train_size})."
            )

        fold_size = remaining // self.n_splits
        if fold_size < 1:
            raise ValueError(
                f"Not enough data for {self.n_splits} folds.  "
                f"Reduce n_splits or min_train_size."
            )

        for fold_idx in range(self.n_splits):
            val_start = self.min_train_size + fold_idx * fold_size
            val_end = val_start + fold_size
            # Last fold may extend to the end of the dataset
            if fold_idx == self.n_splits - 1:
                val_end = n

            if self.expanding:
                train_start = 0
            else:
                # Fixed-size rolling window — keep the training window
                # the same size as the initial training window.
                train_start = max(0, val_start - self.min_train_size)

            train_indices = np.arange(train_start, val_start)
            val_indices = np.arange(val_start, val_end)

            # Sanity check: strict temporal ordering
            assert train_indices[-1] < val_indices[0], (
                f"Temporal ordering violation at fold {fold_idx}: "
                f"train_max={train_indices[-1]}, val_min={val_indices[0]}"
            )

            logger.info(
                "Fold %d/%d — train: [%d, %d) (%d rows), val: [%d, %d) (%d rows)",
                fold_idx + 1,
                self.n_splits,
                train_start,
                val_start,
                len(train_indices),
                val_start,
                val_end,
                len(val_indices),
            )
            yield train_indices, val_indices

    # ------------------------------------------------------------------
    # Full validation loop
    # ------------------------------------------------------------------

    def validate(
        self,
        model_class: type,
        data: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        label_col: str,
        lookback: int,
        horizon: int,
        config: Optional[AppConfig] = None,
        **model_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Run walk-forward validation across all folds.

        For each fold the method:
        1. Splits into training and validation data.
        2. Fits a scaler on the training data.
        3. Creates sequences.
        4. Instantiates and trains a fresh model.
        5. Evaluates and records per-fold metrics.

        Parameters
        ----------
        model_class:
            The model *class* (not an instance).  A fresh instance is
            created for each fold.
        data:
            Full feature-engineered DataFrame (NaN rows already dropped).
        feature_cols:
            Column names used as model input features.
        target_col:
            Regression target column (e.g. ``"close"``).
        label_col:
            Classification label column (e.g. ``"direction"``).
        lookback:
            Sequence lookback window.
        horizon:
            Forecast horizon.
        config:
            Application configuration.  Loaded from disk when *None*.
        **model_kwargs:
            Extra keyword arguments forwarded to the model constructor.

        Returns
        -------
        list[dict[str, Any]]
            One dict per fold with keys ``fold``, ``train_size``,
            ``val_size``, and all metric names returned by
            :meth:`Evaluator.evaluate_all`.
        """
        config = config or load_config()
        fold_results: list[dict[str, Any]] = []

        logger.info(
            "Starting walk-forward validation — %d folds, lookback=%d, horizon=%d",
            self.n_splits,
            lookback,
            horizon,
        )

        for fold_idx, (train_idx, val_idx) in enumerate(self.split(data)):
            logger.info("-" * 60)
            logger.info("Fold %d/%d", fold_idx + 1, self.n_splits)
            logger.info("-" * 60)

            train_df = data.iloc[train_idx].reset_index(drop=True)
            val_df = data.iloc[val_idx].reset_index(drop=True)

            # Fit scaler on fold training data
            preprocessor = DataPreprocessor(
                data_config=config.data,
                feature_config=config.features,
            )
            preprocessor.fit_scaler(train_df, feature_cols)

            # Determine close column index
            close_col_idx = feature_cols.index(target_col) if target_col in feature_cols else 3

            # Scale
            train_scaled = preprocessor.transform(train_df, feature_cols)
            val_scaled = preprocessor.transform(val_df, feature_cols)

            train_labels = train_df[label_col].values
            val_labels = val_df[label_col].values

            # Create sequences
            try:
                X_train, y_train_reg, y_train_cls = preprocessor.create_sequences(
                    train_scaled, train_labels, lookback, horizon, close_col_idx,
                )
                X_val, y_val_reg, y_val_cls = preprocessor.create_sequences(
                    val_scaled, val_labels, lookback, horizon, close_col_idx,
                )
            except ValueError as exc:
                logger.warning("Fold %d skipped — %s", fold_idx + 1, exc)
                continue

            # Build DataLoaders
            batch_size = config.model.common.batch_size
            train_loader, val_loader = create_dataloaders(
                X_train, y_train_reg, y_train_cls,
                X_val, y_val_reg, y_val_cls,
                batch_size=batch_size,
            )

            # Initialise a fresh model
            n_features = X_train.shape[2]
            model = model_class(
                input_size=n_features,
                lookback=lookback,
                horizon=horizon,
                config=config,
                **model_kwargs,
            )

            # Determine device
            device_cfg = config.model.common.device
            if device_cfg == "auto":
                if hasattr(model, "auto_device"):
                    model.auto_device()
                    device = str(model.device)
                else:
                    device = "cpu"
            else:
                device = device_cfg
                if hasattr(model, "to"):
                    model.to(device)

            # Train
            trainer = Trainer(model=model, config=config, device=device)
            history = trainer.train(train_loader=train_loader, val_loader=val_loader)

            # Evaluate on fold validation set
            import torch

            assert model.model is not None
            model.model.eval()

            all_pred_reg, all_pred_cls = [], []
            all_y_reg, all_y_cls = [], []

            with torch.no_grad():
                for x_b, yr_b, yc_b in val_loader:
                    x_b = x_b.to(model.device)
                    reg_out, cls_out = model.model(x_b)
                    all_pred_reg.append(reg_out.cpu().numpy())
                    all_pred_cls.append(cls_out.cpu().numpy())
                    all_y_reg.append(yr_b.numpy())
                    all_y_cls.append(yc_b.numpy())

            import numpy as np

            y_true_reg = np.concatenate(all_y_reg).ravel()
            y_pred_reg = np.concatenate(all_pred_reg).ravel()
            y_true_cls = np.concatenate(all_y_cls).ravel()
            y_pred_cls = np.concatenate(all_pred_cls).ravel()

            metrics = Evaluator.evaluate_all(y_true_reg, y_pred_reg, y_true_cls, y_pred_cls)

            fold_result: dict[str, Any] = {
                "fold": fold_idx + 1,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "best_val_loss": history.get("best_val_loss"),
                **metrics,
            }
            fold_results.append(fold_result)

            logger.info("Fold %d metrics: %s", fold_idx + 1, metrics)

        # Aggregate summary
        if fold_results:
            logger.info("=" * 60)
            logger.info("Walk-forward validation summary (%d folds):", len(fold_results))
            metric_keys = [k for k in fold_results[0] if k.startswith(("reg_", "cls_"))]
            for key in metric_keys:
                values = [r[key] for r in fold_results if key in r]
                if values:
                    logger.info(
                        "  %s — mean: %.6f, std: %.6f",
                        key,
                        np.mean(values),
                        np.std(values),
                    )
            logger.info("=" * 60)

        return fold_results
