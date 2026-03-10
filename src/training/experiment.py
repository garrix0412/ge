"""
Single-experiment orchestrator for crypto price prediction.

Coordinates data loading, preprocessing, model creation, training,
evaluation, and artifact persistence in a single reproducible run.

Usage::

    from src.training.experiment import Experiment

    exp = Experiment(model_name="lstm", symbol="BTC/USDT", timeframe="1h",
                     lookback=24, horizon=1)
    results = exp.run()
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import create_dataloaders, create_test_loader
from src.data.preprocessor import DataPreprocessor
from src.models.base_model import BaseModel, BaseTorchModel
from src.models.registry import get_model
from src.training.evaluator import Evaluator
from src.training.trainer import Trainer
from src.utils.config import AppConfig, load_config
from src.utils.constants import (
    CHECKPOINTS_DIR,
    DIRECTION_COL,
    FEATURE_COLUMNS,
    METRICS_DIR,
    PROCESSED_DIR,
    TARGET_COL,
)
from src.utils.io import ensure_dir, load_dataframe, save_metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Experiment:
    """Orchestrate a single model-training experiment.

    Parameters
    ----------
    model_name:
        Key in the model registry (e.g. ``"lstm"``, ``"gru"``).
    symbol:
        Trading pair (e.g. ``"BTC/USDT"``).
    timeframe:
        Candle interval (e.g. ``"1h"``).
    lookback:
        Number of historical time-steps fed to the model.
    horizon:
        Number of future time-steps to predict.
    config:
        Full application config.  When *None* the defaults are loaded.
    """

    def __init__(
        self,
        model_name: str,
        symbol: str,
        timeframe: str,
        lookback: int,
        horizon: int,
        config: Optional[AppConfig] = None,
    ) -> None:
        self.model_name = model_name
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback = lookback
        self.horizon = horizon
        self.config = config or load_config()

        self.experiment_id = self._build_experiment_id()

        logger.info("Experiment created: %s", self.experiment_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Execute the full experiment pipeline.

        Steps
        -----
        1. Load processed data (parquet).
        2. Create sequences via ``DataPreprocessor``.
        3. Initialise the model from the registry.
        4. Train and evaluate (PyTorch or sklearn-style, depending on model type).
        5. Save checkpoint, metrics JSON, and training history.

        Returns
        -------
        dict[str, Any]
            Experiment results including metrics, history, and paths.
        """
        start_time = time.time()
        logger.info("=" * 70)
        logger.info("Running experiment: %s", self.experiment_id)
        logger.info("=" * 70)

        # ---- 1. Load processed data ----
        safe_symbol = self.symbol.replace("/", "_")
        data_path = PROCESSED_DIR / f"{safe_symbol}_{self.timeframe}_processed.parquet"
        logger.info("Loading data from %s", data_path)
        df = load_dataframe(data_path)

        # ---- 2. Create sequences ----
        preprocessor = DataPreprocessor(
            data_config=self.config.data,
            feature_config=self.config.features,
        )

        # Filter feature columns to only those present in the dataframe
        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        if not feature_cols:
            raise ValueError(
                f"No matching feature columns found in data. "
                f"Expected some of {FEATURE_COLUMNS}, got {list(df.columns)}"
            )

        split_cfg = self.config.data.split
        sequences = preprocessor.prepare_all(
            df=df,
            feature_columns=feature_cols,
            target_col=TARGET_COL,
            label_col=DIRECTION_COL,
            lookback=self.lookback,
            horizon=self.horizon,
            train_ratio=split_cfg.train,
            val_ratio=split_cfg.val,
            test_ratio=split_cfg.test,
        )

        # ---- 3. Initialise model ----
        n_features = sequences["X_train"].shape[2]
        device = self._resolve_device()

        model = get_model(
            self.model_name,
            input_size=n_features,
            lookback=self.lookback,
            horizon=self.horizon,
            config=self.config,
        )

        is_torch = isinstance(model, BaseTorchModel)

        # ---- 4 & 5 & 6. Train and evaluate ----
        checkpoint_dir = CHECKPOINTS_DIR / self.experiment_id

        if is_torch:
            # ----- PyTorch path (LSTM, GRU, Transformer, TFT) -----
            model.to(device)
            logger.info(
                "Model initialised: %s (%d parameters, device=%s)",
                self.model_name,
                model.num_parameters,
                device,
            )

            # Build DataLoaders
            batch_size = self.config.model.common.batch_size
            num_workers = self.config.model.common.num_workers

            train_loader, val_loader = create_dataloaders(
                X_train=sequences["X_train"],
                y_train_reg=sequences["y_train_reg"],
                y_train_cls=sequences["y_train_cls"],
                X_val=sequences["X_val"],
                y_val_reg=sequences["y_val_reg"],
                y_val_cls=sequences["y_val_cls"],
                batch_size=batch_size,
                num_workers=num_workers,
            )
            test_loader = create_test_loader(
                X_test=sequences["X_test"],
                y_test_reg=sequences["y_test_reg"],
                y_test_cls=sequences["y_test_cls"],
                batch_size=batch_size,
                num_workers=num_workers,
            )

            # Train
            trainer = Trainer(model=model, config=self.config, device=device)
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                checkpoint_dir=checkpoint_dir,
            )

            # Evaluate on test set
            metrics = self._evaluate_test_torch(model, test_loader, device)
        else:
            # ----- Non-PyTorch path (ARIMA, XGBoost) -----
            logger.info("Model initialised: %s (non-PyTorch)", self.model_name)

            X_train = sequences["X_train"]
            y_train_reg = sequences["y_train_reg"]
            y_train_cls = sequences["y_train_cls"]
            X_val = sequences["X_val"]
            y_val_reg = sequences["y_val_reg"]
            X_test = sequences["X_test"]
            y_test_reg = sequences["y_test_reg"]
            y_test_cls = sequences["y_test_cls"]

            if getattr(model, "name", "") == "arima":
                history = model.fit(X_train, y_train_reg)
                y_pred = model.predict(X_test)
            else:
                # XGBoost and other sklearn-style models
                history = model.fit(
                    X_train, y_train_reg, X_val, y_val_reg,
                    y_train_cls=y_train_cls,
                )
                y_pred = model.predict(X_test)

            # Evaluate with sklearn-style method
            metrics = self._evaluate_test_sklearn(
                model=model,
                X_test=X_test,
                y_pred_reg=y_pred,
                y_test_reg=y_test_reg,
                y_test_cls=y_test_cls,
            )

            # Save non-torch model checkpoint
            ensure_dir(checkpoint_dir)
            checkpoint_path = checkpoint_dir / f"{self.model_name}_model.pkl"
            model.save(checkpoint_path)
            logger.info("Non-PyTorch model saved -> %s", checkpoint_path)

        # ---- 7. Save artifacts ----
        metrics_path = METRICS_DIR / f"{self.experiment_id}_metrics.json"
        history_path = METRICS_DIR / f"{self.experiment_id}_history.json"

        save_metrics(metrics, metrics_path)
        save_metrics(history, history_path)

        # Save the scaler alongside the checkpoint
        scaler_path = checkpoint_dir / "scaler.joblib"
        preprocessor.save_scaler(scaler_path)

        elapsed = time.time() - start_time
        logger.info(
            "Experiment %s completed in %.1f seconds.", self.experiment_id, elapsed,
        )

        results: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "lookback": self.lookback,
            "horizon": self.horizon,
            "metrics": metrics,
            "history": history,
            "checkpoint_dir": str(checkpoint_dir),
            "metrics_path": str(metrics_path),
            "elapsed_seconds": elapsed,
        }
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_experiment_id(self) -> str:
        """Generate a descriptive, filesystem-safe experiment identifier."""
        safe_symbol = self.symbol.replace("/", "_")
        return (
            f"{self.model_name}_{safe_symbol}_{self.timeframe}"
            f"_lb{self.lookback}_h{self.horizon}"
        )

    def _resolve_device(self) -> str:
        """Pick the best available device based on config."""
        device_cfg = self.config.model.common.device
        if device_cfg == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device_cfg

    def _evaluate_test_torch(
        self,
        model: BaseTorchModel,
        test_loader: DataLoader,
        device: str,
    ) -> dict[str, float]:
        """Run inference on the test set for PyTorch dual-head models.

        Parameters
        ----------
        model : Trained BaseTorchModel instance.
        test_loader : Test ``DataLoader``.
        device : Target device string.

        Returns
        -------
        dict[str, float]
            Combined regression and classification metrics.
        """
        assert model.model is not None
        model.model.eval()

        all_y_reg: list[np.ndarray] = []
        all_y_cls: list[np.ndarray] = []
        all_pred_reg: list[np.ndarray] = []
        all_pred_cls: list[np.ndarray] = []

        torch_device = torch.device(device)

        with torch.no_grad():
            for x_batch, y_reg_batch, y_cls_batch in test_loader:
                x_batch = x_batch.to(torch_device)

                reg_out, cls_out = model.model(x_batch)

                # TFT returns (batch, horizon, n_quantiles); extract median for eval.
                if reg_out.ndim == 3:
                    reg_out = reg_out[:, :, 1]
                all_pred_reg.append(reg_out.cpu().numpy())
                all_pred_cls.append(cls_out.cpu().numpy())
                all_y_reg.append(y_reg_batch.numpy())
                all_y_cls.append(y_cls_batch.numpy())

        y_true_reg = np.concatenate(all_y_reg, axis=0).ravel()
        y_pred_reg = np.concatenate(all_pred_reg, axis=0).ravel()
        y_true_cls = np.concatenate(all_y_cls, axis=0).ravel()
        y_pred_cls = np.concatenate(all_pred_cls, axis=0).ravel()

        metrics = Evaluator.evaluate_all(y_true_reg, y_pred_reg, y_true_cls, y_pred_cls)
        return metrics

    def _evaluate_test_sklearn(
        self,
        model: BaseModel,
        X_test: np.ndarray,
        y_pred_reg: np.ndarray,
        y_test_reg: np.ndarray,
        y_test_cls: np.ndarray,
    ) -> dict[str, float]:
        """Compute metrics for non-PyTorch models (ARIMA, XGBoost, etc.).

        Regression metrics are always computed.  Classification metrics
        are added when the model exposes a ``predict_cls`` method.

        Parameters
        ----------
        model : Trained BaseModel instance.
        X_test : Test input array (3-D sequences or 2-D features).
        y_pred_reg : Regression predictions already produced by the model.
        y_test_reg : Ground-truth regression targets.
        y_test_cls : Ground-truth classification labels.

        Returns
        -------
        dict[str, float]
            Metrics dict.  Regression keys are prefixed with ``reg_``;
            classification keys (if available) with ``cls_``.
        """
        # Regression metrics
        reg_metrics = Evaluator.evaluate_regression(
            y_test_reg.ravel(), y_pred_reg.ravel(),
        )
        metrics: dict[str, float] = {
            f"reg_{k}": v for k, v in reg_metrics.items()
        }

        # Classification metrics (only if model supports it)
        if hasattr(model, "predict_cls"):
            try:
                y_pred_cls = model.predict_cls(X_test)
                cls_metrics = Evaluator.evaluate_classification(
                    y_test_cls.ravel(), y_pred_cls.ravel(),
                )
                metrics.update({f"cls_{k}": v for k, v in cls_metrics.items()})
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Classification evaluation failed for %s: %s",
                    self.model_name,
                    exc,
                )

        logger.info("Sklearn-style model metrics: %s", metrics)
        return metrics
