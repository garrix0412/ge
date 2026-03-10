"""Train an anomaly-detection autoencoder and flag anomalous periods.

The script:

1. Loads processed data for a given symbol and timeframe.
2. Preprocesses and creates sequences.
3. Trains an ``AnomalyAutoencoder`` using its built-in ``fit()`` method.
4. Detects anomalies on the test set via ``detect_anomalies()``.
5. Saves results and per-sample errors to ``results/metrics/``.

Usage::

    python scripts/run_anomaly_detection.py --symbol BTC/USDT --timeframe 1h
    python scripts/run_anomaly_detection.py --symbol ETH/USDT --timeframe 4h --lookback 48
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path for ``src`` imports.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from src.data.preprocessor import DataPreprocessor
from src.models.registry import get_model
from src.utils.config import load_config
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

logger = get_logger("run_anomaly_detection")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train an anomaly-detection autoencoder on crypto data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/USDT",
        help="Trading pair symbol (default: BTC/USDT).",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Candle timeframe (default: 1h).",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=24,
        help="Lookback window for sequence construction (default: 24).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Forecast horizon for sequence construction (default: 1).",
    )
    parser.add_argument(
        "--threshold-sigma",
        type=float,
        default=None,
        help="Number of standard deviations for anomaly threshold. "
             "Defaults to model config value.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry-point: train autoencoder, compute errors, flag anomalies."""
    args = _parse_args()
    config = load_config()

    safe_symbol = args.symbol.replace("/", "_")
    experiment_id = (
        f"anomaly_{safe_symbol}_{args.timeframe}_lb{args.lookback}_h{args.horizon}"
    )

    start_time = time.time()
    logger.info("=" * 70)
    logger.info("Anomaly detection: %s", experiment_id)
    logger.info("=" * 70)

    # ---- 1. Load processed data ----
    data_path = PROCESSED_DIR / f"{safe_symbol}_{args.timeframe}_processed.parquet"
    logger.info("Loading data from %s", data_path)
    df = load_dataframe(data_path)

    # ---- 2. Preprocess ----
    preprocessor = DataPreprocessor(
        data_config=config.data,
        feature_config=config.features,
    )

    feature_cols: list[str] = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not feature_cols:
        logger.error("No matching feature columns in data.")
        sys.exit(1)

    split_cfg = config.data.split
    sequences = preprocessor.prepare_all(
        df=df,
        feature_columns=feature_cols,
        target_col=TARGET_COL,
        label_col=DIRECTION_COL,
        lookback=args.lookback,
        horizon=args.horizon,
        train_ratio=split_cfg.train,
        val_ratio=split_cfg.val,
        test_ratio=split_cfg.test,
    )

    # ---- 3. Instantiate anomaly model via registry ----
    n_features: int = sequences["X_train"].shape[2]

    model = get_model(
        "anomaly",
        input_size=n_features,
        lookback=args.lookback,
        horizon=args.horizon,
        config=config,
    )
    # auto_device() is already called inside AnomalyAutoencoder.__init__,
    # but call it explicitly to be safe.
    model.auto_device()

    logger.info(
        "Anomaly model initialised -- %d parameters, device=%s",
        model.num_parameters,
        model.device,
    )

    # ---- 4. Train using the model's own fit() method ----
    max_epochs: int = config.model.common.max_epochs
    history: dict[str, list[float]] = model.fit(
        X_train=sequences["X_train"],
        y_train=None,
        X_val=sequences["X_val"],
        y_val=None,
        max_epochs=max_epochs,
    )

    # ---- 5. Detect anomalies on the test set ----
    result: dict[str, Any] = model.detect_anomalies(sequences["X_test"])

    errors: np.ndarray = result["reconstruction_errors"]
    anomaly_flags: np.ndarray = result["anomaly_flags"]
    threshold: float = result["threshold"]

    mean_error = float(errors.mean())
    std_error = float(errors.std())
    n_anomalies = int(anomaly_flags.sum())
    anomaly_ratio = n_anomalies / len(errors) if len(errors) > 0 else 0.0

    # Override threshold if the user requested a custom sigma
    if args.threshold_sigma is not None:
        custom_threshold = mean_error + args.threshold_sigma * std_error
        anomaly_flags = (errors > custom_threshold).astype(int)
        n_anomalies = int(anomaly_flags.sum())
        anomaly_ratio = n_anomalies / len(errors) if len(errors) > 0 else 0.0
        threshold = custom_threshold
        logger.info(
            "Custom threshold applied (%.1f sigma): %.6f",
            args.threshold_sigma,
            threshold,
        )

    threshold_sigma: float = (
        args.threshold_sigma
        if args.threshold_sigma is not None
        else config.model.anomaly.threshold_sigma
    )

    logger.info("Reconstruction error -- mean: %.6f, std: %.6f", mean_error, std_error)
    logger.info("Threshold (%.1f sigma): %.6f", threshold_sigma, threshold)
    logger.info(
        "Anomalies detected: %d / %d (%.2f%%)",
        n_anomalies,
        len(errors),
        anomaly_ratio * 100,
    )

    # ---- 6. Save model checkpoint ----
    checkpoint_dir = CHECKPOINTS_DIR / experiment_id
    ensure_dir(checkpoint_dir)
    checkpoint_path = checkpoint_dir / "anomaly_model.pt"
    model.save(checkpoint_path)

    # ---- 7. Save results ----
    best_val_loss: float | None = None
    if history.get("val_loss"):
        best_val_loss = float(min(history["val_loss"]))

    results_dict: dict[str, Any] = {
        "experiment_id": experiment_id,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "threshold_sigma": threshold_sigma,
        "mean_reconstruction_error": mean_error,
        "std_reconstruction_error": std_error,
        "threshold": threshold,
        "n_test_samples": len(errors),
        "n_anomalies": n_anomalies,
        "anomaly_ratio": anomaly_ratio,
        "training_history": {
            "best_val_loss": best_val_loss,
            "total_epochs": len(history.get("train_loss", [])),
        },
    }

    metrics_path = METRICS_DIR / f"{experiment_id}_results.json"
    save_metrics(results_dict, metrics_path)

    # Also save per-sample errors and flags for downstream analysis
    errors_path = METRICS_DIR / f"{experiment_id}_errors.json"
    errors_data: dict[str, Any] = {
        "reconstruction_errors": errors.tolist(),
        "anomaly_flags": anomaly_flags.tolist(),
    }
    save_metrics(errors_data, errors_path)

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("Anomaly detection complete -- %s", experiment_id)
    logger.info(
        "  Anomalies  : %d / %d (%.2f%%)",
        n_anomalies,
        len(errors),
        anomaly_ratio * 100,
    )
    logger.info("  Threshold  : %.6f (%.1f sigma)", threshold, threshold_sigma)
    logger.info("  Checkpoint : %s", checkpoint_path)
    logger.info("  Results    : %s", metrics_path)
    logger.info("  Errors     : %s", errors_path)
    logger.info("  Elapsed    : %.1f seconds", elapsed)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
