"""Train a single model on a specific symbol and timeframe.

Usage examples
--------------
Train LSTM on BTC hourly data::

    python scripts/run_training.py --model lstm --symbol BTC/USDT --timeframe 1h --lookback 24 --horizon 1

Train GRU on ETH 4-hour data with a longer lookback::

    python scripts/run_training.py --model gru --symbol ETH/USDT --timeframe 4h --lookback 48 --horizon 4

List available models::

    python scripts/run_training.py --list-models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path for ``src`` imports.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.registry import list_models
from src.training.experiment import Experiment
from src.utils.config import load_config
from src.utils.logger import get_logger


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a crypto price-prediction model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lstm",
        help="Model name from the registry (default: lstm). "
             "Use --list-models to see all options.",
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
        help="Number of historical time-steps for the input window (default: 24).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Number of future time-steps to forecast (default: 1).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print available model names and exit.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry-point: parse arguments, run experiment, report results."""
    args = _parse_args()
    logger = get_logger("run_training")

    # Quick listing mode
    if args.list_models:
        models = list_models()
        print("Available models:")
        for name in models:
            print(f"  - {name}")
        sys.exit(0)

    config = load_config()

    logger.info(
        "Starting training — model=%s, symbol=%s, timeframe=%s, "
        "lookback=%d, horizon=%d",
        args.model,
        args.symbol,
        args.timeframe,
        args.lookback,
        args.horizon,
    )

    experiment = Experiment(
        model_name=args.model,
        symbol=args.symbol,
        timeframe=args.timeframe,
        lookback=args.lookback,
        horizon=args.horizon,
        config=config,
    )

    try:
        results = experiment.run()
    except Exception:
        logger.exception("Experiment failed.")
        sys.exit(1)

    # Report final metrics
    logger.info("=" * 60)
    logger.info("FINAL RESULTS — %s", results["experiment_id"])
    logger.info("=" * 60)

    metrics = results["metrics"]
    for key, value in sorted(metrics.items()):
        logger.info("  %-30s : %.6f", key, value)

    logger.info("Checkpoint dir : %s", results["checkpoint_dir"])
    logger.info("Metrics file   : %s", results["metrics_path"])
    logger.info("Elapsed time   : %.1f seconds", results["elapsed_seconds"])


if __name__ == "__main__":
    main()
