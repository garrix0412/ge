"""Run all experiment combinations and produce a comparison table.

Iterates over every combination of model, symbol, timeframe, lookback, and
horizon specified in the project configuration files.  Results are written
to ``results/metrics/comparison_table.csv``.

The script is **resumable**: if a metrics JSON already exists for a given
experiment ID, that experiment is skipped.

Usage::

    python scripts/run_all_experiments.py

Override which models to run::

    python scripts/run_all_experiments.py --models lstm gru transformer
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path for ``src`` imports.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

from src.models.registry import list_models
from src.training.experiment import Experiment
from src.utils.config import load_config
from src.utils.constants import METRICS_DIR
from src.utils.io import load_metrics
from src.utils.logger import get_logger

logger = get_logger("run_all_experiments")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run all experiment combinations and save a comparison table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Override the list of model names (default: all registered models).",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Override the list of symbols (default: from data config).",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=None,
        help="Override the list of timeframes (default: from data config).",
    )
    parser.add_argument(
        "--lookbacks",
        nargs="+",
        type=int,
        default=None,
        help="Override lookback windows (default: from feature config).",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=None,
        help="Override forecast horizons (default: from feature config).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for the comparison CSV (default: results/metrics/comparison_table.csv).",
    )
    return parser.parse_args()


def _experiment_id(model: str, symbol: str, timeframe: str, lookback: int, horizon: int) -> str:
    """Reproduce the experiment ID generation logic."""
    safe_symbol = symbol.replace("/", "_")
    return f"{model}_{safe_symbol}_{timeframe}_lb{lookback}_h{horizon}"


def main() -> None:
    """Entry-point: enumerate combinations, run experiments, save table."""
    args = _parse_args()
    config = load_config()

    # Resolve experiment axes
    models = args.models or list_models()
    symbols = args.symbols or config.data.symbols
    timeframes = args.timeframes or config.data.timeframes
    lookbacks = args.lookbacks or config.features.lookback_windows
    horizons = args.horizons or config.features.forecast_horizons

    output_path = Path(args.output) if args.output else METRICS_DIR / "comparison_table.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combinations = list(itertools.product(models, symbols, timeframes, lookbacks, horizons))
    total = len(combinations)

    logger.info("=" * 70)
    logger.info("Batch experiment runner")
    logger.info("  Models     : %s", models)
    logger.info("  Symbols    : %s", symbols)
    logger.info("  Timeframes : %s", timeframes)
    logger.info("  Lookbacks  : %s", lookbacks)
    logger.info("  Horizons   : %s", horizons)
    logger.info("  Total combinations: %d", total)
    logger.info("=" * 70)

    all_results: list[dict] = []
    success_count = 0
    skip_count = 0
    fail_count = 0

    start_time = time.time()

    for model_name, symbol, timeframe, lookback, horizon in tqdm(
        combinations, desc="Experiments", unit="exp"
    ):
        exp_id = _experiment_id(model_name, symbol, timeframe, lookback, horizon)
        metrics_path = METRICS_DIR / f"{exp_id}_metrics.json"

        # ---- Skip if already completed ----
        if metrics_path.exists():
            logger.info("Skipping %s (metrics already exist).", exp_id)
            try:
                existing = load_metrics(metrics_path)
                row = {
                    "experiment_id": exp_id,
                    "model": model_name,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "lookback": lookback,
                    "horizon": horizon,
                    "status": "cached",
                    **existing,
                }
                all_results.append(row)
            except Exception:
                logger.warning("Could not load cached metrics for %s.", exp_id)
            skip_count += 1
            continue

        # ---- Run experiment ----
        logger.info("Running %s …", exp_id)
        try:
            experiment = Experiment(
                model_name=model_name,
                symbol=symbol,
                timeframe=timeframe,
                lookback=lookback,
                horizon=horizon,
                config=config,
            )
            results = experiment.run()
            row = {
                "experiment_id": exp_id,
                "model": model_name,
                "symbol": symbol,
                "timeframe": timeframe,
                "lookback": lookback,
                "horizon": horizon,
                "status": "success",
                "elapsed_seconds": results["elapsed_seconds"],
                **results["metrics"],
            }
            all_results.append(row)
            success_count += 1
        except Exception:
            logger.exception("Experiment %s FAILED.", exp_id)
            row = {
                "experiment_id": exp_id,
                "model": model_name,
                "symbol": symbol,
                "timeframe": timeframe,
                "lookback": lookback,
                "horizon": horizon,
                "status": "failed",
            }
            all_results.append(row)
            fail_count += 1

    # ---- Save comparison table ----
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_path, index=False)
        logger.info("Comparison table saved -> %s (%d rows)", output_path, len(df))
    else:
        logger.warning("No results to save.")

    # ---- Summary ----
    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("Batch run complete")
    logger.info("  Total      : %d", total)
    logger.info("  Succeeded  : %d", success_count)
    logger.info("  Cached     : %d", skip_count)
    logger.info("  Failed     : %d", fail_count)
    logger.info("  Elapsed    : %.1f seconds", elapsed)
    logger.info("  Output     : %s", output_path)
    logger.info("=" * 70)

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
