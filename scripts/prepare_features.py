"""Process raw OHLCV CSVs into feature-engineered parquet files.

For each configured symbol and timeframe the script:

1. Loads the raw CSV from ``data/raw/{BTC_USDT}_{1h}.csv``.
2. Applies ``FeatureEngineer.add_all_features()`` to compute technical
   indicators and derived columns.
3. Drops warm-up NaN rows via ``FeatureEngineer.drop_na()``.
4. Saves the result to ``data/processed/{BTC_USDT}_{1h}_processed.parquet``.

Usage::

    python scripts/prepare_features.py
    python scripts/prepare_features.py --symbol BTC/USDT --timeframe 1h
"""

from __future__ import annotations

import argparse
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

from src.data.feature_engineer import FeatureEngineer
from src.utils.config import load_config
from src.utils.constants import PROCESSED_DIR, RAW_DIR
from src.utils.io import save_dataframe
from src.utils.logger import get_logger

logger = get_logger("prepare_features")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process raw OHLCV CSVs into feature-engineered parquet files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Single symbol to process (e.g. BTC/USDT). "
             "Defaults to all symbols in the config.",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=None,
        help="Single timeframe to process (e.g. 1h). "
             "Defaults to all timeframes in the config.",
    )
    return parser.parse_args()


def process_single(
    symbol: str,
    timeframe: str,
    feature_engineer: FeatureEngineer,
) -> bool:
    """Process a single symbol/timeframe combination.

    Parameters
    ----------
    symbol:
        Trading pair, e.g. ``"BTC/USDT"``.
    timeframe:
        Candle interval, e.g. ``"1h"``.
    feature_engineer:
        Pre-configured :class:`FeatureEngineer` instance.

    Returns
    -------
    bool
        ``True`` if processing succeeded, ``False`` otherwise.
    """
    safe_symbol: str = symbol.replace("/", "_")
    raw_filename: str = f"{safe_symbol}_{timeframe}.csv"
    raw_path: Path = RAW_DIR / raw_filename

    out_filename: str = f"{safe_symbol}_{timeframe}_processed.parquet"
    out_path: Path = PROCESSED_DIR / out_filename

    # -- Load raw CSV --
    if not raw_path.exists():
        logger.error("Raw data file not found: %s", raw_path)
        return False

    logger.info("Loading raw CSV: %s", raw_path)
    df: pd.DataFrame = pd.read_csv(raw_path)
    logger.info("  Loaded %d rows, %d columns.", len(df), len(df.columns))

    if df.empty:
        logger.warning("Raw data is empty for %s %s -- skipping.", symbol, timeframe)
        return False

    # -- Feature engineering --
    logger.info("Computing features for %s %s ...", symbol, timeframe)
    df = feature_engineer.add_all_features(df)

    # -- Drop NaN rows from indicator warm-up --
    df = feature_engineer.drop_na(df)

    if df.empty:
        logger.warning(
            "All rows were NaN after feature engineering for %s %s -- skipping.",
            symbol,
            timeframe,
        )
        return False

    # -- Save processed parquet --
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_dataframe(df, out_path)
    logger.info(
        "Saved processed data: %s (%d rows, %d columns).",
        out_path,
        len(df),
        len(df.columns),
    )
    return True


def main() -> None:
    """Entry-point: process raw CSVs into feature-engineered parquet files."""
    args = _parse_args()
    config = load_config()

    symbols: list[str] = (
        [args.symbol] if args.symbol else config.data.symbols
    )
    timeframes: list[str] = (
        [args.timeframe] if args.timeframe else config.data.timeframes
    )

    feature_engineer = FeatureEngineer(config=config.features)

    total: int = len(symbols) * len(timeframes)
    success_count: int = 0
    fail_count: int = 0

    start_time = time.time()

    logger.info("=" * 70)
    logger.info("Feature preparation pipeline")
    logger.info("  Symbols    : %s", symbols)
    logger.info("  Timeframes : %s", timeframes)
    logger.info("  Total      : %d", total)
    logger.info("=" * 70)

    for symbol in symbols:
        for timeframe in timeframes:
            try:
                ok = process_single(symbol, timeframe, feature_engineer)
                if ok:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception:
                logger.exception(
                    "Failed to process %s %s.", symbol, timeframe,
                )
                fail_count += 1

    elapsed = time.time() - start_time

    logger.info("=" * 70)
    logger.info("Feature preparation complete")
    logger.info("  Succeeded : %d / %d", success_count, total)
    logger.info("  Failed    : %d / %d", fail_count, total)
    logger.info("  Elapsed   : %.1f seconds", elapsed)
    logger.info("=" * 70)

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
