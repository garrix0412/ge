"""Download historical OHLCV data for configured symbols and timeframes.

Usage examples
--------------
Download all symbols/timeframes from the config::

    python scripts/download_data.py

Download a single pair::

    python scripts/download_data.py --symbol BTC/USDT --timeframe 1h

Override the date range::

    python scripts/download_data.py --start 2023-01-01 --end 2023-12-31
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on ``sys.path`` so that ``src`` is importable
# regardless of the working directory the script is invoked from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.fetcher import DataFetcher
from src.utils.config import load_config
from src.utils.logger import get_logger


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download crypto OHLCV data from an exchange.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Single symbol to download (e.g. BTC/USDT). "
        "Defaults to all symbols in the config.",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=None,
        help="Single timeframe (e.g. 1h). "
        "Defaults to all timeframes in the config.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date override (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date override (YYYY-MM-DD).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry-point: resolve parameters and download data."""
    args = _parse_args()
    config = load_config()
    logger = get_logger(__name__)

    symbols: list[str] = [args.symbol] if args.symbol else config.data.symbols
    timeframes: list[str] = [args.timeframe] if args.timeframe else config.data.timeframes
    start: str = args.start or config.data.date_range.start
    end: str = args.end or config.data.date_range.end

    fetcher = DataFetcher(exchange_name=config.data.exchange.name)

    total = len(symbols) * len(timeframes)
    success_count = 0

    for symbol in symbols:
        for timeframe in timeframes:
            logger.info("Downloading %s %s from %s to %s", symbol, timeframe, start, end)
            try:
                fetcher.fetch_and_save(symbol, timeframe, start, end)
                logger.info("Successfully downloaded %s %s", symbol, timeframe)
                success_count += 1
            except Exception:
                logger.exception("Failed to download %s %s", symbol, timeframe)

    logger.info("Done: %d/%d downloads succeeded.", success_count, total)
    if success_count < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
