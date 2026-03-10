"""
Historical OHLCV data downloader using the ccxt library.

Handles exchange pagination, rate limiting, and retries so that
callers can collect multi-year datasets with a single method call.

Usage::

    from src.data.fetcher import DataFetcher

    fetcher = DataFetcher()
    df = fetcher.fetch_ohlcv("BTC/USDT", "1h", "2021-01-01", "2024-12-31")
    fetcher.save_raw(df, "BTC/USDT", "1h")
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd

from src.utils.config import DataConfig
from src.utils.constants import OHLCV_COLUMNS, RAW_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataFetcher:
    """Download historical OHLCV candles from a cryptocurrency exchange.

    Parameters
    ----------
    exchange_name:
        Any exchange identifier supported by ccxt (default ``"binance"``).
    api_key:
        Optional API key.  Falls back to the ``BINANCE_API_KEY``
        environment variable when *None*.
    api_secret:
        Optional API secret.  Falls back to ``BINANCE_SECRET``.
    config:
        An optional :class:`DataConfig` instance.  When *None* the default
        configuration is loaded from disk / defaults.
    """

    # Maximum retries for a single paginated request.
    MAX_RETRIES: int = 3

    def __init__(
        self,
        exchange_name: str = "binance",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        config: Optional[DataConfig] = None,
    ) -> None:
        self.config = config or DataConfig.load()

        resolved_key = api_key or os.getenv("BINANCE_API_KEY")
        resolved_secret = api_secret or os.getenv("BINANCE_SECRET")

        # Try to initialise the ccxt exchange; set to None on failure so
        # the yfinance fallback can still run.
        self.exchange: ccxt.Exchange | None = None
        self.exchange_name = exchange_name

        try:
            exchange_class = getattr(ccxt, exchange_name, None)
            if exchange_class is None:
                raise ValueError(f"Exchange '{exchange_name}' is not supported by ccxt.")

            exchange_params: dict = {
                "enableRateLimit": True,
            }
            if resolved_key and resolved_secret:
                exchange_params["apiKey"] = resolved_key
                exchange_params["secret"] = resolved_secret

            self.exchange = exchange_class(exchange_params)
        except Exception as exc:
            logger.warning(
                "Failed to initialise ccxt exchange '%s': %s. "
                "Will rely on fallback data sources.",
                exchange_name,
                exc,
            )

        # Rate-limit delay derived from config (seconds between requests).
        rps = self.config.exchange.rate_limit.requests_per_second
        self._request_delay: float = 1.0 / max(rps, 1)

        logger.info(
            "DataFetcher initialised – exchange=%s (available=%s), rate_limit=%.2f req/s",
            exchange_name,
            self.exchange is not None,
            rps,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data with automatic pagination.

        Parameters
        ----------
        symbol:
            Trading pair, e.g. ``"BTC/USDT"``.
        timeframe:
            Candle interval, e.g. ``"1h"``, ``"4h"``, ``"1d"``.
        start_date:
            ISO-format start date (inclusive), e.g. ``"2021-01-01"``.
        end_date:
            ISO-format end date (inclusive), e.g. ``"2024-12-31"``.

        Returns
        -------
        pd.DataFrame
            Columns: ``timestamp, open, high, low, close, volume``.
            ``timestamp`` is of dtype ``datetime64[ns, UTC]``.
        """
        if self.exchange is None:
            logger.error("Cannot fetch via ccxt – exchange not initialised.")
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        since_ms = self.exchange.parse8601(f"{start_date}T00:00:00Z")
        end_ms = self.exchange.parse8601(f"{end_date}T23:59:59Z")
        limit = 1000  # most exchanges cap at 1 000 candles per request

        all_candles: list[list] = []
        current_since = since_ms

        logger.info(
            "Fetching %s %s from %s to %s …",
            symbol,
            timeframe,
            start_date,
            end_date,
        )

        while current_since < end_ms:
            candles = self._fetch_with_retry(symbol, timeframe, current_since, limit)

            if not candles:
                logger.debug("No more candles returned – stopping pagination.")
                break

            # Filter out candles beyond end_date.
            candles = [c for c in candles if c[0] <= end_ms]
            all_candles.extend(candles)

            # Advance *since* to just after the last candle's timestamp.
            last_ts = candles[-1][0]
            if last_ts == current_since:
                # Safety guard against infinite loop.
                break
            current_since = last_ts + 1

            logger.debug(
                "Fetched %d candles (total so far: %d).",
                len(candles),
                len(all_candles),
            )

            # Respect exchange rate limits.
            time.sleep(self._request_delay)

        if not all_candles:
            logger.warning("No data returned for %s %s.", symbol, timeframe)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        df = pd.DataFrame(all_candles, columns=OHLCV_COLUMNS)

        # Convert millisecond timestamp to datetime.
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        # Drop potential duplicates from overlapping pagination windows.
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        logger.info(
            "Completed: %s %s – %d candles (%s to %s).",
            symbol,
            timeframe,
            len(df),
            df["timestamp"].iloc[0],
            df["timestamp"].iloc[-1],
        )

        return df

    def save_raw(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Persist a raw OHLCV DataFrame to CSV.

        Parameters
        ----------
        df:
            DataFrame produced by :meth:`fetch_ohlcv`.
        symbol:
            Trading pair, e.g. ``"BTC/USDT"``.
        timeframe:
            Candle interval, e.g. ``"1h"``.
        output_dir:
            Target directory.  Defaults to ``data/raw``.

        Returns
        -------
        Path
            Absolute path to the written CSV file.
        """
        directory = Path(output_dir) if output_dir else RAW_DIR
        directory.mkdir(parents=True, exist_ok=True)

        safe_symbol = self._sanitize_symbol(symbol)
        filename = f"{safe_symbol}_{timeframe}.csv"
        filepath = directory / filename

        df.to_csv(filepath, index=False)
        logger.info("Saved raw data to %s (%d rows).", filepath, len(df))
        return filepath

    def fetch_and_save(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        output_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Convenience method: fetch OHLCV data and save it to disk.

        Parameters
        ----------
        symbol:
            Trading pair.
        timeframe:
            Candle interval.
        start_date:
            ISO-format start date.
        end_date:
            ISO-format end date.
        output_dir:
            Optional directory override for saving.

        Returns
        -------
        pd.DataFrame
            The fetched DataFrame (also persisted to CSV).
        """
        # 1) Try ccxt (primary source).
        try:
            df = self.fetch_ohlcv(symbol, timeframe, start_date, end_date)
        except Exception as exc:
            logger.warning("ccxt fetch failed for %s %s: %s", symbol, timeframe, exc)
            df = pd.DataFrame(columns=OHLCV_COLUMNS)

        # 2) Fallback to yfinance if ccxt returned nothing.
        if df.empty and "yfinance" in self.config.exchange.fallback_sources:
            logger.info("Trying yfinance fallback for %s %s …", symbol, timeframe)
            try:
                from src.data.yfinance_fetcher import YFinanceFetcher

                yf_fetcher = YFinanceFetcher()
                df = yf_fetcher.fetch_ohlcv(symbol, timeframe, start_date, end_date)
            except Exception as exc:
                logger.warning("yfinance fallback also failed: %s", exc)
                df = pd.DataFrame(columns=OHLCV_COLUMNS)

        if not df.empty:
            self.save_raw(df, symbol, timeframe, output_dir=output_dir)
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_with_retry(
        self,
        symbol: str,
        timeframe: str,
        since: int,
        limit: int,
    ) -> list[list]:
        """Fetch a single page of candles with retry logic.

        Returns an empty list when all retries are exhausted.
        """
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=limit,
                )
                return candles
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as exc:
                delay = self.config.exchange.rate_limit.retry_delay_seconds * attempt
                logger.warning(
                    "Transient error (attempt %d/%d): %s – retrying in %.1fs …",
                    attempt,
                    self.MAX_RETRIES,
                    exc,
                    delay,
                )
                time.sleep(delay)
            except ccxt.ExchangeError as exc:
                logger.error("Exchange error (non-retryable): %s", exc)
                return []

        logger.error(
            "All %d retries exhausted for %s %s since=%d.",
            self.MAX_RETRIES,
            symbol,
            timeframe,
            since,
        )
        return []

    @staticmethod
    def _sanitize_symbol(symbol: str) -> str:
        """Replace ``/`` with ``_`` so the symbol is filesystem-safe."""
        return symbol.replace("/", "_")
