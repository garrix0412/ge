"""
yfinance-based fallback fetcher for OHLCV data.

Used when ccxt (Binance) is unreachable (e.g. from China).
Maps crypto pairs like BTC/USDT to yfinance tickers like BTC-USD.

Limitation: yfinance hourly data only covers ~730 days.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from src.utils.constants import OHLCV_COLUMNS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Mapping from ccxt-style symbols to yfinance tickers.
_SYMBOL_MAP: dict[str, str] = {
    "BTC/USDT": "BTC-USD",
    "ETH/USDT": "ETH-USD",
    "BNB/USDT": "BNB-USD",
    "SOL/USDT": "SOL-USD",
    "XRP/USDT": "XRP-USD",
    "ADA/USDT": "ADA-USD",
    "DOGE/USDT": "DOGE-USD",
    "DOT/USDT": "DOT-USD",
    "AVAX/USDT": "AVAX-USD",
    "MATIC/USDT": "MATIC-USD",
}

# yfinance interval strings matching our timeframe convention.
_TIMEFRAME_MAP: dict[str, str] = {
    "1h": "1h",
    "4h": "1h",  # yfinance has no native 4h; we resample from 1h
    "1d": "1d",
}

# yfinance caps intraday data at ~730 days.
_MAX_INTRADAY_DAYS = 729


class YFinanceFetcher:
    """Download OHLCV data via yfinance as a fallback source."""

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data from Yahoo Finance.

        Parameters
        ----------
        symbol:
            Trading pair in ccxt format, e.g. ``"BTC/USDT"``.
        timeframe:
            Candle interval: ``"1h"``, ``"4h"``, or ``"1d"``.
        start_date:
            ISO-format start date, e.g. ``"2021-01-01"``.
        end_date:
            ISO-format end date, e.g. ``"2024-12-31"``.

        Returns
        -------
        pd.DataFrame
            Columns: ``[timestamp, open, high, low, close, volume]``.
            ``timestamp`` is ``datetime64[ns, UTC]``.
        """
        import yfinance as yf  # lazy import

        ticker = self._map_symbol(symbol)
        if ticker is None:
            logger.warning("No yfinance mapping for symbol '%s'.", symbol)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        interval = _TIMEFRAME_MAP.get(timeframe)
        if interval is None:
            logger.warning("Unsupported timeframe '%s' for yfinance.", timeframe)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)

        # yfinance limits intraday data to ~730 days from today.
        if interval != "1d":
            earliest_allowed = datetime.now() - timedelta(days=_MAX_INTRADAY_DAYS)
            if start_dt < earliest_allowed:
                logger.warning(
                    "yfinance hourly data limited to ~730 days. "
                    "Adjusting start from %s to %s.",
                    start_date,
                    earliest_allowed.strftime("%Y-%m-%d"),
                )
                start_dt = earliest_allowed

        logger.info(
            "Fetching %s (%s) via yfinance, interval=%s, %s to %s …",
            symbol,
            ticker,
            interval,
            start_dt.strftime("%Y-%m-%d"),
            end_dt.strftime("%Y-%m-%d"),
        )

        yf_ticker = yf.Ticker(ticker)
        df = yf_ticker.history(
            start=start_dt.strftime("%Y-%m-%d"),
            end=(end_dt + timedelta(days=1)).strftime("%Y-%m-%d"),  # end is exclusive
            interval=interval,
            auto_adjust=True,
        )

        if df.empty:
            logger.warning("yfinance returned no data for %s.", ticker)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        # Resample to 4h if needed.
        if timeframe == "4h" and interval == "1h":
            df = df.resample("4h").agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }).dropna()

        # Normalize to our standard format.
        df = df.reset_index()
        ts_col = "Datetime" if "Datetime" in df.columns else "Date"
        df = df.rename(columns={
            ts_col: "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        df = df[OHLCV_COLUMNS]

        # Ensure UTC timezone.
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(
            "yfinance: fetched %d candles for %s %s (%s to %s).",
            len(df),
            symbol,
            timeframe,
            df["timestamp"].iloc[0],
            df["timestamp"].iloc[-1],
        )

        return df

    @staticmethod
    def _map_symbol(symbol: str) -> str | None:
        """Convert a ccxt symbol to a yfinance ticker."""
        if symbol in _SYMBOL_MAP:
            return _SYMBOL_MAP[symbol]
        # Generic fallback: BTC/USDT → BTC-USD, XXX/USDT → XXX-USD
        if symbol.endswith("/USDT"):
            base = symbol.split("/")[0]
            return f"{base}-USD"
        return None
