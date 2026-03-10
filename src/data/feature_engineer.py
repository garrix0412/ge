"""
Technical indicator computation and derived feature generation.

All indicator parameters are driven by :class:`FeatureConfig` so that
experiments can be reproduced with different settings by simply swapping
the YAML file.

Usage::

    from src.data.feature_engineer import FeatureEngineer

    fe = FeatureEngineer()
    df = fe.add_all_features(ohlcv_df)
    df = fe.drop_na(df)
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import pandas_ta as ta

from src.utils.config import FeatureConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Compute technical indicators and derived features on an OHLCV DataFrame.

    Parameters
    ----------
    config:
        A :class:`FeatureConfig` instance.  When *None* the default is
        loaded from disk / built-in defaults.
    """

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        self.config = config or FeatureConfig.load()
        logger.info("FeatureEngineer initialised with config: %s", self.config.model_dump())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add every configured technical indicator and derived feature.

        The input DataFrame must contain at least the columns
        ``open, high, low, close, volume``.  A *copy* is returned so that
        the original is never mutated.

        Parameters
        ----------
        df:
            OHLCV DataFrame.

        Returns
        -------
        pd.DataFrame
            The original columns plus all newly computed features.
        """
        df = df.copy()
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Input DataFrame is missing required columns: {missing}")

        logger.info("Computing features on DataFrame with %d rows …", len(df))

        df = self._add_rsi(df)
        df = self._add_macd(df)
        df = self._add_bollinger_bands(df)
        df = self._add_ema(df)
        df = self._add_atr(df)
        df = self._add_log_returns(df)
        df = self._add_price_change_pct(df)
        df = self._add_volume_change_pct(df)
        df = self._add_high_low_range(df)
        df = self._add_open_close_range(df)
        df = self._add_direction_label(df)

        logger.info(
            "Feature computation complete – %d columns total.",
            len(df.columns),
        )
        return df

    def drop_na(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows containing NaN values (typically from indicator warm-up).

        Parameters
        ----------
        df:
            DataFrame with computed features.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with the index reset.
        """
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        after = len(df)
        logger.info(
            "Dropped %d NaN rows (%d -> %d).",
            before - after,
            before,
            after,
        )
        return df

    # ------------------------------------------------------------------
    # Private indicator methods
    # ------------------------------------------------------------------

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Relative Strength Index."""
        period = self.config.indicators.rsi.period
        df["rsi"] = ta.rsi(df["close"], length=period)
        logger.debug("Added RSI(%d).", period)
        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD line, signal line, and histogram."""
        cfg = self.config.indicators.macd
        macd_df = ta.macd(
            df["close"],
            fast=cfg.fast_period,
            slow=cfg.slow_period,
            signal=cfg.signal_period,
        )
        # pandas_ta returns columns like MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
        col_macd = f"MACD_{cfg.fast_period}_{cfg.slow_period}_{cfg.signal_period}"
        col_signal = f"MACDs_{cfg.fast_period}_{cfg.slow_period}_{cfg.signal_period}"
        col_hist = f"MACDh_{cfg.fast_period}_{cfg.slow_period}_{cfg.signal_period}"

        df["macd"] = macd_df[col_macd]
        df["macd_signal"] = macd_df[col_signal]
        df["macd_hist"] = macd_df[col_hist]
        logger.debug(
            "Added MACD(%d, %d, %d).",
            cfg.fast_period,
            cfg.slow_period,
            cfg.signal_period,
        )
        return df

    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Bands: upper, middle, lower, bandwidth, %%B."""
        cfg = self.config.indicators.bollinger_bands
        bb = ta.bbands(df["close"], length=cfg.period, std=cfg.std_dev)

        # pandas_ta column names vary by version; match by prefix.
        bb_cols = bb.columns.tolist()
        col_map = {
            "bb_lower": next(c for c in bb_cols if c.startswith("BBL_")),
            "bb_middle": next(c for c in bb_cols if c.startswith("BBM_")),
            "bb_upper": next(c for c in bb_cols if c.startswith("BBU_")),
            "bb_bandwidth": next(c for c in bb_cols if c.startswith("BBB_")),
            "bb_pct": next(c for c in bb_cols if c.startswith("BBP_")),
        }
        for target_name, source_col in col_map.items():
            df[target_name] = bb[source_col]

        logger.debug(
            "Added Bollinger Bands(%d, %.1f).",
            cfg.period,
            cfg.std_dev,
        )
        return df

    def _add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exponential Moving Averages at multiple periods."""
        for period in self.config.indicators.ema.periods:
            col_name = f"ema_{period}"
            df[col_name] = ta.ema(df["close"], length=period)
            logger.debug("Added EMA(%d).", period)
        return df

    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Average True Range."""
        period = self.config.indicators.atr.period
        df["atr"] = ta.atr(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            length=period,
        )
        logger.debug("Added ATR(%d).", period)
        return df

    def _add_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Logarithmic returns: log(close / close_prev)."""
        import numpy as np

        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        logger.debug("Added log returns.")
        return df

    def _add_price_change_pct(self, df: pd.DataFrame) -> pd.DataFrame:
        """Percentage change in close price."""
        df["pct_change"] = df["close"].pct_change()
        logger.debug("Added price change percentage.")
        return df

    def _add_volume_change_pct(self, df: pd.DataFrame) -> pd.DataFrame:
        """Percentage change in volume."""
        df["volume_change"] = df["volume"].pct_change()
        logger.debug("Added volume change percentage.")
        return df

    def _add_high_low_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intra-candle high-low range."""
        df["high_low_range"] = df["high"] - df["low"]
        logger.debug("Added high-low range.")
        return df

    def _add_open_close_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intra-candle open-close difference."""
        df["close_open_diff"] = df["close"] - df["open"]
        logger.debug("Added open-close range.")
        return df

    def _add_direction_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Binary classification label: 1 if next close > current close, else 0.

        The label is shifted so that row *i* tells whether the *next*
        candle's close is higher.  The last row will be NaN (no future
        data).
        """
        df["direction"] = (df["close"].shift(-1) > df["close"]).astype(float)
        # The very last row has no future -> set to NaN so it can be
        # dropped later together with other NaN rows.
        df.loc[df.index[-1], "direction"] = float("nan")
        logger.debug("Added direction label.")
        return df
