"""
Project-wide constants, default paths, and naming conventions.
"""

from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: Final[int] = 42

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
OHLCV_COLUMNS: Final[list[str]] = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

TARGET_COL: Final[str] = "close"
DIRECTION_COL: Final[str] = "direction"  # 1 = up, 0 = down

# Technical-indicator feature columns produced by the feature-engineering step.
FEATURE_COLUMNS: Final[list[str]] = [
    # --- raw price & volume ---
    "open",
    "high",
    "low",
    "close",
    "volume",
    # --- returns ---
    "log_return",
    "pct_change",
    # --- RSI ---
    "rsi",
    # --- MACD ---
    "macd",
    "macd_signal",
    "macd_hist",
    # --- Bollinger Bands ---
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "bb_bandwidth",
    "bb_pct",
    # --- Exponential Moving Averages ---
    "ema_7",
    "ema_25",
    "ema_99",
    # --- ATR ---
    "atr",
    # --- derived ---
    "high_low_range",
    "close_open_diff",
    "volume_change",
]

# ---------------------------------------------------------------------------
# Directory layout (all paths relative to PROJECT_ROOT)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
SEQUENCES_DIR: Final[Path] = DATA_DIR / "sequences"
SCALERS_DIR: Final[Path] = DATA_DIR / "scalers"

RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"
CHECKPOINTS_DIR: Final[Path] = RESULTS_DIR / "checkpoints"
METRICS_DIR: Final[Path] = RESULTS_DIR / "metrics"
FIGURES_DIR: Final[Path] = RESULTS_DIR / "figures"

CONFIGS_DIR: Final[Path] = PROJECT_ROOT / "configs"
LOGS_DIR: Final[Path] = PROJECT_ROOT / "logs"

# ---------------------------------------------------------------------------
# Default file-naming patterns
# ---------------------------------------------------------------------------
RAW_DATA_PATTERN: Final[str] = "{symbol}_{timeframe}_raw.parquet"
PROCESSED_DATA_PATTERN: Final[str] = "{symbol}_{timeframe}_processed.parquet"
MODEL_FILE_PATTERN: Final[str] = "{model_name}_{symbol}_{timeframe}_lb{lookback}_h{horizon}"
SCALER_FILE_PATTERN: Final[str] = "{symbol}_{timeframe}_scaler.joblib"
METRICS_FILE_PATTERN: Final[str] = "{model_name}_{symbol}_{timeframe}_lb{lookback}_h{horizon}_metrics.json"
HISTORY_FILE_PATTERN: Final[str] = "{model_name}_{symbol}_{timeframe}_lb{lookback}_h{horizon}_history.json"
ANOMALY_RESULTS_PATTERN: Final[str] = "anomaly_{symbol}_{timeframe}_lb{lookback}_h{horizon}_results.json"
