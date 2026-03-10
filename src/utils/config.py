"""
Pydantic-based YAML configuration loader.

Usage::

    from src.utils.config import load_config

    cfg = load_config()                    # reads from configs/
    print(cfg.data.symbols)
    print(cfg.model.common.batch_size)
    print(cfg.model.lstm.hidden_size)

Individual configs can also be loaded directly::

    from src.utils.config import DataConfig, FeatureConfig
    data_cfg = DataConfig.load()
    feat_cfg = FeatureConfig.load()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

from src.utils.constants import CONFIGS_DIR

# =========================================================================
# Helper
# =========================================================================

def _load_yaml(path: Path) -> dict[str, Any]:
    """Safely load a YAML file; return empty dict if file is missing."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        content = yaml.safe_load(fh)
    return content if isinstance(content, dict) else {}


# =========================================================================
# Sub-configs
# =========================================================================

# ---------------------------------------------------------------------------
# Data  (mirrors configs/data_config.yaml)
# ---------------------------------------------------------------------------

class RateLimitConfig(BaseModel):
    requests_per_second: int = 10
    retry_on_failure: bool = True
    max_retries: int = 5
    retry_delay_seconds: float = 2.0


class ExchangeConfig(BaseModel):
    name: str = "binance"
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)


class StorageConfig(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    sequences_dir: str = "data/sequences"
    scalers_dir: str = "data/scalers"


class SplitConfig(BaseModel):
    train: float = 0.7
    val: float = 0.15
    test: float = 0.15

    @field_validator("test")
    @classmethod
    def _ratios_sum_to_one(cls, v: float, info: Any) -> float:
        train = info.data.get("train", 0.7)
        val = info.data.get("val", 0.15)
        if abs(train + val + v - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {train + val + v:.4f}")
        return v


class DateRangeConfig(BaseModel):
    start: str = "2021-01-01"
    end: str = "2024-12-31"


class DataConfig(BaseModel):
    """Mirrors ``configs/data_config.yaml``."""

    symbols: list[str] = Field(default=["BTC/USDT", "ETH/USDT"])
    timeframes: list[str] = Field(default=["1h", "4h"])
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    date_range: DateRangeConfig = Field(default_factory=DateRangeConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)

    @classmethod
    def load(cls, path: Path | str | None = None) -> DataConfig:
        p = Path(path) if path else CONFIGS_DIR / "data_config.yaml"
        raw = _load_yaml(p)
        return cls(**raw) if raw else cls()


# ---------------------------------------------------------------------------
# Features  (mirrors configs/feature_config.yaml)
# ---------------------------------------------------------------------------

class RSIConfig(BaseModel):
    period: int = 14


class MACDConfig(BaseModel):
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9


class BollingerBandsConfig(BaseModel):
    period: int = 20
    std_dev: float = 2.0


class EMAConfig(BaseModel):
    periods: list[int] = Field(default=[7, 25, 99])


class ATRConfig(BaseModel):
    period: int = 14


class IndicatorsConfig(BaseModel):
    rsi: RSIConfig = Field(default_factory=RSIConfig)
    macd: MACDConfig = Field(default_factory=MACDConfig)
    bollinger_bands: BollingerBandsConfig = Field(default_factory=BollingerBandsConfig)
    ema: EMAConfig = Field(default_factory=EMAConfig)
    atr: ATRConfig = Field(default_factory=ATRConfig)


class FeatureConfig(BaseModel):
    """Mirrors ``configs/feature_config.yaml``."""

    target_column: str = "close"
    indicators: IndicatorsConfig = Field(default_factory=IndicatorsConfig)
    scaler_type: str = Field(default="minmax", description="'minmax', 'standard', or 'robust'")
    lookback_windows: list[int] = Field(default=[24, 48, 96])
    forecast_horizons: list[int] = Field(default=[1, 4, 12, 24])

    @classmethod
    def load(cls, path: Path | str | None = None) -> FeatureConfig:
        p = Path(path) if path else CONFIGS_DIR / "feature_config.yaml"
        raw = _load_yaml(p)
        return cls(**raw) if raw else cls()


# ---------------------------------------------------------------------------
# Model  (mirrors configs/model_config.yaml)
# ---------------------------------------------------------------------------

class CommonTrainingConfig(BaseModel):
    """Common training hyper-parameters shared across all models."""
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10
    learning_rate: float = 1e-3
    seed: int = 42
    num_workers: int = 4
    device: str = "auto"


class LSTMConfig(BaseModel):
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False


class GRUConfig(BaseModel):
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2


class TransformerConfig(BaseModel):
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1


class TFTConfig(BaseModel):
    hidden_size: int = 128
    num_attention_heads: int = 4
    dropout: float = 0.1
    quantiles: list[float] = Field(default=[0.1, 0.5, 0.9])


class ARIMAConfig(BaseModel):
    max_p: int = 5
    max_d: int = 2
    max_q: int = 5
    seasonal: bool = False


class XGBoostConfig(BaseModel):
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 20


class AnomalyConfig(BaseModel):
    hidden_size: int = 64
    latent_dim: int = 32
    num_layers: int = 2
    threshold_sigma: float = 3.0


class DualHeadConfig(BaseModel):
    alpha: float = 0.7  # regression loss weight; (1 - alpha) for classification


class ModelConfig(BaseModel):
    """Mirrors ``configs/model_config.yaml``."""

    common: CommonTrainingConfig = Field(default_factory=CommonTrainingConfig)
    lstm: LSTMConfig = Field(default_factory=LSTMConfig)
    gru: GRUConfig = Field(default_factory=GRUConfig)
    transformer: TransformerConfig = Field(default_factory=TransformerConfig)
    tft: TFTConfig = Field(default_factory=TFTConfig)
    arima: ARIMAConfig = Field(default_factory=ARIMAConfig)
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)
    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)
    dual_head: DualHeadConfig = Field(default_factory=DualHeadConfig)


# ---------------------------------------------------------------------------
# Dashboard  (mirrors configs/dashboard_config.yaml)
# ---------------------------------------------------------------------------

class DashboardDefaultsConfig(BaseModel):
    symbol: str = "BTC/USDT"
    model: str = "lstm"
    timeframe: str = "1h"
    lookback_window: int = 48
    forecast_horizon: int = 4


class DashboardLayoutConfig(BaseModel):
    page_title: str = "Crypto Prediction Dashboard"
    sidebar_width: int = 300
    chart_height: int = 600
    sections: list[dict[str, Any]] = Field(default_factory=list)


class DashboardConfig(BaseModel):
    """Mirrors ``configs/dashboard_config.yaml``."""

    refresh_interval_seconds: int = 60
    theme: str = "dark"
    defaults: DashboardDefaultsConfig = Field(default_factory=DashboardDefaultsConfig)
    layout: DashboardLayoutConfig = Field(default_factory=DashboardLayoutConfig)
    available_models: list[str] = Field(
        default=["lstm", "gru", "transformer", "tft", "arima", "xgboost"]
    )
    available_symbols: list[str] = Field(default=["BTC/USDT", "ETH/USDT"])


# =========================================================================
# Top-level application config
# =========================================================================

class AppConfig(BaseModel):
    """Root configuration object aggregating every sub-config."""

    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)


# =========================================================================
# Loader
# =========================================================================

_YAML_FILENAMES: dict[str, str] = {
    "data": "data_config.yaml",
    "features": "feature_config.yaml",
    "model": "model_config.yaml",
    "dashboard": "dashboard_config.yaml",
}


def load_config(config_dir: Optional[Path] = None) -> AppConfig:
    """Build an :class:`AppConfig` by merging YAML files from *config_dir*.

    Parameters
    ----------
    config_dir:
        Directory containing the YAML files.  Defaults to the project-level
        ``configs/`` directory.

    Returns
    -------
    AppConfig
        Validated configuration object.
    """
    config_dir = Path(config_dir) if config_dir is not None else CONFIGS_DIR

    merged: dict[str, Any] = {}
    for section, filename in _YAML_FILENAMES.items():
        yaml_path = config_dir / filename
        section_data = _load_yaml(yaml_path)
        if section_data:
            merged[section] = section_data

    return AppConfig(**merged)
