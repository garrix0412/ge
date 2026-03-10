# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Crypto price prediction & anomaly detection system for course GCAP3123. Multi-model (ARIMA, XGBoost, LSTM, GRU, Transformer, TFT) cryptocurrency forecasting with an LSTM-Autoencoder for anomaly detection and a Streamlit dashboard.

## Environment

- **Conda env**: `ge` (Python 3.12, macOS with MPS support)
- Activate: `conda activate ge`
- Install deps: `pip install -e ".[dev]"`

## Key Commands

```bash
# One-click full pipeline (download → features → train → anomaly → comparison)
python scripts/run_pipeline.py --quick          # Quick demo (~4 experiments)
python scripts/run_pipeline.py                  # Full grid (all models × symbols × horizons)
python scripts/run_pipeline.py --skip-download  # Skip data download step

# Individual steps
python scripts/download_data.py --symbol BTC/USDT --timeframe 1h
python scripts/prepare_features.py
python scripts/run_training.py --model lstm --symbol BTC/USDT --timeframe 1h --lookback 24 --horizon 1
python scripts/run_anomaly_detection.py

# Dashboard
streamlit run src/dashboard/app.py

# Tests
pytest tests/ -v
```

## Architecture

### Data Flow
```
Binance API (ccxt) or synthetic GBM fallback
  → data/raw/*.csv (OHLCV)
  → feature_engineer.py (RSI, MACD, BB, EMA, ATR, etc.)
  → data/processed/*_processed.parquet (23 features)
  → preprocessor.py (MinMaxScaler on train only, sliding windows)
  → dataset.py (PyTorch DataLoader)
  → models → results/metrics/*.json + results/checkpoints/
```

### Config System
All parameters live in `configs/*.yaml`, parsed by Pydantic models in `src/utils/config.py`. Call `load_config()` to get an `AppConfig` with `.data`, `.features`, `.model`, `.dashboard` sections. YAML filenames map: `data_config.yaml`, `feature_config.yaml`, `model_config.yaml`, `dashboard_config.yaml`.

### Model Interface
All models implement `BaseModel` (ABC in `src/models/base_model.py`) with `fit()`, `predict()`, `save()`, `load()`. PyTorch models extend `BaseTorchModel` which adds device management, state_dict persistence, and MC-Dropout uncertainty estimation.

**Dual-head DL models**: LSTM/GRU/Transformer/TFT forward returns `(reg_output, cls_output)`. Combined loss: `alpha * MSE + (1-alpha) * BCE`. The `alpha` weight is configured in `model_config.yaml` under `dual_head.alpha`.

### Two Training Paths in experiment.py
- **PyTorch models** (LSTM, GRU, Transformer, TFT): uses `Trainer` class with DataLoaders, evaluated via `_evaluate_test_torch()`
- **Non-PyTorch models** (ARIMA, XGBoost): calls `model.fit()` directly with numpy arrays, evaluated via `_evaluate_test_sklearn()`

The check is `isinstance(model, BaseTorchModel)`.

### Model Registry
`src/models/registry.py` uses lazy imports. `get_model(name, **kwargs)` instantiates by name. Available: arima, xgboost, lstm, gru, transformer, tft, anomaly.

### Anomaly Detection
`AnomalyAutoencoder` uses its own `fit()` and `detect_anomalies()` — does NOT go through `Trainer` (which expects dual-head output). Threshold = mean + 3*std of reconstruction errors.

## Important Conventions

- All directory paths are defined in `src/utils/constants.py` via pathlib
- Processed data files: `{symbol}_{timeframe}_processed.parquet` (symbol uses `_` not `/`)
- Scaler is fitted on training data ONLY (no future leakage)
- Sequence shapes: X = `(n, lookback, features)`, y_reg = `(n, horizon)`, y_cls = `(n, horizon)`
- Bollinger Bands columns are matched by prefix (`startswith("BBL_")`) due to pandas_ta version differences
- When Binance API is unreachable, `run_pipeline.py` falls back to synthetic data via Geometric Brownian Motion

## Known Quirks

- LSTM/GRU models have their own built-in training loops in `fit()` AND can also be trained via the external `Trainer` class (experiment.py uses `Trainer`)
- XGBoost's `fit()` accepts explicit `y_train_cls` via kwargs; if not provided, derives labels from regression targets
- XGBoost skips classifier training if only 1 class is present in labels
