"""One-click pipeline: download -> features -> train -> anomaly -> results.

Orchestrates every stage of the project in a single invocation.  Each
step is wrapped in error handling so that partial failures do not prevent
later stages from running.

Usage::

    python scripts/run_pipeline.py                     # Full run
    python scripts/run_pipeline.py --quick              # Quick demo (fewer combos)
    python scripts/run_pipeline.py --skip-download      # Skip data download
    python scripts/run_pipeline.py --quick --skip-download
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path for ``src`` imports.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.utils.config import AppConfig, load_config
from src.utils.constants import METRICS_DIR, PROCESSED_DIR
from src.utils.io import load_metrics, save_metrics
from src.utils.logger import get_logger

logger = get_logger("run_pipeline")

# =========================================================================
# CLI
# =========================================================================


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="One-click end-to-end pipeline for crypto prediction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a minimal demo set (~6 experiments) instead of the full grid.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the data-download step (assumes raw CSVs already exist).",
    )
    return parser.parse_args()


# =========================================================================
# Pipeline steps
# =========================================================================


def _generate_synthetic_ohlcv(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
) -> "pd.DataFrame":
    """Generate realistic synthetic OHLCV data when the API is unreachable."""
    import numpy as np

    # Determine number of candles
    freq_map = {"1h": "h", "4h": "4h", "1d": "D"}
    freq = freq_map.get(timeframe, "h")
    dates = pd.date_range(start, end, freq=freq)
    n = len(dates)

    np.random.seed(hash(symbol) % 2**31)

    # Starting prices per symbol
    base = {"BTC/USDT": 30000.0, "ETH/USDT": 2000.0}.get(symbol, 1000.0)
    vol = {"BTC/USDT": 100.0, "ETH/USDT": 10.0}.get(symbol, 5.0)

    # Geometric Brownian Motion for realistic price series
    returns = np.random.normal(0.00001, 0.005, n)
    close = base * np.exp(np.cumsum(returns))
    spread = close * np.abs(np.random.normal(0.003, 0.002, n))

    df = pd.DataFrame({
        "timestamp": dates[:n],
        "open": close + np.random.randn(n) * vol * 0.3,
        "high": close + spread,
        "low": close - spread,
        "close": close,
        "volume": np.abs(np.random.normal(500, 200, n)) * (base / 1000),
    })
    return df


def _step_download(config: AppConfig) -> None:
    """Step 1 -- Download raw OHLCV data, with synthetic fallback."""
    from src.data.fetcher import DataFetcher
    from src.utils.constants import RAW_DIR

    symbols: list[str] = config.data.symbols
    timeframes: list[str] = config.data.timeframes
    start: str = config.data.date_range.start
    end: str = config.data.date_range.end

    # Try real download first
    real_ok = 0
    try:
        fetcher = DataFetcher(exchange_name=config.data.exchange.name)
        for symbol in symbols:
            for timeframe in timeframes:
                logger.info("Downloading %s %s (%s -> %s) ...", symbol, timeframe, start, end)
                try:
                    df = fetcher.fetch_and_save(symbol, timeframe, start, end)
                    if df is not None and not df.empty:
                        real_ok += 1
                except Exception:
                    logger.warning("Download failed for %s %s.", symbol, timeframe)
    except Exception:
        logger.warning("Could not initialise exchange — will use synthetic data.")

    total = len(symbols) * len(timeframes)

    # Fallback to synthetic data for any missing pairs
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    synth_count = 0
    for symbol in symbols:
        for timeframe in timeframes:
            safe_symbol = symbol.replace("/", "_")
            raw_path = RAW_DIR / f"{safe_symbol}_{timeframe}.csv"
            if not raw_path.exists():
                logger.info(
                    "Generating synthetic data for %s %s (API unreachable).",
                    symbol, timeframe,
                )
                df = _generate_synthetic_ohlcv(symbol, timeframe, start, end)
                df.to_csv(raw_path, index=False)
                synth_count += 1
                logger.info("  Saved %d rows -> %s", len(df), raw_path)

    logger.info(
        "Download step: %d real + %d synthetic / %d total.",
        real_ok, synth_count, total,
    )


def _step_features(config: AppConfig) -> None:
    """Step 2 -- Feature-engineer raw CSVs into processed parquets."""
    from src.data.feature_engineer import FeatureEngineer
    from src.utils.constants import RAW_DIR
    from src.utils.io import save_dataframe

    fe = FeatureEngineer(config=config.features)

    symbols: list[str] = config.data.symbols
    timeframes: list[str] = config.data.timeframes

    total = len(symbols) * len(timeframes)
    ok = 0

    for symbol in symbols:
        for timeframe in timeframes:
            safe_symbol = symbol.replace("/", "_")
            raw_path = RAW_DIR / f"{safe_symbol}_{timeframe}.csv"
            out_path = PROCESSED_DIR / f"{safe_symbol}_{timeframe}_processed.parquet"

            if not raw_path.exists():
                logger.warning("Raw file missing, skipping: %s", raw_path)
                continue

            try:
                df = pd.read_csv(raw_path)
                df = fe.add_all_features(df)
                df = fe.drop_na(df)

                if df.empty:
                    logger.warning("Empty after features: %s %s", symbol, timeframe)
                    continue

                PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
                save_dataframe(df, out_path)
                ok += 1
            except Exception:
                logger.exception("Feature engineering failed for %s %s.", symbol, timeframe)

    logger.info("Feature step: %d / %d succeeded.", ok, total)


def _step_train(
    config: AppConfig,
    combinations: list[tuple[str, str, str, int, int]],
) -> list[dict[str, Any]]:
    """Step 3 -- Train prediction models for each combination.

    Parameters
    ----------
    config:
        Application configuration.
    combinations:
        List of ``(model_name, symbol, timeframe, lookback, horizon)`` tuples.

    Returns
    -------
    list[dict]
        One result dict per combination with at least keys
        ``experiment_id``, ``model``, ``status``, and metric values on
        success.
    """
    from src.training.experiment import Experiment

    all_results: list[dict[str, Any]] = []

    for i, (model_name, symbol, timeframe, lookback, horizon) in enumerate(
        combinations, 1
    ):
        safe_symbol = symbol.replace("/", "_")
        exp_id = f"{model_name}_{safe_symbol}_{timeframe}_lb{lookback}_h{horizon}"
        metrics_path = METRICS_DIR / f"{exp_id}_metrics.json"

        logger.info(
            "[%d/%d] %s ...",
            i,
            len(combinations),
            exp_id,
        )

        # Skip if already computed
        if metrics_path.exists():
            logger.info("  Cached -- skipping.")
            try:
                existing = load_metrics(metrics_path)
                row: dict[str, Any] = {
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
                logger.warning("  Could not load cached metrics for %s.", exp_id)
            continue

        try:
            exp = Experiment(
                model_name=model_name,
                symbol=symbol,
                timeframe=timeframe,
                lookback=lookback,
                horizon=horizon,
                config=config,
            )
            result = exp.run()
            row = {
                "experiment_id": exp_id,
                "model": model_name,
                "symbol": symbol,
                "timeframe": timeframe,
                "lookback": lookback,
                "horizon": horizon,
                "status": "success",
                "elapsed_seconds": result.get("elapsed_seconds"),
                **result.get("metrics", {}),
            }
            all_results.append(row)
            logger.info("  Success (%.1fs).", result.get("elapsed_seconds", 0))
        except Exception:
            logger.exception("  FAILED: %s", exp_id)
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

    return all_results


def _step_anomaly(config: AppConfig) -> list[dict[str, Any]]:
    """Step 4 -- Run anomaly detection for each symbol/timeframe."""
    from src.data.preprocessor import DataPreprocessor
    from src.models.registry import get_model
    from src.utils.constants import (
        CHECKPOINTS_DIR,
        DIRECTION_COL,
        FEATURE_COLUMNS,
        TARGET_COL,
    )
    from src.utils.io import ensure_dir, load_dataframe

    symbols: list[str] = config.data.symbols
    timeframes: list[str] = config.data.timeframes
    lookback: int = config.features.lookback_windows[0]  # use first lookback
    horizon: int = 1

    anomaly_results: list[dict[str, Any]] = []

    for symbol in symbols:
        for timeframe in timeframes:
            safe_symbol = symbol.replace("/", "_")
            exp_id = f"anomaly_{safe_symbol}_{timeframe}_lb{lookback}_h{horizon}"
            data_path = PROCESSED_DIR / f"{safe_symbol}_{timeframe}_processed.parquet"

            logger.info("Anomaly detection: %s ...", exp_id)

            if not data_path.exists():
                logger.warning("Processed data missing: %s -- skipping.", data_path)
                continue

            try:
                df = load_dataframe(data_path)

                preprocessor = DataPreprocessor(
                    data_config=config.data,
                    feature_config=config.features,
                )
                feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
                if not feature_cols:
                    logger.warning("No feature columns for %s %s.", symbol, timeframe)
                    continue

                split_cfg = config.data.split
                sequences = preprocessor.prepare_all(
                    df=df,
                    feature_columns=feature_cols,
                    target_col=TARGET_COL,
                    label_col=DIRECTION_COL,
                    lookback=lookback,
                    horizon=horizon,
                    train_ratio=split_cfg.train,
                    val_ratio=split_cfg.val,
                    test_ratio=split_cfg.test,
                )

                n_features = sequences["X_train"].shape[2]
                model = get_model(
                    "anomaly",
                    input_size=n_features,
                    lookback=lookback,
                    horizon=horizon,
                    config=config,
                )
                model.auto_device()

                model.fit(
                    X_train=sequences["X_train"],
                    y_train=None,
                    X_val=sequences["X_val"],
                    y_val=None,
                    max_epochs=config.model.common.max_epochs,
                )

                result = model.detect_anomalies(sequences["X_test"])

                errors = result["reconstruction_errors"]
                anomaly_flags = result["anomaly_flags"]
                n_anomalies = int(anomaly_flags.sum())
                anomaly_ratio = (
                    n_anomalies / len(errors) if len(errors) > 0 else 0.0
                )

                # Save checkpoint
                ckpt_dir = CHECKPOINTS_DIR / exp_id
                ensure_dir(ckpt_dir)
                model.save(ckpt_dir / "anomaly_model.pt")

                # Save results
                res: dict[str, Any] = {
                    "experiment_id": exp_id,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "n_test_samples": len(errors),
                    "n_anomalies": n_anomalies,
                    "anomaly_ratio": anomaly_ratio,
                    "threshold": result["threshold"],
                    "mean_error": float(errors.mean()),
                }
                save_metrics(res, METRICS_DIR / f"{exp_id}_results.json")
                anomaly_results.append(res)

                logger.info(
                    "  Anomalies: %d / %d (%.2f%%)",
                    n_anomalies,
                    len(errors),
                    anomaly_ratio * 100,
                )
            except Exception:
                logger.exception("  Anomaly detection FAILED: %s", exp_id)

    return anomaly_results


def _step_comparison_table(
    all_results: list[dict[str, Any]],
) -> pd.DataFrame | None:
    """Step 5 -- Build and save a comparison CSV from training results."""
    if not all_results:
        logger.warning("No training results to build comparison table.")
        return None

    df = pd.DataFrame(all_results)
    out_path = METRICS_DIR / "comparison_table.csv"
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Comparison table saved -> %s (%d rows).", out_path, len(df))
    return df


# =========================================================================
# Summary printer
# =========================================================================


def _print_summary(
    results_df: pd.DataFrame | None,
    anomaly_results: list[dict[str, Any]],
    timings: dict[str, float],
) -> None:
    """Print a human-friendly summary table to stdout."""
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)

    # Timings
    print("\nStep timings:")
    for step, elapsed in timings.items():
        print(f"  {step:<25s} : {elapsed:>8.1f}s")
    total_elapsed = sum(timings.values())
    print(f"  {'TOTAL':<25s} : {total_elapsed:>8.1f}s")

    # Training results summary
    if results_df is not None and not results_df.empty:
        success_df = results_df[results_df["status"].isin(["success", "cached"])]
        failed_count = len(results_df[results_df["status"] == "failed"])

        print(f"\nTraining experiments: {len(results_df)} total, "
              f"{len(success_df)} succeeded/cached, {failed_count} failed")

        # Best model per metric (only for numeric metric columns)
        metric_candidates = [
            "reg_mae", "reg_rmse", "reg_mape", "reg_directional_accuracy",
            "cls_accuracy", "cls_f1_score", "cls_auc_roc",
        ]
        lower_is_better = {"reg_mae", "reg_rmse", "reg_mape"}

        if not success_df.empty:
            print("\nBest model per metric:")
            for metric in metric_candidates:
                if metric not in success_df.columns:
                    continue
                col = pd.to_numeric(success_df[metric], errors="coerce")
                valid = col.dropna()
                if valid.empty:
                    continue

                if metric in lower_is_better:
                    best_idx = valid.idxmin()
                else:
                    best_idx = valid.idxmax()

                best_row = success_df.loc[best_idx]
                print(
                    f"  {metric:<12s} : {float(valid.loc[best_idx]):.6f}  "
                    f"({best_row.get('experiment_id', 'N/A')})"
                )

    # Anomaly summary
    if anomaly_results:
        print(f"\nAnomaly detection: {len(anomaly_results)} run(s)")
        for ar in anomaly_results:
            print(
                f"  {ar['experiment_id']}: "
                f"{ar['n_anomalies']}/{ar['n_test_samples']} anomalies "
                f"({ar['anomaly_ratio'] * 100:.2f}%)"
            )

    print("\n" + "=" * 70)


# =========================================================================
# Combination builders
# =========================================================================


def _build_quick_combinations() -> list[tuple[str, str, str, int, int]]:
    """Return a minimal set of experiments for a quick demo."""
    models = ["lstm", "xgboost"]
    symbols = ["BTC/USDT"]
    timeframes = ["1h"]
    lookbacks = [24]
    horizons = [1, 4]

    return list(itertools.product(models, symbols, timeframes, lookbacks, horizons))


def _build_full_combinations(
    config: AppConfig,
) -> list[tuple[str, str, str, int, int]]:
    """Return the full experiment grid from configuration.

    Uses all models (excluding ``anomaly``), all symbols, all timeframes,
    all lookback windows, and the first two forecast horizons.
    """
    from src.models.registry import list_models

    all_models = [m for m in list_models() if m != "anomaly"]
    symbols: list[str] = config.data.symbols
    timeframes: list[str] = config.data.timeframes
    lookbacks: list[int] = config.features.lookback_windows
    horizons: list[int] = config.features.forecast_horizons[:2]

    return list(itertools.product(all_models, symbols, timeframes, lookbacks, horizons))


# =========================================================================
# Main
# =========================================================================


def main() -> None:
    """Run the end-to-end pipeline."""
    args = _parse_args()
    config = load_config()

    timings: dict[str, float] = {}
    all_train_results: list[dict[str, Any]] = []
    anomaly_results: list[dict[str, Any]] = []

    mode = "QUICK" if args.quick else "FULL"
    logger.info("=" * 70)
    logger.info("End-to-end pipeline (%s mode)", mode)
    logger.info("=" * 70)

    pipeline_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Download data
    # ------------------------------------------------------------------
    if not args.skip_download:
        logger.info("-" * 70)
        logger.info("STEP 1 / 5 : Download data")
        logger.info("-" * 70)
        t0 = time.time()
        try:
            _step_download(config)
        except Exception:
            logger.exception("Download step encountered a fatal error.")
        timings["1_download"] = time.time() - t0
    else:
        logger.info("STEP 1 / 5 : Download data -- SKIPPED (--skip-download)")
        timings["1_download"] = 0.0

    # ------------------------------------------------------------------
    # Step 2: Feature engineering
    # ------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 2 / 5 : Feature engineering")
    logger.info("-" * 70)
    t0 = time.time()
    try:
        _step_features(config)
    except Exception:
        logger.exception("Feature engineering step encountered a fatal error.")
    timings["2_features"] = time.time() - t0

    # ------------------------------------------------------------------
    # Step 3: Train models
    # ------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 3 / 5 : Train models")
    logger.info("-" * 70)
    t0 = time.time()

    if args.quick:
        combinations = _build_quick_combinations()
    else:
        combinations = _build_full_combinations(config)

    logger.info("  %d experiment(s) to run.", len(combinations))
    try:
        all_train_results = _step_train(config, combinations)
    except Exception:
        logger.exception("Training step encountered a fatal error.")
    timings["3_train"] = time.time() - t0

    # ------------------------------------------------------------------
    # Step 4: Anomaly detection
    # ------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 4 / 5 : Anomaly detection")
    logger.info("-" * 70)
    t0 = time.time()
    try:
        anomaly_results = _step_anomaly(config)
    except Exception:
        logger.exception("Anomaly detection step encountered a fatal error.")
    timings["4_anomaly"] = time.time() - t0

    # ------------------------------------------------------------------
    # Step 5: Comparison table
    # ------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 5 / 5 : Comparison table & summary")
    logger.info("-" * 70)
    t0 = time.time()
    results_df: pd.DataFrame | None = None
    try:
        results_df = _step_comparison_table(all_train_results)
    except Exception:
        logger.exception("Comparison table step encountered a fatal error.")
    timings["5_comparison"] = time.time() - t0

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    _print_summary(results_df, anomaly_results, timings)

    total_elapsed = time.time() - pipeline_start
    logger.info("Pipeline finished in %.1f seconds.", total_elapsed)


if __name__ == "__main__":
    main()
