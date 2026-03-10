"""
Data preprocessing: temporal train/val/test splitting, scaling, and
sequence construction for time-series models.

Usage::

    from src.data.preprocessor import DataPreprocessor

    prep = DataPreprocessor()
    train_df, val_df, test_df = prep.split_data(df)
    prep.fit_scaler(train_df, feature_cols)
    X_train = prep.transform(train_df, feature_cols)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.utils.config import DataConfig, FeatureConfig
from src.utils.constants import SCALERS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Map config string to sklearn scaler class.
_SCALER_MAP: dict[str, type] = {
    "minmax": MinMaxScaler,
    "standard": StandardScaler,
    "robust": RobustScaler,
}


class DataPreprocessor:
    """Scale features, split data temporally, and build model-ready sequences.

    Parameters
    ----------
    data_config:
        Data pipeline configuration (split ratios, paths, etc.).
    feature_config:
        Feature engineering configuration (scaler type, lookback, etc.).
    """

    def __init__(
        self,
        data_config: Optional[DataConfig] = None,
        feature_config: Optional[FeatureConfig] = None,
    ) -> None:
        self.data_config = data_config or DataConfig.load()
        self.feature_config = feature_config or FeatureConfig.load()

        scaler_name = self.feature_config.scaler_type.lower()
        scaler_cls = _SCALER_MAP.get(scaler_name)
        if scaler_cls is None:
            raise ValueError(
                f"Unknown scaler type '{scaler_name}'. "
                f"Choose from: {list(_SCALER_MAP.keys())}"
            )

        self.scaler: MinMaxScaler | StandardScaler | RobustScaler = scaler_cls()
        self._is_fitted: bool = False
        # Index of the close-price column inside the feature array.
        # Set automatically by `prepare_all`; callers of `create_sequences`
        # can override via the `close_col_idx` parameter.
        self._close_col_idx: int = 3  # sensible default for OHLCV ordering

        logger.info(
            "DataPreprocessor initialised – scaler=%s.",
            scaler_name,
        )

    # ------------------------------------------------------------------
    # Train / val / test split
    # ------------------------------------------------------------------

    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split *df* into train / validation / test sets in temporal order.

        Parameters
        ----------
        df:
            The full feature DataFrame, sorted chronologically.
        train_ratio, val_ratio, test_ratio:
            Fractions of the dataset.  Must sum to 1.0.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            ``(train_df, val_df, test_df)`` -- no shuffling is applied.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio:.4f}"
        )

        n = len(df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_df = df.iloc[:train_end].reset_index(drop=True)
        val_df = df.iloc[train_end:val_end].reset_index(drop=True)
        test_df = df.iloc[val_end:].reset_index(drop=True)

        logger.info(
            "Split data: train=%d, val=%d, test=%d (total=%d).",
            len(train_df),
            len(val_df),
            len(test_df),
            n,
        )
        return train_df, val_df, test_df

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    def fit_scaler(
        self,
        train_df: pd.DataFrame,
        feature_columns: list[str],
    ) -> None:
        """Fit the scaler on the **training set only** to prevent data leakage.

        Parameters
        ----------
        train_df:
            Training-set DataFrame.
        feature_columns:
            Column names to scale.
        """
        self.scaler.fit(train_df[feature_columns].values)
        self._is_fitted = True
        logger.info(
            "Scaler fitted on training data (%d rows, %d features).",
            len(train_df),
            len(feature_columns),
        )

    def transform(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
    ) -> np.ndarray:
        """Transform *df* using the previously fitted scaler.

        Parameters
        ----------
        df:
            Any split (train / val / test) DataFrame.
        feature_columns:
            Column names to transform (must match those used in :meth:`fit_scaler`).

        Returns
        -------
        np.ndarray
            Scaled feature array of shape ``(n_samples, n_features)``.
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler has not been fitted yet. Call fit_scaler() first.")
        scaled = self.scaler.transform(df[feature_columns].values)
        logger.debug("Transformed %d rows.", len(df))
        return scaled

    def inverse_transform(
        self,
        data: np.ndarray,
        feature_columns: list[str],
    ) -> np.ndarray:
        """Reverse the scaling transformation to recover original values.

        Parameters
        ----------
        data:
            Scaled array whose columns correspond to *feature_columns*.
        feature_columns:
            Column names (used only for logging / validation).

        Returns
        -------
        np.ndarray
            Data in the original scale.
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler has not been fitted yet. Call fit_scaler() first.")
        original = self.scaler.inverse_transform(data)
        logger.debug(
            "Inverse-transformed array of shape %s.",
            original.shape,
        )
        return original

    # ------------------------------------------------------------------
    # Sequence construction
    # ------------------------------------------------------------------

    def create_sequences(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        lookback: int,
        horizon: int,
        close_col_idx: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build sliding-window sequences for time-series modelling.

        Parameters
        ----------
        data:
            Scaled feature array of shape ``(n_samples, n_features)``.
        labels:
            Direction labels (0/1) of shape ``(n_samples,)``.
        lookback:
            Number of past time-steps the model sees as input.
        horizon:
            Number of future time-steps the model predicts.
        close_col_idx:
            Column index of the close price inside *data*.  When *None*
            the instance attribute ``_close_col_idx`` is used (set
            automatically by :meth:`prepare_all` or defaulting to 3).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(X, y_reg, y_cls)`` where:

            * ``X``     -- shape ``(num_samples, lookback, num_features)``
            * ``y_reg`` -- shape ``(num_samples, horizon)``  (future close prices, scaled)
            * ``y_cls`` -- shape ``(num_samples, horizon)``  (direction labels)

        No future data is leaked: each window's targets start **after** the
        window's last observation.
        """
        n_samples = len(data) - lookback - horizon + 1
        if n_samples <= 0:
            raise ValueError(
                f"Not enough data to create sequences: "
                f"len(data)={len(data)}, lookback={lookback}, horizon={horizon}."
            )

        n_features = data.shape[1]
        X = np.empty((n_samples, lookback, n_features), dtype=np.float32)
        y_reg = np.empty((n_samples, horizon), dtype=np.float32)
        y_cls = np.empty((n_samples, horizon), dtype=np.float32)

        close_idx = close_col_idx if close_col_idx is not None else self._close_col_idx

        for i in range(n_samples):
            X[i] = data[i : i + lookback]
            y_reg[i] = data[i + lookback : i + lookback + horizon, close_idx]
            y_cls[i] = labels[i + lookback : i + lookback + horizon]

        logger.info(
            "Created %d sequences (lookback=%d, horizon=%d, features=%d).",
            n_samples,
            lookback,
            n_features,
            n_features,
        )
        return X, y_reg, y_cls

    # ------------------------------------------------------------------
    # Scaler persistence
    # ------------------------------------------------------------------

    def save_scaler(self, path: Optional[str | Path] = None) -> Path:
        """Persist the fitted scaler to disk using joblib.

        Parameters
        ----------
        path:
            File path.  Defaults to ``data/scalers/scaler.joblib``.

        Returns
        -------
        Path
            Absolute path to the saved file.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted scaler.")

        filepath = Path(path) if path else SCALERS_DIR / "scaler.joblib"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, filepath)
        logger.info("Scaler saved to %s.", filepath)
        return filepath

    def load_scaler(self, path: str | Path) -> None:
        """Load a previously saved scaler from disk.

        Parameters
        ----------
        path:
            Path to the ``.joblib`` file.
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Scaler file not found: {filepath}")
        self.scaler = joblib.load(filepath)
        self._is_fitted = True
        logger.info("Scaler loaded from %s.", filepath)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def prepare_all(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_col: str,
        label_col: str,
        lookback: int,
        horizon: int,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> dict[str, np.ndarray]:
        """Execute the complete preprocessing pipeline.

        1. Split data temporally.
        2. Fit scaler on training set.
        3. Transform all splits.
        4. Build sequences for each split.

        Parameters
        ----------
        df:
            Feature-engineered DataFrame (NaN rows already dropped).
        feature_columns:
            Columns to use as model input features.
        target_col:
            Column name of the regression target (e.g. ``"close"``).
        label_col:
            Column name of the classification label (e.g. ``"direction"``).
        lookback:
            Sequence lookback window.
        horizon:
            Forecast horizon (number of future steps).
        train_ratio, val_ratio, test_ratio:
            Temporal split ratios.

        Returns
        -------
        dict[str, np.ndarray]
            Keys: ``X_train, y_train_reg, y_train_cls,
            X_val, y_val_reg, y_val_cls,
            X_test, y_test_reg, y_test_cls``.
        """
        # 1. Split
        train_df, val_df, test_df = self.split_data(
            df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        # 2. Fit scaler on training data only
        self.fit_scaler(train_df, feature_columns)

        # Store the close-column index for sequence construction.
        self._close_col_idx = feature_columns.index(target_col)

        # 3. Transform
        train_scaled = self.transform(train_df, feature_columns)
        val_scaled = self.transform(val_df, feature_columns)
        test_scaled = self.transform(test_df, feature_columns)

        # Extract classification labels *before* scaling (they are 0/1).
        train_labels = train_df[label_col].values
        val_labels = val_df[label_col].values
        test_labels = test_df[label_col].values

        # 4. Create sequences
        X_train, y_train_reg, y_train_cls = self.create_sequences(
            train_scaled, train_labels, lookback, horizon,
        )
        X_val, y_val_reg, y_val_cls = self.create_sequences(
            val_scaled, val_labels, lookback, horizon,
        )
        X_test, y_test_reg, y_test_cls = self.create_sequences(
            test_scaled, test_labels, lookback, horizon,
        )

        logger.info("Full preprocessing pipeline complete.")

        return {
            "X_train": X_train,
            "y_train_reg": y_train_reg,
            "y_train_cls": y_train_cls,
            "X_val": X_val,
            "y_val_reg": y_val_reg,
            "y_val_cls": y_val_cls,
            "X_test": X_test,
            "y_test_reg": y_test_reg,
            "y_test_cls": y_test_cls,
        }
