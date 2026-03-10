"""
Metrics computation for regression and classification evaluation.

Provides a single :class:`Evaluator` class with static methods for every
metric used in the crypto-price prediction project.  Regression metrics
assess price-level accuracy; classification metrics assess directional
(up/down) prediction quality.

Usage::

    from src.training.evaluator import Evaluator

    reg = Evaluator.evaluate_regression(y_true, y_pred)
    cls = Evaluator.evaluate_classification(y_true_cls, y_pred_cls)
    all_metrics = Evaluator.evaluate_all(y_true, y_pred, y_true_cls, y_pred_cls)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score as sklearn_f1,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Static-method collection for regression and classification metrics.

    Every public method accepts NumPy arrays and returns either a single
    scalar or a ``dict[str, float]`` of named metrics.
    """

    # ------------------------------------------------------------------
    # Regression metrics
    # ------------------------------------------------------------------

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error.

        Parameters
        ----------
        y_true : Ground-truth values.
        y_pred : Predicted values.

        Returns
        -------
        float
            Non-negative MAE value.
        """
        value = float(mean_absolute_error(y_true, y_pred))
        assert value >= 0, f"MAE must be >= 0, got {value}"
        return value

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error.

        Parameters
        ----------
        y_true : Ground-truth values.
        y_pred : Predicted values.

        Returns
        -------
        float
            Non-negative RMSE value.
        """
        value = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        assert value >= 0, f"RMSE must be >= 0, got {value}"
        return value

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error.

        Entries where ``y_true == 0`` are excluded to avoid division by
        zero.  If *all* entries are zero the function returns ``0.0``.

        Parameters
        ----------
        y_true : Ground-truth values.
        y_pred : Predicted values.

        Returns
        -------
        float
            MAPE as a percentage in ``[0, 100)``.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        mask = y_true != 0.0
        if not mask.any():
            logger.warning("All y_true values are zero; MAPE is undefined. Returning 0.0.")
            return 0.0

        value = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
        assert value < 100.0 or np.isclose(value, 100.0, atol=1e-6), (
            f"MAPE should be < 100%% for a reasonable model, got {value:.2f}%%"
        )
        return value

    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Percentage of correctly predicted price-change directions.

        Direction is inferred from consecutive differences (sign of delta).
        The first element is dropped because no prior value exists to
        compute a direction.

        Parameters
        ----------
        y_true : Ground-truth price series.
        y_pred : Predicted price series.

        Returns
        -------
        float
            Fraction of correct directions in ``[0, 1]``.
        """
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

        if len(y_true) < 2:
            logger.warning("Need at least 2 values for directional accuracy. Returning 0.0.")
            return 0.0

        true_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))

        value = float(np.mean(true_dir == pred_dir))
        assert 0.0 <= value <= 1.0, f"Directional accuracy must be in [0, 1], got {value}"
        return value

    # ------------------------------------------------------------------
    # Classification metrics
    # ------------------------------------------------------------------

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
        """Binary classification accuracy.

        Parameters
        ----------
        y_true : Ground-truth binary labels (0 or 1).
        y_pred : Predicted probabilities or logits.
        threshold : Decision boundary for converting probabilities to labels.

        Returns
        -------
        float
            Accuracy in ``[0, 1]``.
        """
        y_pred_binary = (np.asarray(y_pred) >= threshold).astype(int)
        value = float(accuracy_score(y_true, y_pred_binary))
        assert 0.0 <= value <= 1.0, f"Accuracy must be in [0, 1], got {value}"
        return value

    @staticmethod
    def f1_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
        """Binary F1 score.

        Parameters
        ----------
        y_true : Ground-truth binary labels.
        y_pred : Predicted probabilities or logits.
        threshold : Decision boundary.

        Returns
        -------
        float
            F1 score in ``[0, 1]``.
        """
        y_pred_binary = (np.asarray(y_pred) >= threshold).astype(int)
        value = float(sklearn_f1(y_true, y_pred_binary, zero_division=0.0))
        assert 0.0 <= value <= 1.0, f"F1 must be in [0, 1], got {value}"
        return value

    @staticmethod
    def auc_roc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Area Under the ROC Curve.

        Returns ``0.5`` (random-chance baseline) when only one class is
        present in *y_true*, since AUC is undefined in that case.

        Parameters
        ----------
        y_true : Ground-truth binary labels.
        y_pred : Predicted probabilities (continuous).

        Returns
        -------
        float
            AUC-ROC in ``[0, 1]``.
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            logger.warning(
                "Only one class present in y_true (%s). AUC is undefined; returning 0.5.",
                unique_classes,
            )
            return 0.5

        try:
            value = float(roc_auc_score(y_true, y_pred))
        except ValueError as exc:
            logger.warning("AUC computation failed (%s). Returning 0.5.", exc)
            return 0.5

        return value

    # ------------------------------------------------------------------
    # Aggregate helpers
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Compute all regression metrics at once.

        Parameters
        ----------
        y_true : Ground-truth values.
        y_pred : Predicted values.

        Returns
        -------
        dict[str, float]
            Keys: ``mae``, ``rmse``, ``mape``, ``directional_accuracy``.
        """
        metrics = {
            "mae": Evaluator.mae(y_true, y_pred),
            "rmse": Evaluator.rmse(y_true, y_pred),
            "mape": Evaluator.mape(y_true, y_pred),
            "directional_accuracy": Evaluator.directional_accuracy(y_true, y_pred),
        }
        logger.info(
            "Regression metrics — MAE: %.6f | RMSE: %.6f | MAPE: %.2f%% | DirAcc: %.4f",
            metrics["mae"],
            metrics["rmse"],
            metrics["mape"],
            metrics["directional_accuracy"],
        )
        return metrics

    @staticmethod
    def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Compute all classification metrics at once.

        Parameters
        ----------
        y_true : Ground-truth binary labels.
        y_pred : Predicted probabilities.

        Returns
        -------
        dict[str, float]
            Keys: ``accuracy``, ``f1_score``, ``auc_roc``.
        """
        metrics = {
            "accuracy": Evaluator.accuracy(y_true, y_pred),
            "f1_score": Evaluator.f1_score(y_true, y_pred),
            "auc_roc": Evaluator.auc_roc(y_true, y_pred),
        }
        logger.info(
            "Classification metrics — Acc: %.4f | F1: %.4f | AUC: %.4f",
            metrics["accuracy"],
            metrics["f1_score"],
            metrics["auc_roc"],
        )
        return metrics

    @staticmethod
    def evaluate_all(
        y_true_reg: np.ndarray,
        y_pred_reg: np.ndarray,
        y_true_cls: np.ndarray,
        y_pred_cls: np.ndarray,
    ) -> dict[str, float]:
        """Compute combined regression and classification metrics.

        Parameters
        ----------
        y_true_reg : Ground-truth regression targets.
        y_pred_reg : Predicted regression values.
        y_true_cls : Ground-truth classification labels.
        y_pred_cls : Predicted classification probabilities.

        Returns
        -------
        dict[str, float]
            Merged dictionary with all regression and classification metrics.
            Regression keys are prefixed with ``reg_`` and classification
            keys with ``cls_``.
        """
        reg_metrics = Evaluator.evaluate_regression(y_true_reg, y_pred_reg)
        cls_metrics = Evaluator.evaluate_classification(y_true_cls, y_pred_cls)

        combined: dict[str, float] = {}
        for key, val in reg_metrics.items():
            combined[f"reg_{key}"] = val
        for key, val in cls_metrics.items():
            combined[f"cls_{key}"] = val

        logger.info("All metrics computed: %s", combined)
        return combined
