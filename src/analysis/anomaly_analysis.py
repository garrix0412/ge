"""
Anomaly-impact analysis for crypto price predictions.

Correlates detected anomalies (e.g. from an autoencoder) with prediction
errors to quantify how model accuracy changes around unusual market events.

Usage::

    from src.analysis.anomaly_analysis import AnomalyAnalyzer

    analyzer = AnomalyAnalyzer(anomaly_flags, prediction_errors, timestamps)
    report = analyzer.generate_report()
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.constants import FIGURES_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Known major crypto events (UTC timestamps)
# ---------------------------------------------------------------------------
_KNOWN_EVENTS: list[dict[str, Any]] = [
    {
        "name": "May 2021 Crash (China mining ban / Tesla reversal)",
        "start": datetime(2021, 5, 12, tzinfo=timezone.utc),
        "end": datetime(2021, 5, 25, tzinfo=timezone.utc),
    },
    {
        "name": "Sep 2021 China crypto ban",
        "start": datetime(2021, 9, 20, tzinfo=timezone.utc),
        "end": datetime(2021, 9, 28, tzinfo=timezone.utc),
    },
    {
        "name": "Jan 2022 Fed rate-hike selloff",
        "start": datetime(2022, 1, 20, tzinfo=timezone.utc),
        "end": datetime(2022, 1, 27, tzinfo=timezone.utc),
    },
    {
        "name": "May 2022 LUNA/UST collapse",
        "start": datetime(2022, 5, 7, tzinfo=timezone.utc),
        "end": datetime(2022, 5, 15, tzinfo=timezone.utc),
    },
    {
        "name": "Jun 2022 3AC / Celsius crisis",
        "start": datetime(2022, 6, 12, tzinfo=timezone.utc),
        "end": datetime(2022, 6, 20, tzinfo=timezone.utc),
    },
    {
        "name": "Nov 2022 FTX collapse",
        "start": datetime(2022, 11, 6, tzinfo=timezone.utc),
        "end": datetime(2022, 11, 14, tzinfo=timezone.utc),
    },
    {
        "name": "Mar 2023 SVB / USDC de-peg",
        "start": datetime(2023, 3, 10, tzinfo=timezone.utc),
        "end": datetime(2023, 3, 15, tzinfo=timezone.utc),
    },
    {
        "name": "Jan 2024 Bitcoin ETF approval",
        "start": datetime(2024, 1, 10, tzinfo=timezone.utc),
        "end": datetime(2024, 1, 15, tzinfo=timezone.utc),
    },
]


class AnomalyAnalyzer:
    """Correlate detected anomalies with prediction errors.

    Parameters
    ----------
    anomaly_flags:
        Boolean array where ``True`` indicates an anomaly at that timestep.
    prediction_errors:
        Absolute (or signed) prediction errors aligned with *anomaly_flags*.
    timestamps:
        Datetime-like timestamps aligned with the other arrays.  Needed for
        matching against known crypto events.
    reconstruction_errors:
        Optional array of autoencoder reconstruction errors.  When provided,
        the correlation between reconstruction error and prediction error
        is also computed.
    """

    def __init__(
        self,
        anomaly_flags: np.ndarray | pd.Series,
        prediction_errors: np.ndarray | pd.Series,
        timestamps: np.ndarray | pd.Series | pd.DatetimeIndex,
        reconstruction_errors: Optional[np.ndarray | pd.Series] = None,
    ) -> None:
        self.anomaly_flags = np.asarray(anomaly_flags, dtype=bool).ravel()
        self.prediction_errors = np.asarray(prediction_errors, dtype=np.float64).ravel()
        self.timestamps = pd.to_datetime(timestamps)
        self.reconstruction_errors = (
            np.asarray(reconstruction_errors, dtype=np.float64).ravel()
            if reconstruction_errors is not None
            else None
        )

        n = len(self.anomaly_flags)
        if not (len(self.prediction_errors) == n and len(self.timestamps) == n):
            raise ValueError(
                f"Length mismatch: anomaly_flags={n}, "
                f"prediction_errors={len(self.prediction_errors)}, "
                f"timestamps={len(self.timestamps)}."
            )
        if self.reconstruction_errors is not None and len(self.reconstruction_errors) != n:
            raise ValueError(
                f"reconstruction_errors length ({len(self.reconstruction_errors)}) "
                f"does not match anomaly_flags ({n})."
            )

        self._n_anomalies = int(self.anomaly_flags.sum())
        self._n_normal = n - self._n_anomalies
        logger.info(
            "AnomalyAnalyzer initialised — %d total, %d anomalies, %d normal.",
            n,
            self._n_anomalies,
            self._n_normal,
        )

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def correlate_anomalies_with_errors(self) -> dict[str, Any]:
        """Compare prediction errors during anomaly vs. normal periods.

        Returns
        -------
        dict[str, Any]
            - ``mean_error_anomaly`` / ``mean_error_normal``
            - ``median_error_anomaly`` / ``median_error_normal``
            - ``std_error_anomaly`` / ``std_error_normal``
            - ``test_statistic``, ``p_value``, ``test_name``
            - ``correlation`` (if reconstruction errors were provided)
            - ``significant`` (bool, p < 0.05)
        """
        errors_anomaly = self.prediction_errors[self.anomaly_flags]
        errors_normal = self.prediction_errors[~self.anomaly_flags]

        result: dict[str, Any] = {
            "n_anomaly": self._n_anomalies,
            "n_normal": self._n_normal,
            "mean_error_anomaly": float(np.mean(errors_anomaly)) if len(errors_anomaly) > 0 else float("nan"),
            "mean_error_normal": float(np.mean(errors_normal)) if len(errors_normal) > 0 else float("nan"),
            "median_error_anomaly": float(np.median(errors_anomaly)) if len(errors_anomaly) > 0 else float("nan"),
            "median_error_normal": float(np.median(errors_normal)) if len(errors_normal) > 0 else float("nan"),
            "std_error_anomaly": float(np.std(errors_anomaly)) if len(errors_anomaly) > 0 else float("nan"),
            "std_error_normal": float(np.std(errors_normal)) if len(errors_normal) > 0 else float("nan"),
        }

        # Statistical test: Mann-Whitney U (does not assume normality)
        if len(errors_anomaly) >= 2 and len(errors_normal) >= 2:
            try:
                stat, p_val = stats.mannwhitneyu(
                    errors_anomaly, errors_normal, alternative="two-sided"
                )
                result["test_name"] = "Mann-Whitney U"
                result["test_statistic"] = float(stat)
                result["p_value"] = float(p_val)
                result["significant"] = p_val < 0.05
            except Exception as exc:
                logger.warning("Mann-Whitney test failed: %s", exc)
                result["test_name"] = "Mann-Whitney U"
                result["test_statistic"] = float("nan")
                result["p_value"] = float("nan")
                result["significant"] = False
        else:
            logger.warning(
                "Not enough samples for statistical test "
                "(anomaly=%d, normal=%d).",
                len(errors_anomaly),
                len(errors_normal),
            )
            result["test_name"] = None
            result["test_statistic"] = float("nan")
            result["p_value"] = float("nan")
            result["significant"] = False

        # Correlation between reconstruction error and prediction error
        if self.reconstruction_errors is not None:
            valid = ~(np.isnan(self.reconstruction_errors) | np.isnan(self.prediction_errors))
            if valid.sum() >= 3:
                corr, corr_p = stats.pearsonr(
                    self.reconstruction_errors[valid],
                    self.prediction_errors[valid],
                )
                result["correlation"] = float(corr)
                result["correlation_p_value"] = float(corr_p)
            else:
                result["correlation"] = float("nan")
                result["correlation_p_value"] = float("nan")

        logger.info(
            "Anomaly correlation — mean error anomaly: %.6f vs normal: %.6f "
            "(p=%.4g, significant=%s)",
            result["mean_error_anomaly"],
            result["mean_error_normal"],
            result.get("p_value", float("nan")),
            result.get("significant", "N/A"),
        )

        return result

    # ------------------------------------------------------------------
    # Known-event matching
    # ------------------------------------------------------------------

    @staticmethod
    def identify_known_events(
        timestamps: np.ndarray | pd.Series | pd.DatetimeIndex,
        anomaly_flags: np.ndarray | pd.Series,
    ) -> list[dict[str, Any]]:
        """Check whether detected anomalies overlap with known crypto events.

        Parameters
        ----------
        timestamps:
            Datetime-like array of observation times.
        anomaly_flags:
            Boolean array of detected anomalies.

        Returns
        -------
        list[dict[str, Any]]
            Each entry: ``{"event_name": str, "start": datetime, "end": datetime, "detected": bool}``.
        """
        timestamps = pd.to_datetime(timestamps)
        anomaly_flags = np.asarray(anomaly_flags, dtype=bool).ravel()

        # Make timestamps timezone-aware (UTC) if they are not already
        if timestamps.tz is None:
            timestamps_utc = timestamps.tz_localize("UTC")
        else:
            timestamps_utc = timestamps.tz_convert("UTC")

        results: list[dict[str, Any]] = []

        for event in _KNOWN_EVENTS:
            event_start = event["start"]
            event_end = event["end"]

            # Check if any anomaly was detected within the event window
            in_window = (timestamps_utc >= event_start) & (timestamps_utc <= event_end)

            if not in_window.any():
                # Event window not covered by the data
                continue

            detected = bool(anomaly_flags[in_window].any())
            n_detected = int(anomaly_flags[in_window].sum())
            n_total = int(in_window.sum())

            results.append(
                {
                    "event_name": event["name"],
                    "start": event_start.isoformat(),
                    "end": event_end.isoformat(),
                    "detected": detected,
                    "anomalies_in_window": n_detected,
                    "datapoints_in_window": n_total,
                    "detection_rate": round(n_detected / n_total, 4) if n_total > 0 else 0.0,
                }
            )

            status = "DETECTED" if detected else "MISSED"
            logger.info(
                "Event '%s': %s (%d/%d anomalies in window)",
                event["name"],
                status,
                n_detected,
                n_total,
            )

        return results

    # ------------------------------------------------------------------
    # Anomaly impact windows
    # ------------------------------------------------------------------

    def compute_anomaly_impact(
        self,
        window_before: int = 24,
        window_after: int = 24,
    ) -> dict[str, Any]:
        """Measure prediction-error changes around anomaly onsets.

        For each anomaly onset (first ``True`` in a contiguous anomaly block),
        collect prediction errors in a window *before* and *after* the onset,
        then report aggregate statistics.

        Parameters
        ----------
        window_before:
            Number of time steps before the anomaly to consider.
        window_after:
            Number of time steps after the anomaly to consider.

        Returns
        -------
        dict[str, Any]
            - ``mean_error_before``, ``mean_error_after``, ``mean_error_during``
            - ``n_onsets``: number of anomaly onset events
            - ``error_increase_pct``: relative change in mean error (before -> after)
        """
        n = len(self.anomaly_flags)

        # Find anomaly onsets (transitions from False -> True)
        onset_indices: list[int] = []
        for i in range(n):
            if self.anomaly_flags[i] and (i == 0 or not self.anomaly_flags[i - 1]):
                onset_indices.append(i)

        if not onset_indices:
            logger.warning("No anomaly onsets found — cannot compute impact.")
            return {
                "mean_error_before": float("nan"),
                "mean_error_after": float("nan"),
                "mean_error_during": float("nan"),
                "n_onsets": 0,
                "error_increase_pct": float("nan"),
            }

        errors_before: list[float] = []
        errors_after: list[float] = []
        errors_during: list[float] = []

        for onset in onset_indices:
            # Before window
            start_before = max(0, onset - window_before)
            before_slice = self.prediction_errors[start_before:onset]
            errors_before.extend(before_slice.tolist())

            # After window
            end_after = min(n, onset + window_after)
            after_slice = self.prediction_errors[onset:end_after]
            errors_after.extend(after_slice.tolist())

            # During the anomaly (contiguous block starting at onset)
            j = onset
            while j < n and self.anomaly_flags[j]:
                errors_during.append(float(self.prediction_errors[j]))
                j += 1

        mean_before = float(np.mean(errors_before)) if errors_before else float("nan")
        mean_after = float(np.mean(errors_after)) if errors_after else float("nan")
        mean_during = float(np.mean(errors_during)) if errors_during else float("nan")

        error_increase_pct = float("nan")
        if mean_before > 0 and not np.isnan(mean_before) and not np.isnan(mean_after):
            error_increase_pct = (mean_after - mean_before) / mean_before * 100.0

        result = {
            "mean_error_before": mean_before,
            "mean_error_after": mean_after,
            "mean_error_during": mean_during,
            "n_onsets": len(onset_indices),
            "error_increase_pct": round(error_increase_pct, 2),
            "window_before": window_before,
            "window_after": window_after,
        }

        logger.info(
            "Anomaly impact — %d onsets, mean error before: %.6f, "
            "after: %.6f, during: %.6f (change: %.1f%%)",
            len(onset_indices),
            mean_before,
            mean_after,
            mean_during,
            error_increase_pct if not np.isnan(error_increase_pct) else 0.0,
        )

        return result

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def generate_report(self) -> dict[str, Any]:
        """Run all analyses and return a consolidated summary.

        Returns
        -------
        dict[str, Any]
            Keys: ``correlation``, ``known_events``, ``impact``, ``summary``.
        """
        correlation = self.correlate_anomalies_with_errors()
        known_events = self.identify_known_events(self.timestamps, self.anomaly_flags)
        impact = self.compute_anomaly_impact()

        n_known_detected = sum(1 for e in known_events if e["detected"])
        n_known_total = len(known_events)

        report: dict[str, Any] = {
            "correlation": correlation,
            "known_events": known_events,
            "impact": impact,
            "summary": {
                "total_observations": len(self.anomaly_flags),
                "total_anomalies": self._n_anomalies,
                "anomaly_rate_pct": round(self._n_anomalies / len(self.anomaly_flags) * 100, 2),
                "known_events_detected": n_known_detected,
                "known_events_total": n_known_total,
                "error_significantly_different": correlation.get("significant", False),
            },
        }

        logger.info(
            "Anomaly report generated — %d anomalies (%.1f%%), "
            "%d/%d known events detected, significant=%s",
            self._n_anomalies,
            report["summary"]["anomaly_rate_pct"],
            n_known_detected,
            n_known_total,
            correlation.get("significant", "N/A"),
        )

        return report
