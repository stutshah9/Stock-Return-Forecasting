"""Event-conditioned conformal calibration utilities."""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np


SURPRISE_BANDS = ("low_surprise", "medium_surprise", "high_surprise")
VOLATILITY_BANDS = ("low_vol", "high_vol")
VALID_REGIMES = tuple(
    f"{surprise_band}_{volatility_band}"
    for surprise_band in SURPRISE_BANDS
    for volatility_band in VOLATILITY_BANDS
)
LEGACY_REGIME_MAP = {
    "low": "low_surprise",
    "mid": "medium_surprise",
    "medium": "medium_surprise",
    "high": "high_surprise",
    "low_surprise": "low_surprise_low_vol",
    "medium_surprise": "medium_surprise_low_vol",
    "high_surprise": "high_surprise_high_vol",
}


def _to_float(value: Any) -> float:
    """Convert a scalar-like value to ``float``."""

    try:
        return float(value.item())
    except AttributeError:
        return float(value)


def _normalize_regime(regime: str) -> str:
    """Normalize regime labels to the canonical surprise/volatility regime names."""

    normalized_regime = LEGACY_REGIME_MAP.get(regime, regime)
    if normalized_regime not in VALID_REGIMES:
        raise ValueError(f"Unsupported regime: {regime}")
    return normalized_regime


def _conformal_quantile(scores: list[float], quantile_level: float) -> float:
    """Compute a conservative conformal quantile for a score list."""

    clamped_level = min(max(quantile_level, 0.0), 1.0)
    scores_array = np.asarray(scores, dtype=float)
    try:
        return float(np.quantile(scores_array, clamped_level, method="higher"))
    except TypeError:
        return float(
            np.quantile(scores_array, clamped_level, interpolation="higher")
        )


def assign_regime(
    sue: float,
    implied_vol: float = 0.0,
    low_thresh: float = 0.5,
    high_thresh: float = 1.5,
    vol_thresh: float = 0.30,
) -> str:
    """Assign an event to a conformal regime using surprise and volatility.

    Args:
        sue: Standardized unexpected earnings value.
        low_thresh: Magnitude threshold below which an event is low surprise.
        high_thresh: Magnitude threshold above which an event is high surprise.
        implied_vol: Pre-event implied or realized volatility proxy.
        vol_thresh: Threshold separating low- and high-volatility events.

    Returns:
        One of the composite regime labels in ``VALID_REGIMES``.
    """

    absolute_sue = abs(float(sue))
    if absolute_sue < low_thresh:
        surprise_band = "low_surprise"
    elif absolute_sue < high_thresh:
        surprise_band = "medium_surprise"
    else:
        surprise_band = "high_surprise"

    try:
        volatility_band = "high_vol" if float(implied_vol) >= vol_thresh else "low_vol"
    except Exception:
        volatility_band = "low_vol"
    return f"{surprise_band}_{volatility_band}"


class EventConditionedConformalPredictor:
    """Calibrate regime-aware conformal intervals for forecasted event returns."""

    def __init__(self, coverage_levels: list[float] = [0.80, 0.90, 0.95]) -> None:
        """Initialize the predictor with target coverage levels.

        Args:
            coverage_levels: Coverage levels to calibrate for each regime.
        """

        self.coverage_levels = coverage_levels
        self.thresholds: dict[tuple[str, float], float] = {}

    def calibrate(
        self,
        cal_outputs: list[dict[str, Any]] | None = None,
        cal_labels: list[float] | None = None,
        cal_regimes: list[str] | None = None,
        **legacy_kwargs: Any,
    ) -> None:
        """Fit regime-specific nonconformity thresholds from calibration data.

        Args:
            cal_outputs: Model outputs containing ``mu``, ``log_sigma``, and
                ``introspective_score``.
            cal_labels: Ground-truth calibration labels.
            cal_regimes: Event-regime label for each calibration sample.
            **legacy_kwargs: Compatibility aliases such as ``outputs`` or
                ``regimes`` from older call sites.
        """

        if cal_outputs is None:
            cal_outputs = legacy_kwargs.get("outputs")
        if cal_labels is None:
            cal_labels = legacy_kwargs.get("labels")
        if cal_regimes is None:
            cal_regimes = legacy_kwargs.get("regimes")

        if cal_outputs is None or cal_labels is None or cal_regimes is None:
            raise ValueError("Calibration requires outputs, labels, and regimes.")
        if not (len(cal_outputs) == len(cal_labels) == len(cal_regimes)):
            raise ValueError("Calibration inputs must have the same length.")

        regime_scores: dict[str, list[float]] = {}
        for output, label, regime in zip(cal_outputs, cal_labels, cal_regimes):
            normalized_regime = _normalize_regime(regime)
            mu = _to_float(output["mu"])
            log_sigma = _to_float(output["log_sigma"])
            sigma = math.exp(log_sigma)
            score = abs(float(label) - mu) / sigma
            regime_scores.setdefault(normalized_regime, []).append(score)

        for regime, scores in regime_scores.items():
            n = len(scores)
            if n == 0:
                continue
            for coverage in self.coverage_levels:
                quantile_level = coverage * (1.0 + 1.0 / n)
                threshold = _conformal_quantile(scores, quantile_level)
                self.thresholds[(regime, float(coverage))] = threshold

    def predict_interval(
        self,
        output: dict[str, Any],
        regime: str = "medium_surprise",
        coverage: float = 0.90,
    ) -> tuple[float, float]:
        """Construct a conformal interval for a single model output.

        Args:
            output: Model output dictionary with ``mu``, ``log_sigma``, and
                ``introspective_score`` entries.
            regime: Event regime for the example.
            coverage: Desired calibrated coverage level.

        Returns:
            Lower and upper prediction interval bounds.
        """

        normalized_regime = _normalize_regime(regime)
        threshold_key = (normalized_regime, float(coverage))
        if threshold_key not in self.thresholds:
            raise KeyError(
                f"Missing conformal threshold for regime={normalized_regime}, "
                f"coverage={coverage}."
            )

        mu = _to_float(output["mu"])
        log_sigma = _to_float(output["log_sigma"])
        introspective_score = _to_float(output["introspective_score"])

        threshold = self.thresholds[threshold_key]
        base_half_width = threshold * math.exp(log_sigma)
        adjustment = 1.0 + 0.15 * (1.0 - introspective_score)
        adjusted_half_width = base_half_width * adjustment

        return mu - adjusted_half_width, mu + adjusted_half_width

    def selective_predict(
        self,
        output: dict[str, Any],
        regime: str = "medium_surprise",
        coverage: float = 0.90,
        min_score: float = 0.6,
    ) -> tuple[float, float] | None:
        """Return an interval or abstain based on confidence and sign ambiguity.

        Args:
            output: Model output dictionary.
            regime: Event regime for the example.
            coverage: Desired calibrated coverage level.
            min_score: Minimum acceptable introspective confidence.

        Returns:
            The calibrated interval, or ``None`` when abstaining.
        """

        introspective_score = _to_float(output["introspective_score"])
        if introspective_score < min_score:
            return None

        lower, upper = self.predict_interval(
            output=output,
            regime=regime,
            coverage=coverage,
        )
        if lower <= 0.0 <= upper:
            return None
        return lower, upper


class RegimeConformalPredictor(EventConditionedConformalPredictor):
    """Backward-compatible alias for the previous conformal predictor name."""


if __name__ == "__main__":
    random.seed(42)

    predictor = EventConditionedConformalPredictor()
    calibration_outputs: list[dict[str, float]] = []
    calibration_labels: list[float] = []
    calibration_regimes: list[str] = []

    for _ in range(50):
        mu = random.uniform(-0.06, 0.06)
        log_sigma = random.uniform(-1.2, 0.8)
        score = random.uniform(0.1, 0.95)
        label = mu + random.gauss(0.0, math.exp(log_sigma))
        sue = random.uniform(-2.5, 2.5)
        implied_vol = random.uniform(0.1, 0.6)

        calibration_outputs.append(
            {
                "mu": mu,
                "log_sigma": log_sigma,
                "introspective_score": score,
            }
        )
        calibration_labels.append(label)
        calibration_regimes.append(assign_regime(sue=sue, implied_vol=implied_vol))

    predictor.calibrate(
        cal_outputs=calibration_outputs,
        cal_labels=calibration_labels,
        cal_regimes=calibration_regimes,
    )

    sample_output = {
        "mu": 0.01,
        "log_sigma": -0.2,
        "introspective_score": 0.72,
    }
    sample_regime = "medium_surprise_low_vol"
    interval = predictor.predict_interval(sample_output, regime=sample_regime)
    decision = predictor.selective_predict(sample_output, regime=sample_regime)

    print("calibrated_keys", len(predictor.thresholds))
    print("interval", interval)
    print("selective_decision", decision)
