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
AUXILIARY_BANDS = ("low", "medium", "high")
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


def _event_feature_triplet(event: dict[str, Any]) -> tuple[float, float, float]:
    """Extract ``(sue, momentum, implied_vol)`` from a cached event dictionary."""

    raw_features = event.get("raw_features", event.get("features", []))
    if hasattr(raw_features, "detach"):
        values = [_to_float(value) for value in raw_features.detach().cpu().view(-1).tolist()[:3]]
    elif isinstance(raw_features, dict):
        values = [
            _to_float(raw_features.get("sue", 0.0)),
            _to_float(raw_features.get("momentum", 0.0)),
            _to_float(raw_features.get("implied_vol", 0.0)),
        ]
    else:
        values = [_to_float(value) for value in list(raw_features)[:3]]

    while len(values) < 3:
        values.append(0.0)
    sue, momentum, implied_vol = values[:3]
    return float(sue), float(momentum), float(implied_vol)


def fit_regime_thresholds(
    events: list[dict[str, Any]],
    low_quantile: float = 0.60,
    high_quantile: float = 0.90,
    vol_quantile: float = 0.60,
) -> dict[str, float]:
    """Fit regime thresholds from pre-calibration event features.

    The proposal conditions conformal intervals on observable event features
    such as surprise magnitude and pre-event volatility. Using quantiles from
    historical events keeps regime sizes more balanced than fixed hand-tuned
    cutoffs, especially when the event distribution is skewed.
    """

    absolute_sues: list[float] = []
    implied_vols: list[float] = []
    for event in events:
        sue, _momentum, implied_vol = _event_feature_triplet(event)
        if math.isfinite(sue):
            absolute_sues.append(abs(float(sue)))
        if math.isfinite(implied_vol):
            implied_vols.append(float(implied_vol))

    if len(absolute_sues) < 10 or len(implied_vols) < 10:
        return {
            "low_thresh": 0.5,
            "high_thresh": 1.5,
            "vol_thresh": 0.30,
        }

    low_thresh = float(np.quantile(np.asarray(absolute_sues, dtype=float), low_quantile))
    high_thresh = float(np.quantile(np.asarray(absolute_sues, dtype=float), high_quantile))
    vol_thresh = float(np.quantile(np.asarray(implied_vols, dtype=float), vol_quantile))

    if not math.isfinite(low_thresh):
        low_thresh = 0.5
    if not math.isfinite(high_thresh):
        high_thresh = max(low_thresh + 0.5, 1.5)
    if not math.isfinite(vol_thresh):
        vol_thresh = 0.30

    if high_thresh <= low_thresh:
        high_thresh = low_thresh + max(abs(low_thresh) * 0.25, 0.25)

    return {
        "low_thresh": float(low_thresh),
        "high_thresh": float(high_thresh),
        "vol_thresh": float(vol_thresh),
    }


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


def _fit_band_edges(
    values: list[float],
    low_quantile: float,
    high_quantile: float,
) -> dict[str, float] | None:
    """Fit low/high cut points for a three-band partition."""

    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite_values) < 12:
        return None

    values_array = np.asarray(finite_values, dtype=float)
    value_range = float(np.max(values_array) - np.min(values_array))
    if value_range <= 1e-8:
        return None

    low_cut = float(np.quantile(values_array, low_quantile))
    high_cut = float(np.quantile(values_array, high_quantile))
    if not math.isfinite(low_cut) or not math.isfinite(high_cut):
        return None
    if high_cut <= low_cut:
        high_cut = low_cut + max(value_range * 0.05, 1e-6)

    return {
        "low_cut": low_cut,
        "high_cut": high_cut,
    }


def _assign_band(value: float, edges: dict[str, float] | None) -> str:
    """Map a scalar to a low/medium/high band."""

    if edges is None or not math.isfinite(float(value)):
        return "medium"
    if float(value) < float(edges["low_cut"]):
        return "low"
    if float(value) < float(edges["high_cut"]):
        return "medium"
    return "high"


def _message_volume(metadata: Any) -> float:
    """Extract message volume from optional calibration metadata."""

    if isinstance(metadata, dict):
        return _to_float(metadata.get("message_volume", 0.0))
    if isinstance(metadata, (list, tuple)) and len(metadata) >= 2:
        return _to_float(metadata[1])
    return 0.0


def assign_regime(
    sue: float,
    implied_vol: float = 0.0,
    low_thresh: float = 0.5,
    high_thresh: float = 1.5,
    vol_thresh: float = 0.30,
    thresholds: dict[str, float] | None = None,
) -> str:
    """Assign an event to a conformal regime using surprise and volatility.

    Args:
        sue: Standardized unexpected earnings value.
        low_thresh: Magnitude threshold below which an event is low surprise.
        high_thresh: Magnitude threshold above which an event is high surprise.
        implied_vol: Pre-event implied or realized volatility proxy.
        vol_thresh: Threshold separating low- and high-volatility events.
        thresholds: Optional fitted thresholds dictionary overriding the scalar
            threshold arguments.

    Returns:
        One of the composite regime labels in ``VALID_REGIMES``.
    """

    if thresholds is not None:
        low_thresh = float(thresholds.get("low_thresh", low_thresh))
        high_thresh = float(thresholds.get("high_thresh", high_thresh))
        vol_thresh = float(thresholds.get("vol_thresh", vol_thresh))

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

    def __init__(
        self,
        coverage_levels: list[float] = [0.80, 0.90, 0.95],
        minimum_bucket_size: int = 24,
    ) -> None:
        """Initialize the predictor with target coverage levels.

        Args:
            coverage_levels: Coverage levels to calibrate for each regime.
        """

        self.coverage_levels = coverage_levels
        self.thresholds: dict[tuple[str, float], float] = {}
        self.minimum_bucket_size = max(int(minimum_bucket_size), 8)
        self.score_band_thresholds: dict[tuple[str, str, float], float] = {}
        self.sigma_band_thresholds: dict[tuple[str, str, float], float] = {}
        self.attention_band_thresholds: dict[tuple[str, str, float], float] = {}
        self.global_score_band_thresholds: dict[tuple[str, float], float] = {}
        self.global_sigma_band_thresholds: dict[tuple[str, float], float] = {}
        self.global_attention_band_thresholds: dict[tuple[str, float], float] = {}
        self.score_band_edges: dict[str, float] | None = None
        self.sigma_band_edges: dict[str, float] | None = None
        self.attention_band_edges: dict[str, float] | None = None

    def calibrate(
        self,
        cal_outputs: list[dict[str, Any]] | None = None,
        cal_labels: list[float] | None = None,
        cal_regimes: list[str] | None = None,
        cal_metadata: list[dict[str, float] | list[float] | tuple[float, ...] | None] | None = None,
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
        if cal_metadata is None:
            cal_metadata = legacy_kwargs.get("metadata")

        if cal_outputs is None or cal_labels is None or cal_regimes is None:
            raise ValueError("Calibration requires outputs, labels, and regimes.")
        if cal_metadata is None:
            cal_metadata = [{} for _ in cal_outputs]
        if not (len(cal_outputs) == len(cal_labels) == len(cal_regimes)):
            raise ValueError("Calibration inputs must have the same length.")
        if len(cal_metadata) != len(cal_outputs):
            raise ValueError("Calibration metadata must align with calibration outputs.")

        self.thresholds.clear()
        self.score_band_thresholds.clear()
        self.sigma_band_thresholds.clear()
        self.attention_band_thresholds.clear()
        self.global_score_band_thresholds.clear()
        self.global_sigma_band_thresholds.clear()
        self.global_attention_band_thresholds.clear()

        regime_scores: dict[str, list[float]] = {}
        calibration_rows: list[dict[str, float | str]] = []
        sigma_values: list[float] = []
        confidence_values: list[float] = []
        attention_values: list[float] = []

        for output, label, regime, metadata in zip(
            cal_outputs,
            cal_labels,
            cal_regimes,
            cal_metadata,
        ):
            normalized_regime = _normalize_regime(regime)
            mu = _to_float(output["mu"])
            log_sigma = _to_float(output["log_sigma"])
            sigma = max(math.exp(log_sigma), 1e-6)
            nonconformity = abs(float(label) - mu) / sigma
            confidence = _to_float(output.get("introspective_score", 0.5))
            message_volume = _message_volume(metadata)
            regime_scores.setdefault(normalized_regime, []).append(nonconformity)
            calibration_rows.append(
                {
                    "regime": normalized_regime,
                    "nonconformity": float(nonconformity),
                    "sigma": float(sigma),
                    "confidence": float(confidence),
                    "message_volume": float(message_volume),
                }
            )
            sigma_values.append(float(sigma))
            confidence_values.append(float(confidence))
            if math.isfinite(message_volume):
                attention_values.append(float(message_volume))

        for regime, scores in regime_scores.items():
            n = len(scores)
            if n == 0:
                continue
            for coverage in self.coverage_levels:
                quantile_level = coverage * (1.0 + 1.0 / n)
                threshold = _conformal_quantile(scores, quantile_level)
                self.thresholds[(regime, float(coverage))] = threshold

        self.score_band_edges = _fit_band_edges(
            confidence_values,
            low_quantile=0.25,
            high_quantile=0.75,
        )
        self.sigma_band_edges = _fit_band_edges(
            sigma_values,
            low_quantile=0.33,
            high_quantile=0.67,
        )
        self.attention_band_edges = _fit_band_edges(
            attention_values,
            low_quantile=0.50,
            high_quantile=0.85,
        )

        self._fit_auxiliary_thresholds(
            calibration_rows,
            band_edges=self.score_band_edges,
            value_key="confidence",
            regime_store=self.score_band_thresholds,
            global_store=self.global_score_band_thresholds,
        )
        self._fit_auxiliary_thresholds(
            calibration_rows,
            band_edges=self.sigma_band_edges,
            value_key="sigma",
            regime_store=self.sigma_band_thresholds,
            global_store=self.global_sigma_band_thresholds,
        )
        self._fit_auxiliary_thresholds(
            calibration_rows,
            band_edges=self.attention_band_edges,
            value_key="message_volume",
            regime_store=self.attention_band_thresholds,
            global_store=self.global_attention_band_thresholds,
        )

    def _fit_auxiliary_thresholds(
        self,
        calibration_rows: list[dict[str, float | str]],
        band_edges: dict[str, float] | None,
        value_key: str,
        regime_store: dict[tuple[str, str, float], float],
        global_store: dict[tuple[str, float], float],
    ) -> None:
        """Fit band-conditioned threshold tables from calibration rows."""

        if band_edges is None:
            return

        band_scores_by_regime: dict[tuple[str, str], list[float]] = {}
        global_band_scores: dict[str, list[float]] = {}
        for row in calibration_rows:
            band = _assign_band(float(row[value_key]), band_edges)
            nonconformity = float(row["nonconformity"])
            regime = str(row["regime"])
            band_scores_by_regime.setdefault((regime, band), []).append(nonconformity)
            global_band_scores.setdefault(band, []).append(nonconformity)

        for (regime, band), scores in band_scores_by_regime.items():
            if len(scores) < self.minimum_bucket_size:
                continue
            for coverage in self.coverage_levels:
                quantile_level = coverage * (1.0 + 1.0 / len(scores))
                regime_store[(regime, band, float(coverage))] = _conformal_quantile(
                    scores,
                    quantile_level,
                )

        for band, scores in global_band_scores.items():
            if len(scores) < self.minimum_bucket_size:
                continue
            for coverage in self.coverage_levels:
                quantile_level = coverage * (1.0 + 1.0 / len(scores))
                global_store[(band, float(coverage))] = _conformal_quantile(
                    scores,
                    quantile_level,
                )

    def _combined_threshold(
        self,
        normalized_regime: str,
        coverage: float,
        output: dict[str, Any],
        metadata: dict[str, float] | list[float] | tuple[float, ...] | None = None,
    ) -> tuple[float, dict[str, float | str]]:
        """Combine base regime calibration with local uncertainty-aware corrections."""

        threshold_key = (normalized_regime, float(coverage))
        if threshold_key not in self.thresholds:
            raise KeyError(
                f"Missing conformal threshold for regime={normalized_regime}, "
                f"coverage={coverage}."
            )

        base_threshold = float(self.thresholds[threshold_key])
        sigma = max(math.exp(_to_float(output["log_sigma"])), 1e-6)
        confidence = _to_float(output.get("introspective_score", 0.5))
        message_volume = _message_volume(metadata)

        components: dict[str, float | str] = {
            "base_regime_threshold": float(base_threshold),
            "score_band": "medium",
            "sigma_band": "medium",
            "attention_band": "medium",
        }
        auxiliary_thresholds: list[float] = []

        score_band = _assign_band(confidence, self.score_band_edges)
        components["score_band"] = score_band
        score_threshold = self.score_band_thresholds.get((normalized_regime, score_band, float(coverage)))
        if score_threshold is None:
            score_threshold = self.global_score_band_thresholds.get((score_band, float(coverage)))
        if score_threshold is not None:
            score_threshold = float(score_threshold)
            components["score_threshold"] = score_threshold
            auxiliary_thresholds.append(score_threshold)

        sigma_band = _assign_band(sigma, self.sigma_band_edges)
        components["sigma_band"] = sigma_band
        sigma_threshold = self.sigma_band_thresholds.get((normalized_regime, sigma_band, float(coverage)))
        if sigma_threshold is None:
            sigma_threshold = self.global_sigma_band_thresholds.get((sigma_band, float(coverage)))
        if sigma_threshold is not None:
            sigma_threshold = float(sigma_threshold)
            components["sigma_threshold"] = sigma_threshold
            auxiliary_thresholds.append(sigma_threshold)

        attention_band = _assign_band(message_volume, self.attention_band_edges)
        components["attention_band"] = attention_band
        attention_threshold = self.attention_band_thresholds.get(
            (normalized_regime, attention_band, float(coverage))
        )
        if attention_threshold is None:
            attention_threshold = self.global_attention_band_thresholds.get(
                (attention_band, float(coverage))
            )
        if attention_threshold is not None:
            attention_threshold = float(attention_threshold)
            components["attention_threshold"] = attention_threshold
            auxiliary_thresholds.append(attention_threshold)

        if auxiliary_thresholds:
            combined_threshold = 0.5 * base_threshold + 0.5 * (
                sum(auxiliary_thresholds) / len(auxiliary_thresholds)
            )
        else:
            combined_threshold = base_threshold
        components["combined_threshold"] = float(combined_threshold)
        return float(combined_threshold), components

    def predict_interval(
        self,
        output: dict[str, Any],
        regime: str = "medium_surprise",
        coverage: float = 0.90,
        metadata: dict[str, float] | list[float] | tuple[float, ...] | None = None,
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
        mu = _to_float(output["mu"])
        log_sigma = _to_float(output["log_sigma"])
        threshold, _details = self._combined_threshold(
            normalized_regime=normalized_regime,
            coverage=coverage,
            output=output,
            metadata=metadata,
        )
        adjusted_half_width = threshold * math.exp(log_sigma)

        return mu - adjusted_half_width, mu + adjusted_half_width

    def interval_diagnostics(
        self,
        output: dict[str, Any],
        regime: str = "medium_surprise",
        coverage: float = 0.90,
        metadata: dict[str, float] | list[float] | tuple[float, ...] | None = None,
    ) -> dict[str, float | str]:
        """Return the threshold components used for an interval prediction."""

        normalized_regime = _normalize_regime(regime)
        threshold, details = self._combined_threshold(
            normalized_regime=normalized_regime,
            coverage=coverage,
            output=output,
            metadata=metadata,
        )
        sigma = max(math.exp(_to_float(output["log_sigma"])), 1e-6)
        details["effective_half_width"] = float(threshold * sigma)
        details["coverage"] = float(coverage)
        details["regime"] = normalized_regime
        return details

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
            metadata=None,
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
