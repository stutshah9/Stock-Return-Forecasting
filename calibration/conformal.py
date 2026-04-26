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
    """Fit regime thresholds from pre-calibration event features."""

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
        return float(np.quantile(scores_array, clamped_level, interpolation="higher"))


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


def _confidence_score(output: dict[str, Any], metadata: Any) -> float:
    """Extract a confidence score, preferring the model score and blending auxiliaries."""

    model_score = None
    if "introspective_score" in output:
        model_score = min(max(_to_float(output.get("introspective_score", 0.5)), 0.0), 1.0)

    metadata_score = None
    if isinstance(metadata, dict) and "explanation_confidence" in metadata:
        metadata_score = min(max(_to_float(metadata["explanation_confidence"]), 0.0), 1.0)

    if model_score is not None and metadata_score is not None:
        return float(0.7 * model_score + 0.3 * metadata_score)
    if metadata_score is not None:
        return float(metadata_score)
    if model_score is not None:
        return float(model_score)
    return 0.5


def _interval_bounds(
    output: dict[str, Any],
    coverage: float | None = None,
) -> tuple[float, float]:
    """Read an ordered prediction interval from a model output dictionary."""

    if coverage is not None and isinstance(output.get("base_intervals"), dict):
        base_intervals = output["base_intervals"]
        interval = base_intervals.get(float(coverage), base_intervals.get(str(float(coverage))))
        if interval is not None:
            lower = _to_float(interval["lower"])
            upper = _to_float(interval["upper"])
            if upper < lower:
                lower, upper = upper, lower
            return float(lower), float(upper)

    if "q_low" in output and "q_high" in output:
        lower = _to_float(output["q_low"])
        upper = _to_float(output["q_high"])
        if upper < lower:
            lower, upper = upper, lower
        return float(lower), float(upper)

    mu = _to_float(output["mu"])
    log_sigma = _to_float(output.get("log_sigma", 0.0))
    sigma = math.exp(log_sigma)

    # Use coverage-appropriate z-score for Gaussian intervals
    z_scores = {0.80: 1.28, 0.90: 1.645, 0.95: 1.96}
    z = z_scores.get(float(coverage), 1.645) if coverage is not None else 1.0

    half_width = z * sigma
    return float(mu - half_width), float(mu + half_width)


def assign_regime(
    sue: float,
    implied_vol: float = 0.0,
    low_thresh: float = 0.5,
    high_thresh: float = 1.5,
    vol_thresh: float = 0.30,
    thresholds: dict[str, float] | None = None,
) -> str:
    """Assign an event to a conformal regime using surprise and volatility."""

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
        use_attention_conditioning: bool = True,
        use_explanation_adjustment: bool = False,
    ) -> None:
        self.coverage_levels = coverage_levels
        self.minimum_bucket_size = max(int(minimum_bucket_size), 8)
        self.use_attention_conditioning = bool(use_attention_conditioning)
        self.thresholds: dict[tuple[str, float], float] = {}
        self.attention_band_thresholds: dict[tuple[str, str, float], float] = {}
        self.global_attention_band_thresholds: dict[tuple[str, float], float] = {}
        self.attention_band_edges: dict[str, float] | None = None

    def calibrate(
        self,
        cal_outputs: list[dict[str, Any]] | None = None,
        cal_labels: list[float] | None = None,
        cal_regimes: list[str] | None = None,
        cal_metadata: list[dict[str, float] | list[float] | tuple[float, ...] | None] | None = None,
        **legacy_kwargs: Any,
    ) -> None:
        """Fit regime-specific nonconformity thresholds from calibration data."""

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
        self.attention_band_thresholds.clear()
        self.global_attention_band_thresholds.clear()

        regime_scores: dict[tuple[str, float], list[float]] = {}
        calibration_rows: list[dict[str, float | str]] = []
        attention_values: list[float] = []

        for output, label, regime, metadata in zip(
            cal_outputs,
            cal_labels,
            cal_regimes,
            cal_metadata,
        ):
            normalized_regime = _normalize_regime(regime)
            message_volume = _message_volume(metadata)
            if math.isfinite(message_volume):
                attention_values.append(float(message_volume))
            for coverage in self.coverage_levels:
                q_low, q_high = _interval_bounds(output, coverage=float(coverage))
                nonconformity = max(q_low - float(label), float(label) - q_high)
                regime_scores.setdefault((normalized_regime, float(coverage)), []).append(nonconformity)
                calibration_rows.append(
                    {
                        "regime": normalized_regime,
                        "coverage": float(coverage),
                        "nonconformity": float(nonconformity),
                        "message_volume": float(message_volume),
                    }
                )

        for (regime, coverage), scores in regime_scores.items():
            n = len(scores)
            if n == 0:
                continue
            quantile_level = coverage * (1.0 + 1.0 / n)
            self.thresholds[(regime, float(coverage))] = _conformal_quantile(scores, quantile_level)

        self.attention_band_edges = _fit_band_edges(attention_values, 0.50, 0.85)
        self._fit_auxiliary_thresholds(
            calibration_rows=calibration_rows,
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
        if band_edges is None:
            return

        band_scores_by_regime: dict[tuple[str, str], list[float]] = {}
        global_band_scores: dict[str, list[float]] = {}
        for row in calibration_rows:
            band = _assign_band(float(row[value_key]), band_edges)
            nonconformity = float(row["nonconformity"])
            regime = str(row["regime"])
            coverage = float(row["coverage"])
            band_scores_by_regime.setdefault((regime, band, coverage), []).append(nonconformity)
            global_band_scores.setdefault((band, coverage), []).append(nonconformity)

        for (regime, band, coverage), scores in band_scores_by_regime.items():
            if len(scores) < self.minimum_bucket_size:
                continue
            quantile_level = coverage * (1.0 + 1.0 / len(scores))
            regime_store[(regime, band, float(coverage))] = _conformal_quantile(
                scores,
                quantile_level,
            )

        for (band, coverage), scores in global_band_scores.items():
            if len(scores) < self.minimum_bucket_size:
                continue
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
        threshold_key = (normalized_regime, float(coverage))
        if threshold_key not in self.thresholds:
            raise KeyError(
                f"Missing conformal threshold for regime={normalized_regime}, "
                f"coverage={coverage}."
            )

        base_threshold = float(self.thresholds[threshold_key])
        message_volume = _message_volume(metadata)

        details: dict[str, float | str] = {
            "base_regime_threshold": base_threshold,
            "observable_threshold": base_threshold,
            "attention_band": "medium",
            "attention_source": "regime_only",
        }

        attention_band = _assign_band(message_volume, self.attention_band_edges)
        details["attention_band"] = attention_band
        regime_attention_threshold = self.attention_band_thresholds.get(
            (normalized_regime, attention_band, float(coverage))
        )
        global_attention_threshold = self.global_attention_band_thresholds.get(
            (attention_band, float(coverage))
        )
        if regime_attention_threshold is not None:
            regime_attention_threshold = float(regime_attention_threshold)
            details["attention_threshold"] = regime_attention_threshold
        elif global_attention_threshold is not None:
            global_attention_threshold = float(global_attention_threshold)
            details["attention_threshold"] = global_attention_threshold

        observable_threshold = base_threshold
        if self.use_attention_conditioning:
            if regime_attention_threshold is not None:
                observable_threshold = max(base_threshold, regime_attention_threshold)
                details["attention_source"] = "regime_attention"
            elif global_attention_threshold is not None:
                observable_threshold = max(base_threshold, global_attention_threshold)
                details["attention_source"] = "global_attention_fallback"
        observable_threshold = max(float(observable_threshold), 0.0)
        details["observable_threshold"] = float(observable_threshold)
        details["combined_threshold"] = float(observable_threshold)
        return float(observable_threshold), details

    def predict_interval(
        self,
        output: dict[str, Any],
        regime: str = "medium_surprise",
        coverage: float = 0.90,
        metadata: dict[str, float] | list[float] | tuple[float, ...] | None = None,
    ) -> tuple[float, float]:
        normalized_regime = _normalize_regime(regime)
        base_lower, base_upper = _interval_bounds(output, coverage=float(coverage))
        threshold, _details = self._combined_threshold(
            normalized_regime=normalized_regime,
            coverage=coverage,
            output=output,
            metadata=metadata,
        )
        return base_lower - threshold, base_upper + threshold

    def interval_diagnostics(
        self,
        output: dict[str, Any],
        regime: str = "medium_surprise",
        coverage: float = 0.90,
        metadata: dict[str, float] | list[float] | tuple[float, ...] | None = None,
    ) -> dict[str, float | str]:
        normalized_regime = _normalize_regime(regime)
        threshold, details = self._combined_threshold(
            normalized_regime=normalized_regime,
            coverage=coverage,
            output=output,
            metadata=metadata,
        )
        base_lower, base_upper = _interval_bounds(output, coverage=float(coverage))
        details["base_interval_width"] = float(base_upper - base_lower)
        details["effective_interval_width"] = float((base_upper - base_lower) + 2.0 * threshold)
        details["coverage"] = float(coverage)
        details["regime"] = normalized_regime
        return details

    def selective_predict(
        self,
        output: dict[str, Any],
        regime: str = "medium_surprise",
        coverage: float = 0.90,
        min_score: float = 0.6,
        metadata: dict[str, float] | list[float] | tuple[float, ...] | None = None,
    ) -> tuple[float, float] | None:
        confidence = _confidence_score(output, metadata)
        if confidence < min_score:
            return None

        lower, upper = self.predict_interval(
            output=output,
            regime=regime,
            coverage=coverage,
            metadata=metadata,
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
