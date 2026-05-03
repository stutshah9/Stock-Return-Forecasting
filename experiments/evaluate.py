"""Evaluation script for multimodal forecasting and conformal intervals."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration.conformal import (
    EventConditionedConformalPredictor,
    assign_regime,
    fit_regime_thresholds,
)
from data.dataset import EarningsDataset
from data.event_utils import (
    filter_events_by_universe,
    summarize_event_coverage,
)
from models.fusion_model import MultimodalForecastModel


EVENTS_CACHE_PATH = PROJECT_ROOT / "data" / "events_cache.pt"
FEATURE_STATS_PATH = PROJECT_ROOT / "experiments" / "feature_stats.json"
REGIME_THRESHOLDS_PATH = PROJECT_ROOT / "experiments" / "regime_thresholds.json"
FINANCIALS_PATH = PROJECT_ROOT / "data" / "financials.csv"


def _resolve_path(path_str: str) -> Path:
    """Resolve a path relative to the project root when needed."""

    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load configuration values from YAML."""

    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def _select_device(config: dict[str, Any]) -> torch.device:
    """Select the requested evaluation device with sensible GPU fallbacks."""

    requested = str(config.get("training", {}).get("device", "auto")).strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("training.device is set to 'cuda' but CUDA is unavailable.")
        return torch.device("cuda")
    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise ValueError("training.device is set to 'mps' but MPS is unavailable.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("training.device must be one of: auto, cuda, mps, cpu.")


def _log_device_diagnostics(device: torch.device, model: MultimodalForecastModel) -> None:
    """Log device diagnostics and trigger a small warmup allocation."""

    first_parameter = next(model.parameters(), None)
    parameter_device = str(first_parameter.device) if first_parameter is not None else "unknown"
    print(f"Model parameter device: {parameter_device}")

    if device.type != "cuda":
        return

    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    print(f"CUDA current device index: {torch.cuda.current_device()}")

    warmup_tensor = torch.empty((1024, 1024), device=device)
    warmup_tensor.zero_()
    torch.cuda.synchronize(device)

    allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
    print(
        f"CUDA memory after warmup: allocated={allocated_mb:.2f} MB, "
        f"reserved={reserved_mb:.2f} MB"
    )
    del warmup_tensor


def _load_cached_events() -> list[dict[str, Any]]:
    """Load the precomputed offline event cache from disk."""

    if not EVENTS_CACHE_PATH.is_file():
        raise FileNotFoundError(
            "data/events_cache.pt not found. "
            "Run: python3 data/build_cache.py first."
        )
    events = torch.load(EVENTS_CACHE_PATH, map_location="cpu")
    if not isinstance(events, list):
        raise ValueError("data/events_cache.pt must contain a list of event dictionaries.")
    return events


def _load_feature_stats() -> dict[str, list[float]]:
    """Load feature normalization statistics produced during training."""

    if not FEATURE_STATS_PATH.is_file():
        raise FileNotFoundError(
            "experiments/feature_stats.json not found. "
            "Run: python3 experiments/train.py first."
        )
    with FEATURE_STATS_PATH.open("r", encoding="utf-8") as stats_file:
        stats = json.load(stats_file)
    if not isinstance(stats, dict) or "mean" not in stats or "std" not in stats:
        raise ValueError("experiments/feature_stats.json is malformed.")
    return stats


def _load_financial_lookup() -> dict[tuple[str, str], dict[str, float]]:
    """Load per-event earnings metadata for frontend inspection exports."""

    if not FINANCIALS_PATH.is_file():
        return {}

    financials = pd.read_csv(FINANCIALS_PATH)
    required_columns = {"ticker", "date"}
    if not required_columns.issubset(financials.columns):
        return {}

    financials["ticker"] = financials["ticker"].astype(str).str.upper()
    financials["date"] = pd.to_datetime(financials["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    financials = financials.dropna(subset=["date"])

    lookup: dict[tuple[str, str], dict[str, float]] = {}
    for _, row in financials.iterrows():
        key = (str(row["ticker"]).upper(), str(row["date"]))
        lookup[key] = {
            "estimated_earnings": _safe_numeric_row_value(row, "estimated_earnings"),
            "actual_earnings": _safe_numeric_row_value(row, "actual_earnings"),
            "earnings_surprise": _safe_numeric_row_value(row, "earnings_surprise"),
        }
    return lookup


def _safe_numeric_row_value(row: pd.Series, column_name: str) -> float:
    """Read a possibly-missing numeric column from a pandas row as float."""

    if column_name not in row.index:
        return float("nan")
    value = pd.to_numeric(pd.Series([row[column_name]]), errors="coerce").iloc[0]
    if pd.isna(value):
        return float("nan")
    return float(value)


def _split_events_by_year(
    events: list[dict[str, Any]],
    split_config: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Split cached events into train/validation/calibration/test partitions."""

    sorted_events = sorted(events, key=lambda event: (int(event["year"]), str(event["date"])))
    validation_years = {int(year) for year in split_config.get("validation_years", [2023])}
    calibration_years = {int(year) for year in split_config.get("calibration_years", [2024])}
    test_years = {int(year) for year in split_config.get("test_years", [2025])}
    reserved_years = validation_years | calibration_years | test_years

    train_events = [event for event in sorted_events if int(event["year"]) not in reserved_years]
    val_events = [event for event in sorted_events if int(event["year"]) in validation_years]
    cal_events = [event for event in sorted_events if int(event["year"]) in calibration_years]
    test_events = [event for event in sorted_events if int(event["year"]) in test_years]
    if not train_events or not val_events or not cal_events or not test_events:
        raise ValueError(
            "Year-based cache split produced an empty partition. "
            f"Expected validation_years={sorted(validation_years)}, "
            f"calibration_years={sorted(calibration_years)}, "
            f"test_years={sorted(test_years)}."
        )
    return train_events, val_events, cal_events, test_events


def _assert_disjoint_splits(
    train_events: list[dict[str, Any]],
    val_events: list[dict[str, Any]],
    cal_events: list[dict[str, Any]],
    test_events: list[dict[str, Any]],
) -> None:
    """Ensure train/validation/calibration/test events are strictly disjoint."""

    def _keys(events: list[dict[str, Any]]) -> set[tuple[str, str]]:
        return {
            (str(event.get("ticker", "")).upper(), str(event.get("date", "")))
            for event in events
        }

    train_keys = _keys(train_events)
    val_keys = _keys(val_events)
    cal_keys = _keys(cal_events)
    test_keys = _keys(test_events)
    if train_keys & val_keys:
        raise ValueError("Training and validation splits overlap.")
    if train_keys & cal_keys:
        raise ValueError("Training and calibration splits overlap.")
    if train_keys & test_keys:
        raise ValueError("Training and test splits overlap.")
    if val_keys & cal_keys:
        raise ValueError("Validation and calibration splits overlap.")
    if val_keys & test_keys:
        raise ValueError("Validation and test splits overlap.")
    if cal_keys & test_keys:
        raise ValueError("Calibration and test splits overlap.")


def _load_regime_thresholds(config: dict[str, Any], historical_events: list[dict[str, Any]]) -> dict[str, float]:
    """Load fitted regime thresholds from disk, or fit them from history as fallback."""

    if REGIME_THRESHOLDS_PATH.is_file():
        with REGIME_THRESHOLDS_PATH.open("r", encoding="utf-8") as thresholds_file:
            loaded = json.load(thresholds_file)
        if isinstance(loaded, dict):
            return {
                "low_thresh": float(loaded.get("low_thresh", 0.5)),
                "high_thresh": float(loaded.get("high_thresh", 1.5)),
                "vol_thresh": float(loaded.get("vol_thresh", 0.30)),
            }

    regime_fit_config = config.get("calibration", {}).get("regime_fit_quantiles", {})
    return fit_regime_thresholds(
        historical_events,
        low_quantile=float(regime_fit_config.get("low_surprise", 0.60)),
        high_quantile=float(regime_fit_config.get("high_surprise", 0.90)),
        vol_quantile=float(regime_fit_config.get("high_vol", 0.60)),
    )


def _feature_triplet(event: dict[str, Any]) -> list[float]:
    """Extract a three-value financial feature list from a cached event."""

    raw_features = event.get("features", [])
    if isinstance(raw_features, torch.Tensor):
        values = [float(value) for value in raw_features.detach().cpu().view(-1).tolist()[:3]]
        while len(values) < 3:
            values.append(0.0)
        return values
    if isinstance(raw_features, dict):
        return [
            float(raw_features.get("sue", 0.0)),
            float(raw_features.get("momentum", 0.0)),
            float(raw_features.get("implied_vol", 0.0)),
        ]

    values: list[float] = []
    for feature_value in list(raw_features)[:3]:
        values.append(float(feature_value))
    while len(values) < 3:
        values.append(0.0)
    return values


def _sentiment_pair(event: dict[str, Any]) -> list[float]:
    """Extract a two-value cached sentiment feature list from an event."""

    raw_sentiment = event.get("sentiment", event.get("sentiment_features", []))
    if isinstance(raw_sentiment, torch.Tensor):
        values = [float(value) for value in raw_sentiment.detach().cpu().view(-1).tolist()[:2]]
        while len(values) < 2:
            values.append(0.0)
        return values

    values: list[float] = []
    for sentiment_value in list(raw_sentiment)[:2]:
        values.append(float(sentiment_value))
    while len(values) < 2:
        values.append(0.0)
    return values


def _event_sue(event: dict[str, Any]) -> float:
    """Extract the cached SUE value used for regime assignment."""

    raw_features = event.get("raw_features")
    if isinstance(raw_features, torch.Tensor):
        return float(raw_features.detach().cpu().view(-1)[0].item())
    if isinstance(raw_features, (list, tuple)) and raw_features:
        return float(raw_features[0])
    return float(_feature_triplet(event)[0])


def _event_implied_vol(event: dict[str, Any]) -> float:
    """Extract the cached implied-volatility feature used for regime assignment."""

    raw_features = event.get("raw_features")
    if isinstance(raw_features, torch.Tensor):
        return float(raw_features.detach().cpu().view(-1)[2].item())
    if isinstance(raw_features, (list, tuple)) and len(raw_features) >= 3:
        return float(raw_features[2])
    return float(_feature_triplet(event)[2])


def _event_calibration_metadata(event: dict[str, Any]) -> dict[str, float]:
    """Extract observable event metadata used for conditional conformal calibration."""

    avg_sentiment, message_volume = _sentiment_pair(event)
    return {
        "avg_sentiment": float(avg_sentiment),
        "message_volume": float(message_volume),
        "sue_abs": float(abs(_event_sue(event))),
        "implied_vol": float(_event_implied_vol(event)),
    }


def _explanation_confidence_from_disagreement(
    disagreement: float,
    center: float,
    scale: float,
) -> float:
    """Map modality disagreement to a bounded confidence score."""

    normalized = (float(disagreement) - float(center)) / max(float(scale), 1e-6)
    return float(1.0 / (1.0 + math.exp(normalized)))


def _build_explanation_metadata(
    events: list[dict[str, Any]],
    full_outputs: list[dict[str, float]],
    text_outputs: list[dict[str, float]],
    financial_outputs: list[dict[str, float]],
    sentiment_outputs: list[dict[str, float]],
    reference_center: float | None = None,
    reference_scale: float | None = None,
) -> tuple[list[dict[str, float]], float, float]:
    """Build metadata with learned confidence plus disagreement as an auxiliary cue."""

    if not (
        len(events)
        == len(full_outputs)
        == len(text_outputs)
        == len(financial_outputs)
        == len(sentiment_outputs)
    ):
        raise ValueError("Explanation metadata inputs must have matching lengths.")

    def _interval_midpoint(output: dict[str, float]) -> float:
        lower, upper = _output_interval_bounds(output)
        return 0.5 * (float(lower) + float(upper))

    disagreements: list[float] = []
    for text_output, financial_output, sentiment_output in zip(
        text_outputs,
        financial_outputs,
        sentiment_outputs,
    ):
        mu_values = [
            _interval_midpoint(text_output),
            _interval_midpoint(financial_output),
            _interval_midpoint(sentiment_output),
        ]
        disagreements.append(float(np.std(np.asarray(mu_values, dtype=float), ddof=0)))

    if reference_center is None:
        center = float(np.median(np.asarray(disagreements, dtype=float))) if disagreements else 0.0
    else:
        center = float(reference_center)
    if reference_scale is None:
        scale = float(np.std(np.asarray(disagreements, dtype=float), ddof=0)) if disagreements else 1.0
    else:
        scale = float(reference_scale)
    scale = max(scale, 1e-6)

    metadata_rows: list[dict[str, float]] = []
    for event, disagreement, full_output in zip(events, disagreements, full_outputs):
        base_metadata = _event_calibration_metadata(event)
        disagreement_confidence = _explanation_confidence_from_disagreement(
            disagreement=disagreement,
            center=center,
            scale=scale,
        )
        model_confidence = float(full_output.get("introspective_score", 0.5))
        metadata_rows.append(
            {
                **base_metadata,
                "modality_disagreement": float(disagreement),
                "disagreement_confidence": float(disagreement_confidence),
                "model_confidence": model_confidence,
                "explanation_confidence": float(
                    min(max(0.7 * model_confidence + 0.3 * disagreement_confidence, 0.0), 1.0)
                ),
            }
        )
    return metadata_rows, center, scale


def _to_float(value: Any) -> float:
    """Convert tensor-like or scalar-like values to float."""

    try:
        return float(value.item())
    except AttributeError:
        return float(value)


def _point_prediction_for_method(
    output: dict[str, Any],
    method: str,
) -> float:
    """Select the point forecast used for a given evaluation method."""

    if method in {
        "naive_conformal",
        "event_conditioned_conformal",
        "normalized_conformal_width",
        "normalized_conformal_modality",
        "normalized_conformal_combined",
        "ours",
    }:
        return _to_float(output.get("point_mu", output["mu"]))
    return _to_float(output.get("point_mu", output["mu"]))


def _prediction_variance_proxy(output: dict[str, Any]) -> float:
    """Approximate predictive variance from the default interval half-width."""

    lower, upper = _output_interval_bounds(output)
    half_width = max((float(upper) - float(lower)) / 2.0, 1e-6)
    return float(half_width ** 2)

def _output_interval_bounds(output: dict[str, Any]) -> tuple[float, float]:
    """Read an ordered interval from serialized model outputs."""

    if "q_low" in output and "q_high" in output:
        lower = float(output["q_low"])
        upper = float(output["q_high"])
        if upper < lower:
            lower, upper = upper, lower
        return lower, upper

    mu = _to_float(output["mu"])
    sigma = math.exp(_to_float(output.get("log_sigma", 0.0)))
    return float(mu - sigma), float(mu + sigma)


def _output_interval_bounds_for_coverage(
    output: dict[str, Any],
    coverage: float,
) -> tuple[float, float]:
    """Read the base interval matching a target nominal coverage."""

    base_intervals = output.get("base_intervals")
    if isinstance(base_intervals, dict):
        interval = base_intervals.get(float(coverage), base_intervals.get(str(float(coverage))))
        if interval is not None:
            lower = float(interval["lower"])
            upper = float(interval["upper"])
            if upper < lower:
                lower, upper = upper, lower
            return lower, upper
    return _output_interval_bounds(output)


def _conformal_quantile(scores: list[float], quantile_level: float) -> float:
    """Compute a conservative quantile for conformal calibration."""

    clamped_level = min(max(quantile_level, 0.0), 1.0)
    array = np.asarray(scores, dtype=float)
    try:
        return float(np.quantile(array, clamped_level, method="higher"))
    except TypeError:
        return float(np.quantile(array, clamped_level, interpolation="higher"))


def _target_quantile(scores: list[float], quantile_level: float) -> float:
    """Compute an empirical target quantile for nominal-coverage tuning."""

    clamped_level = min(max(quantile_level, 0.0), 1.0)
    array = np.asarray(scores, dtype=float)
    try:
        return float(np.quantile(array, clamped_level, method="linear"))
    except TypeError:
        return float(np.quantile(array, clamped_level, interpolation="linear"))


def _adaptive_tuning_target(coverage: float, sample_size: int) -> float:
    """Use a small finite-sample correction when optimizing adaptive intervals."""

    n = max(int(sample_size), 1)
    standard_error = math.sqrt(float(coverage) * (1.0 - float(coverage)) / n)
    margin = 1.5 * standard_error
    return float(min(max(float(coverage) - margin, 0.0), 1.0))


def _build_global_thresholds(
    cal_outputs: list[dict[str, Any]],
    cal_labels: list[float],
    coverage_levels: list[float],
) -> dict[float, float]:
    """Build pooled CQR nonconformity thresholds for global conformal."""

    thresholds: dict[float, float] = {}
    for coverage in coverage_levels:
        scores: list[float] = []
        for output, label in zip(cal_outputs, cal_labels):
            q_low, q_high = _output_interval_bounds_for_coverage(output, float(coverage))
            scores.append(max(q_low - float(label), float(label) - q_high))
        n = len(scores)
        quantile_level = coverage * (1.0 + 1.0 / n)
        thresholds[float(coverage)] = _conformal_quantile(scores, quantile_level)
    return thresholds


class NormalizedConformalPredictor:
    """Calibrate CQR corrections after normalizing by an event difficulty score."""

    H_CLIP_LOW = 0.5
    H_CLIP_HIGH = 2.0

    def __init__(
        self,
        coverage_levels: list[float],
        difficulty: str,
        minimum_difficulty: float = 1e-4,
    ) -> None:
        self.coverage_levels = [float(coverage) for coverage in coverage_levels]
        self.difficulty = difficulty
        self.minimum_difficulty = max(float(minimum_difficulty), 1e-8)
        self.thresholds: dict[float, float] = {}
        # For width / modality_disagreement: h(x) = clip(raw / median(raw_cal), lo, hi).
        self.normalization_scale: dict[float, float] = {}
        # For combined: h(x) = clip(1 + sum_k beta_k * z_k(x), lo, hi).
        self.combined_coefficients: dict[str, float] = {}
        self.combined_feature_stats: dict[str, tuple[float, float]] = {}

    def _raw_signal(
        self,
        name: str,
        output: dict[str, Any],
        coverage: float,
        metadata: dict[str, float] | None,
    ) -> float:
        if name == "width":
            lower, upper = _output_interval_bounds_for_coverage(output, coverage)
            return float(max(upper - lower, 0.0))
        if name == "modality_disagreement":
            if metadata is None:
                return 0.0
            return float(max(metadata.get("modality_disagreement", 0.0), 0.0))
        raise ValueError(f"Unknown raw signal: {name}")

    def _difficulty(
        self,
        output: dict[str, Any],
        coverage: float,
        metadata: dict[str, float] | None,
    ) -> float:
        """Return the clipped, normalized h(x) used to scale the conformal correction."""

        if self.difficulty in {"width", "modality_disagreement"}:
            raw = self._raw_signal(self.difficulty, output, coverage, metadata)
            denom = max(float(self.normalization_scale.get(float(coverage), 0.0)), self.minimum_difficulty)
            h = raw / denom
        elif self.difficulty == "combined":
            if not self.combined_coefficients:
                h = 1.0
            else:
                h = 1.0
                for name in ("width", "modality_disagreement"):
                    mean, std = self.combined_feature_stats.get(name, (0.0, 1.0))
                    std = std if std > 1e-9 else 1.0
                    z = (self._raw_signal(name, output, coverage, metadata) - mean) / std
                    h += float(self.combined_coefficients.get(name, 0.0)) * z
        else:
            raise ValueError(
                "difficulty must be one of: width, modality_disagreement, combined."
            )

        return float(min(max(h, self.H_CLIP_LOW), self.H_CLIP_HIGH))

    def _fit_combined_coefficients(
        self,
        cal_outputs: list[dict[str, Any]],
        cal_labels: list[float],
        cal_metadata: list[dict[str, float]],
    ) -> None:
        """Fit beta in h(x) = 1 + beta . z(features) by OLS of |residual| on z-features."""

        reference_coverage = float(self.coverage_levels[len(self.coverage_levels) // 2])
        feature_names = ("width", "modality_disagreement")
        feature_rows: list[list[float]] = []
        targets: list[float] = []
        for output, label, metadata in zip(cal_outputs, cal_labels, cal_metadata):
            base_lower, base_upper = _output_interval_bounds_for_coverage(output, reference_coverage)
            mu = float(output.get("mu", 0.5 * (base_lower + base_upper)))
            row = [self._raw_signal(name, output, reference_coverage, metadata) for name in feature_names]
            if not all(math.isfinite(value) for value in row):
                continue
            feature_rows.append(row)
            targets.append(float(abs(float(label) - mu)))

        if len(feature_rows) < max(20, 2 * len(feature_names)):
            self.combined_coefficients = {}
            self.combined_feature_stats = {}
            return

        X = np.asarray(feature_rows, dtype=float)
        y = np.asarray(targets, dtype=float)
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        stds_safe = np.where(stds > 1e-9, stds, 1.0)
        Z = (X - means) / stds_safe
        # Center the target so the intercept is absorbed into the constant 1.0 in h(x).
        beta, *_ = np.linalg.lstsq(Z, y - y.mean(), rcond=None)
        # Rescale so that |beta . z| has comparable magnitude to ~1; keep clipping as the safety net.
        scale = float(np.std(Z @ beta))
        if scale > 1e-9:
            beta = beta / scale
        self.combined_coefficients = {name: float(beta[i]) for i, name in enumerate(feature_names)}
        self.combined_feature_stats = {
            name: (float(means[i]), float(stds[i])) for i, name in enumerate(feature_names)
        }

    def calibrate(
        self,
        cal_outputs: list[dict[str, Any]],
        cal_labels: list[float],
        cal_metadata: list[dict[str, float]],
    ) -> None:
        if not (len(cal_outputs) == len(cal_labels) == len(cal_metadata)):
            raise ValueError("Normalized conformal calibration inputs must align.")

        self.thresholds.clear()
        self.normalization_scale.clear()

        if self.difficulty == "combined":
            self._fit_combined_coefficients(cal_outputs, cal_labels, cal_metadata)

        for coverage in self.coverage_levels:
            if self.difficulty in {"width", "modality_disagreement"}:
                raw_signals = [
                    self._raw_signal(self.difficulty, output, coverage, metadata)
                    for output, metadata in zip(cal_outputs, cal_metadata)
                ]
                positive_signals = [
                    value for value in raw_signals if math.isfinite(value) and value > 0.0
                ]
                scale = (
                    float(np.median(np.asarray(positive_signals, dtype=float)))
                    if positive_signals
                    else self.minimum_difficulty
                )
                self.normalization_scale[float(coverage)] = max(scale, self.minimum_difficulty)

            scores: list[float] = []
            for output, label, metadata in zip(cal_outputs, cal_labels, cal_metadata):
                base_lower, base_upper = _output_interval_bounds_for_coverage(output, coverage)
                nonconformity = max(base_lower - float(label), float(label) - base_upper)
                scores.append(
                    float(nonconformity)
                    / self._difficulty(output, coverage, metadata)
                )
            # The normalized rows are calibration-design ablations for hitting
            # 80/90/95 as closely as possible, so use the empirical target
            # quantile instead of the conservative finite-sample conformal
            # quantile used by the naive/global baseline.
            target_level = _adaptive_tuning_target(float(coverage), len(scores))
            self.thresholds[float(coverage)] = _target_quantile(scores, target_level)

    def predict_interval(
        self,
        output: dict[str, Any],
        coverage: float,
        metadata: dict[str, float] | None = None,
    ) -> tuple[float, float]:
        coverage = float(coverage)
        if coverage not in self.thresholds:
            raise KeyError(f"Missing normalized conformal threshold for coverage={coverage}.")
        base_lower, base_upper = _output_interval_bounds_for_coverage(output, coverage)
        correction = self.thresholds[coverage] * self._difficulty(output, coverage, metadata)
        base_width = max(float(base_upper - base_lower), self.minimum_difficulty)
        correction = max(float(correction), -0.49 * base_width)
        return float(base_lower - correction), float(base_upper + correction)


def _mean_explanation_confidence(metadata: list[dict[str, float]] | None) -> float:
    """Summarize explanation confidence for normalization."""

    if not metadata:
        return 1.0

    values: list[float] = []
    for row in metadata:
        value = float(row.get("explanation_confidence", float("nan")))
        if math.isfinite(value):
            values.append(value)
    if not values:
        return 1.0
    return max(float(np.mean(values)), 1e-6)


def _fallback_threshold(
    predictor: EventConditionedConformalPredictor,
    coverage: float,
) -> float:
    """Return a fallback threshold when a regime-specific threshold is missing."""

    candidate_values = [
        threshold
        for (regime_name, cov), threshold in predictor.thresholds.items()
        if float(cov) == float(coverage)
    ]
    if not candidate_values:
        raise KeyError(f"No thresholds available for coverage={coverage}.")
    return float(sum(candidate_values) / len(candidate_values))


def _prepare_modal_inputs(
    events: list[dict[str, Any]],
    dataset: EarningsDataset,
    method: str,
    device: torch.device,
    regime_thresholds: dict[str, float],
) -> tuple[list[str], Tensor, Tensor, Tensor, list[str], list[str]]:
    """Prepare modality inputs and metadata for a given evaluation method."""

    def _normalized_zero_financial_tensor() -> torch.Tensor:
        zero_raw = torch.zeros((len(dataset), 3), dtype=torch.float32, device=device)
        if not dataset.stats:
            return zero_raw
        mean = torch.tensor(dataset.stats["mean"], dtype=torch.float32, device=device)
        std = torch.tensor(dataset.stats["std"], dtype=torch.float32, device=device)
        return (zero_raw - mean) / std

    samples = [dataset[index] for index in range(len(dataset))]
    transcripts = [str(sample.get("transcript", "")) for sample in samples]
    tickers = [str(event.get("ticker", "")) for event in events]
    regimes = [
        assign_regime(
            sue=_event_sue(event),
            implied_vol=_event_implied_vol(event),
            thresholds=regime_thresholds,
        )
        for event in events
    ]
    labels = torch.stack([sample["label"].float() for sample in samples], dim=0).to(device)
    financial = torch.stack([sample["features"].float() for sample in samples], dim=0).to(device)
    sentiment = torch.stack([sample["sentiment"].float() for sample in samples], dim=0).to(device)

    if method == "text_only":
        financial = _normalized_zero_financial_tensor()
        sentiment = torch.zeros_like(sentiment)
    elif method == "financial_only":
        transcripts = ["" for _ in transcripts]
        sentiment = torch.zeros_like(sentiment)
    elif method == "sentiment_only":
        transcripts = ["" for _ in transcripts]
        financial = _normalized_zero_financial_tensor()
        if os.environ.get("DEBUG_SENTIMENT_ONLY_TRANSCRIPTS") == "1":
            print("sentiment_only transcripts first5:", [repr(value) for value in transcripts[:5]])
            print(
                "sentiment_only transcripts all_empty:",
                all(value == "" for value in transcripts),
            )

    return transcripts, financial, sentiment, labels, regimes, tickers


def _compute_model_outputs(
    model: MultimodalForecastModel,
    events: list[dict[str, Any]],
    dataset: EarningsDataset,
    method: str,
    device: torch.device,
    regime_thresholds: dict[str, float],
) -> tuple[list[dict[str, float]], list[float], list[str], list[str]]:
    """Run a model-based evaluation variant and serialize its outputs."""

    transcripts, financial, sentiment, labels, regimes, tickers = _prepare_modal_inputs(
        events,
        dataset,
        method,
        device,
        regime_thresholds,
    )
    model.eval()
    with torch.no_grad():
        batch_outputs = model(
            transcripts=transcripts,
            financial=financial,
            sentiment=sentiment,
            return_explanations=True,
        )

    outputs: list[dict[str, float]] = []
    quantile_levels = [float(value) for value in batch_outputs["quantile_levels"].cpu().tolist()]
    quantile_predictions = batch_outputs["quantile_predictions"].cpu().tolist()
    base_intervals_raw = batch_outputs.get("base_intervals", {})
    attention_values = batch_outputs.get("attention_stability")
    interval_confidence_values = batch_outputs.get("interval_confidence")
    modality_values = batch_outputs.get("modality_consistency")
    attention_list = (
        attention_values.cpu().tolist()
        if attention_values is not None
        else [0.0 for _ in batch_outputs["q_low"].cpu().tolist()]
    )
    interval_confidence_list = (
        interval_confidence_values.cpu().tolist()
        if interval_confidence_values is not None
        else [0.0 for _ in batch_outputs["q_low"].cpu().tolist()]
    )
    modality_list = (
        modality_values.cpu().tolist()
        if modality_values is not None
        else [0.0 for _ in batch_outputs["q_low"].cpu().tolist()]
    )
    explanations = batch_outputs.get("explanations")
    if explanations is None:
        explanations = ["" for _ in batch_outputs["q_low"].cpu().tolist()]

    for index, (
        q_low,
        q_high,
        mu,
        point_mu,
        quantile_median,
        score,
        attention,
        interval_confidence,
        modality_consistency,
    ) in enumerate(zip(
        batch_outputs["q_low"].cpu().tolist(),
        batch_outputs["q_high"].cpu().tolist(),
        batch_outputs["mu"].cpu().tolist(),
        batch_outputs.get("point_mu", batch_outputs["mu"]).cpu().tolist(),
        batch_outputs.get("quantile_median", batch_outputs["mu"]).cpu().tolist(),
        batch_outputs["introspective_score"].cpu().tolist(),
        attention_list,
        interval_confidence_list,
        modality_list,
    )):
        per_output_quantiles = {
            level: float(prediction)
            for level, prediction in zip(quantile_levels, quantile_predictions[index])
        }
        base_intervals = {
            float(coverage): {
                "lower": float(interval_pair[0][index].detach().cpu().item()),
                "upper": float(interval_pair[1][index].detach().cpu().item()),
            }
            for coverage, interval_pair in base_intervals_raw.items()
        }
        outputs.append(
            {
                "quantiles": per_output_quantiles,
                "base_intervals": base_intervals,
                "q_low": float(q_low),
                "q_high": float(q_high),
                "mu": float(mu),
                "point_mu": float(point_mu),
                "quantile_median": float(quantile_median),
                "introspective_score": float(score),
                "attention_stability": float(attention),
                "interval_confidence": float(interval_confidence),
                "variance_confidence": float(interval_confidence),
                "modality_consistency": float(modality_consistency),
                "explanation": str(explanations[index]),
            }
        )
    return outputs, [float(value) for value in labels.cpu().tolist()], regimes, tickers


def _same_ticker_baseline_parameters(
    ticker: str,
    training_events: list[dict[str, Any]],
) -> tuple[float, float]:
    """Estimate same-ticker baseline mean and volatility from training labels."""

    same_ticker_returns = [
        float(event["label"])
        for event in training_events
        if str(event.get("ticker", "")).upper() == ticker.upper()
        and not math.isnan(float(event.get("label", math.nan)))
    ]
    if len(same_ticker_returns) >= 2:
        mu = float(np.mean(same_ticker_returns))
        sigma = float(np.std(same_ticker_returns, ddof=1))
        if sigma > 0.0:
            return mu, sigma
        return mu, 0.02
    return 0.0, 0.02


def _compute_same_ticker_baseline(
    test_events: list[dict[str, Any]],
    training_events: list[dict[str, Any]],
    regime_thresholds: dict[str, float],
) -> tuple[list[dict[str, float]], list[float], list[str], list[str]]:
    """Build a same-ticker historical-mean and volatility baseline."""

    outputs: list[dict[str, float]] = []
    labels: list[float] = []
    regimes: list[str] = []
    tickers: list[str] = []

    for event in test_events:
        ticker = str(event.get("ticker", "")).upper()
        mu, sigma = _same_ticker_baseline_parameters(ticker, training_events)
        outputs.append(
            {
                "base_intervals": {},
                "q_low": float(mu - sigma),
                "q_high": float(mu + sigma),
                "mu": float(mu),
                "introspective_score": 1.0,
                "variance_confidence": 1.0,
                "explanation": (
                    "The baseline forecasts from the same ticker's historical post-earnings "
                    "return mean and volatility rather than the multimodal explanation head."
                ),
            }
        )
        labels.append(float(event.get("label", 0.0)))
        regimes.append(
            assign_regime(
                sue=_event_sue(event),
                implied_vol=_event_implied_vol(event),
                thresholds=regime_thresholds,
            )
        )
        tickers.append(ticker)
    return outputs, labels, regimes, tickers


def _predict_interval_without_adjustment(
    predictor: EventConditionedConformalPredictor,
    output: dict[str, Any],
    regime: str,
    coverage: float,
) -> tuple[float, float]:
    """Return the model's raw quantile interval before conformal expansion."""

    del predictor, regime
    return _output_interval_bounds_for_coverage(output, coverage)


def _predict_interval_naive(
    output: dict[str, Any],
    coverage: float,
    global_thresholds: dict[float, float],
) -> tuple[float, float]:
    """Predict a globally conformalized CQR interval."""

    q_low, q_high = _output_interval_bounds_for_coverage(output, float(coverage))
    threshold = global_thresholds[float(coverage)]
    return q_low - threshold, q_high + threshold


def _predict_interval_confidence_scaled_naive(
    output: dict[str, Any],
    coverage: float,
    global_thresholds: dict[float, float],
    explanation_confidence: float,
    mean_explanation_confidence: float,
) -> tuple[float, float]:
    """Scale the naive conformal threshold using the configured confidence/variance rule."""

    mu = _point_prediction_for_method(output, "ours")
    base_threshold = float(global_thresholds[float(coverage)])
    variance = max(_prediction_variance_proxy(output), 1e-6)
    normalized_confidence = max(float(explanation_confidence), 1e-6) / max(
        float(mean_explanation_confidence),
        1e-6,
    )
    variance_sqrt = math.sqrt(variance)
    rule = os.environ.get("OURS_INTERVAL_RULE", "confidence_only").strip().lower()
    variance_floor = max(float(os.environ.get("OURS_VARIANCE_FLOOR", "0.0025")), 1e-6)
    max_scale = max(float(os.environ.get("OURS_SCALE_MAX", "10.0")), 1e-6)

    if rule == "divide_variance":
        scale = normalized_confidence / variance
    elif rule == "divide_sqrt_variance":
        scale = normalized_confidence / max(variance_sqrt, 1e-6)
    elif rule == "multiply_variance":
        scale = normalized_confidence * variance
    elif rule == "multiply_sqrt_variance":
        scale = normalized_confidence * variance_sqrt
    elif rule == "confidence_only":
        scale = normalized_confidence
    elif rule == "clamped_divide_variance":
        scale = normalized_confidence / max(variance, variance_floor)
        scale = min(scale, max_scale)
    else:
        raise ValueError(
            "OURS_INTERVAL_RULE must be one of: divide_variance, divide_sqrt_variance, "
            "multiply_variance, multiply_sqrt_variance, confidence_only, clamped_divide_variance."
        )

    adjusted_threshold = base_threshold * scale
    return mu - adjusted_threshold, mu + adjusted_threshold


def _predict_interval_normalized(
    normalized_predictor: NormalizedConformalPredictor,
    output: dict[str, Any],
    coverage: float,
    metadata: dict[str, float] | None,
) -> tuple[float, float]:
    """Predict an interval using normalized conformal scores."""

    return normalized_predictor.predict_interval(
        output=output,
        coverage=coverage,
        metadata=metadata,
    )


def _unscaled_interval_for_mode(
    mode: str,
    output: dict[str, Any],
    regime: str,
    coverage: float,
    predictor: EventConditionedConformalPredictor,
    global_thresholds: dict[float, float],
    metadata: dict[str, float] | None,
    mean_explanation_confidence: float,
    normalized_predictors: dict[str, NormalizedConformalPredictor] | None = None,
) -> tuple[float, float]:
    """Return interval bounds before optional calibration-error scale tuning."""

    if mode == "adaptive":
        try:
            return predictor.predict_interval(
                output=output,
                regime=regime,
                coverage=coverage,
                metadata=metadata,
            )
        except KeyError:
            threshold = _fallback_threshold(predictor, coverage)
            base_lower, base_upper = _output_interval_bounds_for_coverage(output, coverage)
            return base_lower - threshold, base_upper + threshold
    if mode == "naive":
        return _predict_interval_naive(output, coverage, global_thresholds)
    if mode == "confidence_scaled_naive":
        explanation_confidence = float(
            (metadata or {}).get(
                "explanation_confidence",
                output.get("introspective_score", 0.0),
            )
        )
        return _predict_interval_confidence_scaled_naive(
            output=output,
            coverage=coverage,
            global_thresholds=global_thresholds,
            explanation_confidence=explanation_confidence,
            mean_explanation_confidence=mean_explanation_confidence,
        )
    if mode in {"normalized_width", "normalized_modality", "normalized_combined"}:
        if normalized_predictors is None or mode not in normalized_predictors:
            raise KeyError(f"Missing normalized conformal predictor for mode={mode}.")
        return _predict_interval_normalized(
            normalized_predictor=normalized_predictors[mode],
            output=output,
            coverage=coverage,
            metadata=metadata,
        )
    return _predict_interval_without_adjustment(
        predictor,
        output,
        regime,
        coverage,
    )


def _apply_interval_scale(
    output: dict[str, Any],
    coverage: float,
    lower: float,
    upper: float,
    scale: float | None,
) -> tuple[float, float]:
    """Scale only the conformal correction around the model's raw interval."""

    if scale is None:
        return lower, upper
    base_lower, base_upper = _output_interval_bounds_for_coverage(output, coverage)
    lower_correction = float(base_lower) - float(lower)
    upper_correction = float(upper) - float(base_upper)
    scaled_lower = float(base_lower) - float(scale) * lower_correction
    scaled_upper = float(base_upper) + float(scale) * upper_correction
    if scaled_upper < scaled_lower:
        midpoint = 0.5 * (scaled_lower + scaled_upper)
        scaled_lower = midpoint
        scaled_upper = midpoint
    return scaled_lower, scaled_upper


def _fit_interval_scales(
    modes: list[str],
    coverage_levels: list[float],
    cal_outputs: list[dict[str, Any]],
    cal_labels: list[float],
    cal_regimes: list[str],
    predictor: EventConditionedConformalPredictor,
    global_thresholds: dict[float, float],
    cal_metadata: list[dict[str, float]],
    normalized_predictors: dict[str, NormalizedConformalPredictor],
) -> dict[tuple[str, float], float]:
    """Tune correction multipliers to make calibration coverage closer to nominal."""

    scales: dict[tuple[str, float], float] = {}
    mean_explanation_confidence = _mean_explanation_confidence(cal_metadata)
    # The grid includes shrinkage and mild expansion. Ties prefer narrower
    # intervals, which matches the coverage-then-width comparison objective.
    scale_grid = np.linspace(0.0, 1.5, 301)

    for mode in modes:
        for coverage in coverage_levels:
            target_coverage = float(coverage)
            candidates: list[tuple[float, float, float]] = []
            unscaled_bounds = [
                _unscaled_interval_for_mode(
                    mode=mode,
                    output=output,
                    regime=regime,
                    coverage=float(coverage),
                    predictor=predictor,
                    global_thresholds=global_thresholds,
                    metadata=metadata,
                    mean_explanation_confidence=mean_explanation_confidence,
                    normalized_predictors=normalized_predictors,
                )
                for output, regime, metadata in zip(
                    cal_outputs,
                    cal_regimes,
                    cal_metadata,
                )
            ]
            for scale in scale_grid:
                hits: list[float] = []
                widths: list[float] = []
                for output, label, (lower, upper) in zip(
                    cal_outputs,
                    cal_labels,
                    unscaled_bounds,
                ):
                    scaled_lower, scaled_upper = _apply_interval_scale(
                        output=output,
                        coverage=float(coverage),
                        lower=lower,
                        upper=upper,
                        scale=float(scale),
                    )
                    hits.append(1.0 if scaled_lower <= float(label) <= scaled_upper else 0.0)
                    widths.append(float(scaled_upper - scaled_lower))
                empirical_coverage = float(np.mean(hits))
                avg_width = float(np.mean(widths))
                candidates.append(
                    (
                        abs(empirical_coverage - target_coverage),
                        avg_width,
                        float(scale),
                    )
                )
            _error, _width, best_scale = min(candidates, key=lambda item: (item[0], item[1]))
            scales[(mode, float(coverage))] = best_scale
    return scales


def _select_interval_modes_on_validation(
    candidate_modes: dict[str, str],
    validation_outputs: list[dict[str, Any]],
    validation_labels: list[float],
    validation_regimes: list[str],
    predictor: EventConditionedConformalPredictor,
    global_thresholds: dict[float, float],
    validation_metadata: list[dict[str, float]],
    normalized_predictors: dict[str, NormalizedConformalPredictor],
    interval_scales: dict[tuple[str, float], float],
    coverage_levels: list[float],
    reference_explanation_confidence: float,
) -> dict[str, str]:
    """Choose adaptive interval modes only when they beat global CQR on validation."""

    naive_row = _metric_row(
        method="naive_conformal",
        outputs=validation_outputs,
        labels=validation_labels,
        regimes=validation_regimes,
        predictor=predictor,
        global_thresholds=global_thresholds,
        mode="naive",
        metadata=validation_metadata,
        reference_explanation_confidence=reference_explanation_confidence,
        normalized_predictors=normalized_predictors,
        interval_scales=interval_scales,
        coverages=coverage_levels,
    )
    naive_error = float(naive_row["calibration_error"])
    naive_width = float(naive_row["avg_width"])

    selected_modes: dict[str, str] = {}
    for method, candidate_mode in candidate_modes.items():
        candidate_row = _metric_row(
            method=method,
            outputs=validation_outputs,
            labels=validation_labels,
            regimes=validation_regimes,
            predictor=predictor,
            global_thresholds=global_thresholds,
            mode=candidate_mode,
            metadata=validation_metadata,
            reference_explanation_confidence=reference_explanation_confidence,
            normalized_predictors=normalized_predictors,
            interval_scales=interval_scales,
            coverages=coverage_levels,
        )
        candidate_error = float(candidate_row["calibration_error"])
        candidate_width = float(candidate_row["avg_width"])

        # Coverage accuracy is the first objective. Width breaks near-ties.
        error_tolerance = 1e-12
        width_tolerance = 1e-12
        improves_error = candidate_error < naive_error - error_tolerance
        ties_error_and_narrows = (
            abs(candidate_error - naive_error) <= error_tolerance
            and candidate_width <= naive_width + width_tolerance
        )
        selected_modes[method] = (
            candidate_mode if improves_error or ties_error_and_narrows else "naive"
        )
    return selected_modes


def _regime_components(regime: str) -> tuple[str, str]:
    """Split a composite regime into surprise and volatility subgroup labels."""

    normalized_regime = str(regime)
    if normalized_regime.endswith("_low_vol"):
        return normalized_regime[: -len("_low_vol")], "low_vol"
    if normalized_regime.endswith("_high_vol"):
        return normalized_regime[: -len("_high_vol")], "high_vol"
    return normalized_regime, "unknown_vol"


def _attention_volume_bands(
    metadata: list[dict[str, float]] | None,
) -> list[str]:
    """Partition message volume into coarse low/medium/high attention bands."""

    if metadata is None or not metadata:
        return []

    values: list[float] = []
    for row in metadata:
        value = float(row.get("message_volume", 0.0))
        if math.isfinite(value):
            values.append(value)

    if len(values) < 3:
        return ["medium_attention" for _ in metadata]

    values_array = np.asarray(values, dtype=float)
    if float(np.max(values_array) - np.min(values_array)) <= 1e-8:
        return ["medium_attention" for _ in metadata]

    low_cut = float(np.quantile(values_array, 0.33))
    high_cut = float(np.quantile(values_array, 0.67))
    if high_cut <= low_cut:
        return ["medium_attention" for _ in metadata]

    bands: list[str] = []
    for row in metadata:
        value = float(row.get("message_volume", 0.0))
        if not math.isfinite(value):
            bands.append("medium_attention")
        elif value < low_cut:
            bands.append("low_attention")
        elif value < high_cut:
            bands.append("medium_attention")
        else:
            bands.append("high_attention")
    return bands


def _metric_row(
    method: str,
    outputs: list[dict[str, float]],
    labels: list[float],
    regimes: list[str],
    predictor: EventConditionedConformalPredictor,
    global_thresholds: dict[float, float],
    mode: str,
    metadata: list[dict[str, float]] | None = None,
    reference_explanation_confidence: float | None = None,
    normalized_predictors: dict[str, NormalizedConformalPredictor] | None = None,
    interval_scales: dict[tuple[str, float], float] | None = None,
    coverages: list[float] | None = None,
) -> dict[str, float | str]:
    """Compute evaluation metrics for one method row."""

    if coverages is None:
        coverages = [0.80, 0.90, 0.95]
    coverages = [float(c) for c in coverages]
    interval_hits: dict[float, list[float]] = {coverage: [] for coverage in coverages}
    interval_widths: dict[float, list[float]] = {coverage: [] for coverage in coverages}

    mu_values = [_point_prediction_for_method(output, method) for output in outputs]
    mae = float(np.mean([abs(mu - y) for mu, y in zip(mu_values, labels)]))
    rmse = float(np.sqrt(np.mean([(mu - y) ** 2 for mu, y in zip(mu_values, labels)])))
    if method == "same_ticker_baseline":
        dir_acc = float(
            np.mean(
                [
                    1.0
                    if (1 if mu >= 0.0 else -1) == (1 if y >= 0.0 else -1)
                    else 0.0
                    for mu, y in zip(mu_values, labels)
                ]
            )
        )
    else:
        dir_acc = float(
            np.mean(
                [
                    1.0 if np.sign(mu) == np.sign(y) else 0.0
                    for mu, y in zip(mu_values, labels)
                ]
            )
        )

    if metadata is None:
        metadata = [{} for _ in outputs]
    mean_explanation_confidence = (
        float(reference_explanation_confidence)
        if reference_explanation_confidence is not None
        else _mean_explanation_confidence(metadata)
    )

    for output, label, regime, event_metadata in zip(outputs, labels, regimes, metadata):
        for coverage in coverages:
            lower, upper = _unscaled_interval_for_mode(
                mode=mode,
                output=output,
                regime=regime,
                coverage=coverage,
                predictor=predictor,
                global_thresholds=global_thresholds,
                metadata=event_metadata,
                mean_explanation_confidence=mean_explanation_confidence,
                normalized_predictors=normalized_predictors,
            )
            lower, upper = _apply_interval_scale(
                output=output,
                coverage=coverage,
                lower=lower,
                upper=upper,
                scale=(interval_scales or {}).get((mode, float(coverage))),
            )

            interval_hits[coverage].append(1.0 if lower <= label <= upper else 0.0)
            interval_widths[coverage].append(upper - lower)

    per_coverage_width = {c: float(np.mean(interval_widths[c])) for c in coverages}
    per_coverage_hit = {c: float(np.mean(interval_hits[c])) for c in coverages}
    per_coverage_cal_error = {c: abs(per_coverage_hit[c] - c) for c in coverages}
    explanation_confidences = [
        float(row.get("explanation_confidence", output.get("introspective_score", 0.0)))
        for row, output in zip(metadata, outputs)
    ]
    variance_proxies = [_prediction_variance_proxy(output) for output in outputs]
    variance_weighted_errors = [
        abs(mu - label) * variance / max(confidence, 1e-6)
        for mu, label, variance, confidence in zip(
            mu_values,
            labels,
            variance_proxies,
            explanation_confidences,
        )
    ]

    def _tag(coverage: float) -> str:
        return f"{int(round(coverage * 100))}"

    row: dict[str, float | str] = {"method": method}
    for c in coverages:
        row[f"coverage_{_tag(c)}"] = per_coverage_hit[c]
    row["avg_width"] = float(np.mean(list(per_coverage_width.values())))
    for c in coverages:
        row[f"avg_width_{_tag(c)}"] = per_coverage_width[c]
    for c in coverages:
        row[f"cal_error_{_tag(c)}"] = per_coverage_cal_error[c]
    row["calibration_error"] = float(np.mean(list(per_coverage_cal_error.values())))
    row.update(
        {
            "MAE": mae,
            "RMSE": rmse,
            "dir_acc": dir_acc,
            "avg_explanation_confidence": float(np.mean(explanation_confidences)),
            "avg_predicted_variance_proxy": float(np.mean(variance_proxies)),
            "avg_variance_weighted_explanation_error": float(np.mean(variance_weighted_errors)),
        }
    )
    if interval_scales:
        for c in coverages:
            row[f"interval_scale_{_tag(c)}"] = float(interval_scales.get((mode, float(c)), 1.0))
    return row


def _subgroup_metric_rows(
    method: str,
    outputs: list[dict[str, float]],
    labels: list[float],
    regimes: list[str],
    predictor: EventConditionedConformalPredictor,
    global_thresholds: dict[float, float],
    mode: str,
    metadata: list[dict[str, float]] | None = None,
    reference_explanation_confidence: float | None = None,
    normalized_predictors: dict[str, NormalizedConformalPredictor] | None = None,
    interval_scales: dict[tuple[str, float], float] | None = None,
    coverages: list[float] | None = None,
) -> list[dict[str, float | str | int]]:
    """Compute subgroup metrics by surprise band and volatility regime."""

    subgroup_members: dict[tuple[str, str], list[int]] = {}
    for index, regime in enumerate(regimes):
        surprise_band, volatility_band = _regime_components(regime)
        subgroup_members.setdefault(("surprise_band", surprise_band), []).append(index)
        subgroup_members.setdefault(("volatility_band", volatility_band), []).append(index)

    rows: list[dict[str, float | str | int]] = []
    if metadata is None:
        metadata = [{} for _ in outputs]
    attention_bands = _attention_volume_bands(metadata)
    for index, band_name in enumerate(attention_bands):
        subgroup_members.setdefault(("attention_volume_band", band_name), []).append(index)
    for (subgroup_type, subgroup_name), indices in sorted(subgroup_members.items()):
        subgroup_outputs = [outputs[index] for index in indices]
        subgroup_labels = [labels[index] for index in indices]
        subgroup_regimes = [regimes[index] for index in indices]
        subgroup_metadata = [metadata[index] for index in indices]
        row = _metric_row(
            method=method,
            outputs=subgroup_outputs,
            labels=subgroup_labels,
            regimes=subgroup_regimes,
            predictor=predictor,
            global_thresholds=global_thresholds,
            mode=mode,
            metadata=subgroup_metadata,
            reference_explanation_confidence=reference_explanation_confidence,
            normalized_predictors=normalized_predictors,
            interval_scales=interval_scales,
            coverages=coverages,
        )
        row["subgroup_type"] = subgroup_type
        row["subgroup"] = subgroup_name
        row["n"] = len(indices)
        rows.append(row)
    return rows


def _prediction_rows(
    method: str,
    events: list[dict[str, Any]],
    outputs: list[dict[str, float]],
    labels: list[float],
    regimes: list[str],
    tickers: list[str],
    predictor: EventConditionedConformalPredictor,
    global_thresholds: dict[float, float],
    mode: str,
    metadata: list[dict[str, float]] | None = None,
    financial_lookup: dict[tuple[str, str], dict[str, float]] | None = None,
    reference_explanation_confidence: float | None = None,
    normalized_predictors: dict[str, NormalizedConformalPredictor] | None = None,
    interval_scales: dict[tuple[str, float], float] | None = None,
) -> list[dict[str, float | str | int]]:
    """Build per-event prediction rows for export and lightweight inspection."""

    rows: list[dict[str, float | str | int]] = []
    if metadata is None:
        metadata = [_event_calibration_metadata(event) for event in events]
    if financial_lookup is None:
        financial_lookup = {}
    mean_explanation_confidence = (
        float(reference_explanation_confidence)
        if reference_explanation_confidence is not None
        else _mean_explanation_confidence(metadata)
    )

    for event, output, label, regime, ticker, event_metadata in zip(
        events,
        outputs,
        labels,
        regimes,
        tickers,
        metadata,
    ):
        interval_bounds: dict[float, tuple[float, float]] = {}
        for coverage in (0.80, 0.90, 0.95):
            lower, upper = _unscaled_interval_for_mode(
                mode=mode,
                output=output,
                regime=regime,
                coverage=coverage,
                predictor=predictor,
                global_thresholds=global_thresholds,
                metadata=event_metadata,
                mean_explanation_confidence=mean_explanation_confidence,
                normalized_predictors=normalized_predictors,
            )
            lower, upper = _apply_interval_scale(
                output=output,
                coverage=coverage,
                lower=lower,
                upper=upper,
                scale=(interval_scales or {}).get((mode, float(coverage))),
            )
            interval_bounds[coverage] = (float(lower), float(upper))

        raw_sentiment = event.get("sentiment_features", event.get("sentiment", [0.0, 0.0]))
        sentiment_values = [float(value) for value in list(raw_sentiment)[:2]]
        while len(sentiment_values) < 2:
            sentiment_values.append(0.0)
        transcript_text = str(event.get("transcript", "") or "")
        transcript_preview = transcript_text[:240].replace("\n", " ").strip()
        feature_values = _feature_triplet(event)
        mu = _point_prediction_for_method(output, method)
        abs_error = abs(mu - float(label))
        predicted_variance_proxy = _prediction_variance_proxy(output)
        explanation_confidence = float(
            event_metadata.get(
                "explanation_confidence",
                output.get("introspective_score", 0.0),
            )
        )
        explanation_adjusted_abs_error = abs_error / max(explanation_confidence, 1e-6)
        variance_weighted_explanation_error = (
            abs_error * predicted_variance_proxy / max(explanation_confidence, 1e-6)
        )
        financial_metadata = financial_lookup.get(
            (str(ticker).upper(), str(event.get("date", ""))),
            {},
        )
        rows.append(
            {
                "method": method,
                "ticker": str(ticker).upper(),
                "date": str(event.get("date", "")),
                "year": int(event.get("year", 0)),
                "regime": regime,
                "actual_return": float(label),
                "expected_return": mu,
                "predicted_return": mu,
                "actual_minus_expected": float(label) - mu,
                "abs_error": abs_error,
                "prediction_error": mu - float(label),
                "predicted_variance_proxy": predicted_variance_proxy,
                "predicted_q_low": float(output.get("q_low", mu)),
                "predicted_q_high": float(output.get("q_high", mu)),
                "base_interval_width": float(
                    output.get(
                        "q_high",
                        mu,
                    )
                    - output.get(
                        "q_low",
                        mu,
                    )
                ),
                "introspective_score": float(output.get("introspective_score", 0.0)),
                "variance_confidence": float(output.get("variance_confidence", 0.0)),
                "coverage_80_lower": interval_bounds[0.80][0],
                "coverage_80_upper": interval_bounds[0.80][1],
                "interval_80": f"[{interval_bounds[0.80][0]:.4f}, {interval_bounds[0.80][1]:.4f}]",
                "width_80": float(interval_bounds[0.80][1] - interval_bounds[0.80][0]),
                "coverage_90_lower": interval_bounds[0.90][0],
                "coverage_90_upper": interval_bounds[0.90][1],
                "interval_90": f"[{interval_bounds[0.90][0]:.4f}, {interval_bounds[0.90][1]:.4f}]",
                "width_90": float(interval_bounds[0.90][1] - interval_bounds[0.90][0]),
                "coverage_95_lower": interval_bounds[0.95][0],
                "coverage_95_upper": interval_bounds[0.95][1],
                "interval_95": f"[{interval_bounds[0.95][0]:.4f}, {interval_bounds[0.95][1]:.4f}]",
                "width_95": float(interval_bounds[0.95][1] - interval_bounds[0.95][0]),
                "estimated_earnings": float(
                    financial_metadata.get("estimated_earnings", float("nan"))
                ),
                "actual_earnings": float(
                    financial_metadata.get("actual_earnings", float("nan"))
                ),
                "earnings_surprise": float(
                    financial_metadata.get("earnings_surprise", float("nan"))
                ),
                "sue": float(feature_values[0]),
                "momentum": float(feature_values[1]),
                "implied_vol": float(feature_values[2]),
                "avg_sentiment": float(sentiment_values[0]),
                "message_volume": float(sentiment_values[1]),
                "explanation_confidence": explanation_confidence,
                "explanation_adjusted_abs_error": explanation_adjusted_abs_error,
                "variance_weighted_explanation_error": variance_weighted_explanation_error,
                "modality_disagreement": float(
                    event_metadata.get("modality_disagreement", 0.0)
                ),
                "direction_match": int(np.sign(mu) == np.sign(float(label))),
                "explanation": str(output.get("explanation", "")).strip(),
                "transcript_preview": transcript_preview,
            }
        )
    return rows


def _selective_metric_rows(
    outputs: list[dict[str, float]],
    labels: list[float],
    regimes: list[str],
    predictor: EventConditionedConformalPredictor,
    global_thresholds: dict[float, float],
    mode: str,
    min_scores: list[float],
    coverage: float = 0.90,
    metadata: list[dict[str, float]] | None = None,
    reference_explanation_confidence: float | None = None,
) -> list[dict[str, float | str]]:
    """Compute a lightweight selective prediction risk-coverage table."""

    rows: list[dict[str, float | str]] = []
    if metadata is None:
        metadata = [{} for _ in outputs]
    mean_explanation_confidence = (
        float(reference_explanation_confidence)
        if reference_explanation_confidence is not None
        else _mean_explanation_confidence(metadata)
    )
    for min_score in min_scores:
        kept_indices: list[int] = []
        interval_widths: list[float] = []
        interval_hits: list[float] = []
        for index, (output, label, regime, event_metadata) in enumerate(
            zip(outputs, labels, regimes, metadata)
        ):
            explanation_confidence = float(
                event_metadata.get(
                    "explanation_confidence",
                    output.get("introspective_score", 0.0),
                )
            )
            score_for_selection = float(output.get("introspective_score", explanation_confidence))
            if score_for_selection < float(min_score):
                continue
            if mode == "confidence_scaled_naive":
                lower, upper = _predict_interval_confidence_scaled_naive(
                    output=output,
                    coverage=coverage,
                    global_thresholds=global_thresholds,
                    explanation_confidence=explanation_confidence,
                    mean_explanation_confidence=mean_explanation_confidence,
                )
            else:
                interval = predictor.selective_predict(
                    output=output,
                    regime=regime,
                    coverage=coverage,
                    min_score=min_score,
                    metadata=event_metadata,
                )
                if interval is None:
                    continue
                lower, upper = interval
            kept_indices.append(index)
            interval_widths.append(upper - lower)
            interval_hits.append(1.0 if lower <= label <= upper else 0.0)

        if not kept_indices:
            rows.append(
                {
                    "method": "ours_selective",
                    "coverage_level": float(coverage),
                    "min_score": float(min_score),
                    "acceptance_rate": 0.0,
                    "selected_coverage": float("nan"),
                    "selected_avg_width": float("nan"),
                    "selected_mae": float("nan"),
                    "selected_dir_acc": float("nan"),
                }
            )
            continue

        kept_outputs = [outputs[index] for index in kept_indices]
        kept_labels = [labels[index] for index in kept_indices]
        mu_values = [_point_prediction_for_method(output, "ours") for output in kept_outputs]
        rows.append(
            {
                "method": "ours_selective",
                "coverage_level": float(coverage),
                "min_score": float(min_score),
                "acceptance_rate": float(len(kept_indices) / len(outputs)),
                "selected_coverage": float(np.mean(interval_hits)),
                "selected_avg_width": float(np.mean(interval_widths)),
                "selected_mae": float(
                    np.mean([abs(mu - label) for mu, label in zip(mu_values, kept_labels)])
                ),
                "selected_dir_acc": float(
                    np.mean(
                        [
                            1.0 if np.sign(mu) == np.sign(label) else 0.0
                            for mu, label in zip(mu_values, kept_labels)
                        ]
                    )
                ),
            }
        )
    return rows


def evaluate(config_path: str) -> None:
    """Evaluate the trained model, ablations, and conformal baselines.

    Args:
        config_path: Path to the YAML configuration file.
    """

    config_file = _resolve_path(config_path)
    config = _load_config(config_file)
    data_config = config.get("data", {})
    split_config = data_config.get("split", {})
    feature_stats = _load_feature_stats()
    all_events = _load_cached_events()
    all_events = filter_events_by_universe(all_events, data_config.get("universe"))
    if not all_events:
        raise ValueError(
            "No usable cached events were found for evaluation. "
            f"Coverage summary: {summarize_event_coverage(all_events)}"
        )

    try:
        train_events, val_events, cal_events, test_events = _split_events_by_year(
            all_events,
            split_config,
        )
    except Exception as exc:
        raise ValueError(f"Could not build the requested evaluation split. {exc}") from exc
    _assert_disjoint_splits(train_events, val_events, cal_events, test_events)
    regime_thresholds = _load_regime_thresholds(config, train_events + val_events)
    val_dataset = EarningsDataset(val_events, feature_stats=feature_stats)
    cal_dataset = EarningsDataset(cal_events, feature_stats=feature_stats)
    test_dataset = EarningsDataset(test_events, feature_stats=feature_stats)
    financial_lookup = _load_financial_lookup()

    device = _select_device(config)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    model_path = PROJECT_ROOT / "experiments" / "model_best.pt" if (PROJECT_ROOT / "experiments" / "model_best.pt").exists() else PROJECT_ROOT / "experiments" / "model.pt"
    if not model_path.is_file():
        model_path = PROJECT_ROOT / "model.pt"

    model = MultimodalForecastModel.load_from_config(str(config_file))
    state_dict = torch.load(model_path, map_location="cpu")
    model_state = model.state_dict()
    shape_mismatched_keys = [
        key
        for key, value in state_dict.items()
        if key in model_state and tuple(value.shape) != tuple(model_state[key].shape)
    ]
    if shape_mismatched_keys:
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if key not in shape_mismatched_keys
        }
    load_result = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(getattr(load_result, "missing_keys", []))
    unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
    if missing_keys or unexpected_keys or shape_mismatched_keys:
        print(
            "Warning: loaded checkpoint with partial compatibility. "
            f"missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}, "
            f"shape_mismatched_keys={len(shape_mismatched_keys)}"
        )
    model.eval()
    model.to(device)
    _log_device_diagnostics(device, model)

    cal_outputs, cal_labels, cal_regimes, _ = _compute_model_outputs(
        model=model,
        events=cal_events,
        dataset=cal_dataset,
        method="full_multimodal",
        device=device,
        regime_thresholds=regime_thresholds,
    )
    cal_text_outputs, _cal_text_labels, _cal_text_regimes, _ = _compute_model_outputs(
        model=model,
        events=cal_events,
        dataset=cal_dataset,
        method="text_only",
        device=device,
        regime_thresholds=regime_thresholds,
    )
    cal_financial_outputs, _cal_fin_labels, _cal_fin_regimes, _ = _compute_model_outputs(
        model=model,
        events=cal_events,
        dataset=cal_dataset,
        method="financial_only",
        device=device,
        regime_thresholds=regime_thresholds,
    )
    cal_sentiment_outputs, _cal_sent_labels, _cal_sent_regimes, _ = _compute_model_outputs(
        model=model,
        events=cal_events,
        dataset=cal_dataset,
        method="sentiment_only",
        device=device,
        regime_thresholds=regime_thresholds,
    )
    val_outputs, val_labels, val_regimes, _ = _compute_model_outputs(
        model=model,
        events=val_events,
        dataset=val_dataset,
        method="full_multimodal",
        device=device,
        regime_thresholds=regime_thresholds,
    )
    val_text_outputs, _val_text_labels, _val_text_regimes, _ = _compute_model_outputs(
        model=model,
        events=val_events,
        dataset=val_dataset,
        method="text_only",
        device=device,
        regime_thresholds=regime_thresholds,
    )
    val_financial_outputs, _val_fin_labels, _val_fin_regimes, _ = _compute_model_outputs(
        model=model,
        events=val_events,
        dataset=val_dataset,
        method="financial_only",
        device=device,
        regime_thresholds=regime_thresholds,
    )
    val_sentiment_outputs, _val_sent_labels, _val_sent_regimes, _ = _compute_model_outputs(
        model=model,
        events=val_events,
        dataset=val_dataset,
        method="sentiment_only",
        device=device,
        regime_thresholds=regime_thresholds,
    )

    coverage_levels = list(config.get("calibration", {}).get("coverage_levels", [0.80, 0.90, 0.95]))
    include_selective_analysis = bool(
        config.get("evaluation", {}).get("include_selective_analysis", False)
    )
    print(f"Ours interval rule: {os.environ.get('OURS_INTERVAL_RULE', 'confidence_only')}")
    cal_metadata, disagreement_center, disagreement_scale = _build_explanation_metadata(
        events=cal_events,
        full_outputs=cal_outputs,
        text_outputs=cal_text_outputs,
        financial_outputs=cal_financial_outputs,
        sentiment_outputs=cal_sentiment_outputs,
    )
    val_metadata, _val_disagreement_center, _val_disagreement_scale = _build_explanation_metadata(
        events=val_events,
        full_outputs=val_outputs,
        text_outputs=val_text_outputs,
        financial_outputs=val_financial_outputs,
        sentiment_outputs=val_sentiment_outputs,
        reference_center=disagreement_center,
        reference_scale=disagreement_scale,
    )
    calibration_explanation_confidence = _mean_explanation_confidence(cal_metadata)
    global_thresholds = _build_global_thresholds(cal_outputs, cal_labels, coverage_levels)
    historical_events = train_events + val_events + cal_events
    event_conditioned_predictor = EventConditionedConformalPredictor(
        coverage_levels=coverage_levels,
        use_attention_conditioning=True,
        use_explanation_adjustment=False,
    )
    event_conditioned_predictor.calibrate(
        cal_outputs=cal_outputs,
        cal_labels=cal_labels,
        cal_regimes=cal_regimes,
        cal_metadata=cal_metadata,
    )
    normalized_predictors = {
        "normalized_width": NormalizedConformalPredictor(
            coverage_levels=coverage_levels,
            difficulty="width",
        ),
        "normalized_modality": NormalizedConformalPredictor(
            coverage_levels=coverage_levels,
            difficulty="modality_disagreement",
        ),
        "normalized_combined": NormalizedConformalPredictor(
            coverage_levels=coverage_levels,
            difficulty="combined",
        ),
    }
    for normalized_predictor in normalized_predictors.values():
        normalized_predictor.calibrate(
            cal_outputs=cal_outputs,
            cal_labels=cal_labels,
            cal_metadata=cal_metadata,
        )
    interval_scales = _fit_interval_scales(
        modes=["adaptive", "normalized_width", "normalized_modality", "normalized_combined"],
        coverage_levels=coverage_levels,
        cal_outputs=val_outputs,
        cal_labels=val_labels,
        cal_regimes=val_regimes,
        predictor=event_conditioned_predictor,
        global_thresholds=global_thresholds,
        cal_metadata=val_metadata,
        normalized_predictors=normalized_predictors,
    )

    selected_interval_modes: dict[str, str] = {}

    results_rows: list[dict[str, float | str]] = []
    subgroup_rows: list[dict[str, float | str | int]] = []
    selective_rows: list[dict[str, float | str]] = []
    prediction_rows: list[dict[str, float | str | int]] = []

    evaluation_specs: list[tuple[str, str, EventConditionedConformalPredictor]] = [
        ("text_only", "raw_quantile", event_conditioned_predictor),
        ("financial_only", "raw_quantile", event_conditioned_predictor),
        ("sentiment_only", "raw_quantile", event_conditioned_predictor),
        ("full_multimodal", "raw_quantile", event_conditioned_predictor),
        ("naive_conformal", "naive", event_conditioned_predictor),
        ("event_conditioned_conformal", "adaptive", event_conditioned_predictor),
        ("normalized_conformal_width", "normalized_width", event_conditioned_predictor),
        ("normalized_conformal_modality", "normalized_modality", event_conditioned_predictor),
        ("normalized_conformal_combined", "normalized_combined", event_conditioned_predictor),
        ("ours", "normalized_modality", event_conditioned_predictor),
    ]

    test_full_outputs, test_labels, test_regimes, test_tickers = _compute_model_outputs(
        model=model,
        events=test_events,
        dataset=test_dataset,
        method="full_multimodal",
        device=device,
        regime_thresholds=regime_thresholds,
    )
    test_text_outputs, _text_labels, _text_regimes, _text_tickers = _compute_model_outputs(
        model=model,
        events=test_events,
        dataset=test_dataset,
        method="text_only",
        device=device,
        regime_thresholds=regime_thresholds,
    )
    test_financial_outputs, _fin_labels, _fin_regimes, _fin_tickers = _compute_model_outputs(
        model=model,
        events=test_events,
        dataset=test_dataset,
        method="financial_only",
        device=device,
        regime_thresholds=regime_thresholds,
    )
    test_sentiment_outputs, _sent_labels, _sent_regimes, _sent_tickers = _compute_model_outputs(
        model=model,
        events=test_events,
        dataset=test_dataset,
        method="sentiment_only",
        device=device,
        regime_thresholds=regime_thresholds,
    )

    test_metadata, _unused_center, _unused_scale = _build_explanation_metadata(
        events=test_events,
        full_outputs=test_full_outputs,
        text_outputs=test_text_outputs,
        financial_outputs=test_financial_outputs,
        sentiment_outputs=test_sentiment_outputs,
        reference_center=disagreement_center,
        reference_scale=disagreement_scale,
    )

    method_payloads: dict[str, tuple[list[dict[str, float]], list[float], list[str], list[str]]] = {
        "text_only": (test_text_outputs, test_labels, test_regimes, test_tickers),
        "financial_only": (test_financial_outputs, test_labels, test_regimes, test_tickers),
        "sentiment_only": (test_sentiment_outputs, test_labels, test_regimes, test_tickers),
        "full_multimodal": (test_full_outputs, test_labels, test_regimes, test_tickers),
        "naive_conformal": (test_full_outputs, test_labels, test_regimes, test_tickers),
        "event_conditioned_conformal": (test_full_outputs, test_labels, test_regimes, test_tickers),
        "normalized_conformal_width": (test_full_outputs, test_labels, test_regimes, test_tickers),
        "normalized_conformal_modality": (test_full_outputs, test_labels, test_regimes, test_tickers),
        "normalized_conformal_combined": (test_full_outputs, test_labels, test_regimes, test_tickers),
        "ours": (test_full_outputs, test_labels, test_regimes, test_tickers),
    }

    for method, mode, active_predictor in evaluation_specs:
        effective_mode = selected_interval_modes.get(method, mode)
        outputs, labels, regimes, _tickers = method_payloads[method]

        result_row = _metric_row(
            method=method,
            outputs=outputs,
            labels=labels,
            regimes=regimes,
            predictor=active_predictor,
            global_thresholds=global_thresholds,
            mode=effective_mode,
            metadata=test_metadata,
            reference_explanation_confidence=calibration_explanation_confidence,
            normalized_predictors=normalized_predictors,
            interval_scales=interval_scales,
            coverages=coverage_levels,
        )
        result_row["selected_interval_mode"] = effective_mode
        result_row["candidate_interval_mode"] = mode
        results_rows.append(result_row)
        prediction_rows.extend(
            _prediction_rows(
                method=method,
                events=test_events,
                outputs=outputs,
                labels=labels,
                regimes=regimes,
                tickers=_tickers,
                predictor=active_predictor,
                global_thresholds=global_thresholds,
                mode=effective_mode,
                metadata=test_metadata,
                financial_lookup=financial_lookup,
                reference_explanation_confidence=calibration_explanation_confidence,
                normalized_predictors=normalized_predictors,
                interval_scales=interval_scales,
            )
        )
        if include_selective_analysis and method == "ours":
            selective_rows.extend(
                _selective_metric_rows(
                    outputs=outputs,
                    labels=labels,
                    regimes=regimes,
                    predictor=active_predictor,
                    global_thresholds=global_thresholds,
                    mode=effective_mode,
                    min_scores=[
                        float(score)
                        for score in config.get("calibration", {}).get(
                            "selective_min_scores",
                            [0.55, 0.70, 0.85],
                        )
                    ],
                    metadata=test_metadata,
                    reference_explanation_confidence=calibration_explanation_confidence,
                )
            )
        subgroup_rows.extend(
            _subgroup_metric_rows(
                method=method,
                outputs=outputs,
                labels=labels,
                regimes=regimes,
                predictor=active_predictor,
                global_thresholds=global_thresholds,
                mode=effective_mode,
                metadata=test_metadata,
                reference_explanation_confidence=calibration_explanation_confidence,
                normalized_predictors=normalized_predictors,
                interval_scales=interval_scales,
                coverages=coverage_levels,
            )
        )

    baseline_outputs, baseline_labels, baseline_regimes, _baseline_tickers = _compute_same_ticker_baseline(
        test_events=test_events,
        training_events=historical_events,
        regime_thresholds=regime_thresholds,
    )
    results_rows.append(
        _metric_row(
            method="same_ticker_baseline",
            outputs=baseline_outputs,
            labels=baseline_labels,
            regimes=baseline_regimes,
            predictor=event_conditioned_predictor,
            global_thresholds=global_thresholds,
            mode="naive",
            reference_explanation_confidence=calibration_explanation_confidence,
            coverages=coverage_levels,
        )
    )
    prediction_rows.extend(
        _prediction_rows(
            method="same_ticker_baseline",
            events=test_events,
            outputs=baseline_outputs,
            labels=baseline_labels,
            regimes=baseline_regimes,
            tickers=_baseline_tickers,
            predictor=event_conditioned_predictor,
            global_thresholds=global_thresholds,
            mode="naive",
            financial_lookup=financial_lookup,
            reference_explanation_confidence=calibration_explanation_confidence,
        )
    )
    subgroup_rows.extend(
        _subgroup_metric_rows(
            method="same_ticker_baseline",
            outputs=baseline_outputs,
            labels=baseline_labels,
            regimes=baseline_regimes,
            predictor=event_conditioned_predictor,
            global_thresholds=global_thresholds,
            mode="naive",
            reference_explanation_confidence=calibration_explanation_confidence,
            coverages=coverage_levels,
        )
    )

    results_table = pd.DataFrame(results_rows)
    subgroup_results_table = pd.DataFrame(subgroup_rows)
    selective_results_table = pd.DataFrame(selective_rows)
    predictions_table = pd.DataFrame(prediction_rows)

    point_methods = [
        "text_only",
        "financial_only",
        "sentiment_only",
        "full_multimodal",
        "same_ticker_baseline",
    ]
    interval_methods = [
        "full_multimodal",
        "naive_conformal",
        "event_conditioned_conformal",
        "normalized_conformal_width",
        "normalized_conformal_modality",
        "normalized_conformal_combined",
        "ours",
        "same_ticker_baseline",
    ]
    point_columns = ["method", "MAE", "RMSE", "dir_acc"]
    coverage_tags = [f"{int(round(float(c) * 100))}" for c in coverage_levels]
    interval_columns = (
        ["method", "selected_interval_mode", "candidate_interval_mode"]
        + [f"coverage_{tag}" for tag in coverage_tags]
        + [f"avg_width_{tag}" for tag in coverage_tags]
        + ["avg_width"]
        + [f"cal_error_{tag}" for tag in coverage_tags]
        + ["calibration_error"]
    )
    point_table = (
        results_table[results_table["method"].isin(point_methods)][point_columns]
        .set_index("method")
        .reindex([m for m in point_methods if m in set(results_table["method"])])
        .reset_index()
    )
    interval_table = (
        results_table[results_table["method"].isin(interval_methods)][interval_columns]
        .set_index("method")
        .reindex([m for m in interval_methods if m in set(results_table["method"])])
        .reset_index()
    )

    print("Point-forecast comparison")
    print(tabulate(point_table, headers="keys", tablefmt="github", showindex=False))
    print("\nInterval comparison")
    print(tabulate(interval_table, headers="keys", tablefmt="github", showindex=False))
    if not subgroup_results_table.empty:
        subgroup_display_columns = (
            ["method", "subgroup_type", "subgroup", "n"]
            + [f"coverage_{tag}" for tag in coverage_tags]
            + ["avg_width", "calibration_error"]
        )
        print(
            tabulate(
                subgroup_results_table[subgroup_display_columns],
                headers="keys",
                tablefmt="github",
                showindex=False,
            )
        )
    if include_selective_analysis and not selective_results_table.empty:
        print(
            tabulate(
                selective_results_table,
                headers="keys",
                tablefmt="github",
                showindex=False,
            )
        )

    experiments_results_path = PROJECT_ROOT / "experiments" / "results.csv"
    root_results_path = PROJECT_ROOT / "results.csv"
    experiments_subgroup_results_path = PROJECT_ROOT / "experiments" / "results_by_subgroup.csv"
    root_subgroup_results_path = PROJECT_ROOT / "results_by_subgroup.csv"
    experiments_predictions_path = PROJECT_ROOT / "experiments" / "predictions.csv"
    root_predictions_path = PROJECT_ROOT / "predictions.csv"
    experiments_selective_results_path = PROJECT_ROOT / "experiments" / "results_selective.csv"
    root_selective_results_path = PROJECT_ROOT / "results_selective.csv"
    results_table.to_csv(experiments_results_path, index=False)
    results_table.to_csv(root_results_path, index=False)
    point_table.to_csv(PROJECT_ROOT / "experiments" / "results_point.csv", index=False)
    point_table.to_csv(PROJECT_ROOT / "results_point.csv", index=False)
    interval_table.to_csv(PROJECT_ROOT / "experiments" / "results_intervals.csv", index=False)
    interval_table.to_csv(PROJECT_ROOT / "results_intervals.csv", index=False)
    subgroup_results_table.to_csv(experiments_subgroup_results_path, index=False)
    subgroup_results_table.to_csv(root_subgroup_results_path, index=False)
    predictions_table.to_csv(experiments_predictions_path, index=False)
    predictions_table.to_csv(root_predictions_path, index=False)
    if include_selective_analysis:
        selective_results_table.to_csv(experiments_selective_results_path, index=False)
        selective_results_table.to_csv(root_selective_results_path, index=False)
    else:
        for selective_path in (
            experiments_selective_results_path,
            root_selective_results_path,
        ):
            if selective_path.exists():
                selective_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the multimodal forecast model.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--allow-synthetic-fallback",
        action="store_true",
        help="Compatibility flag accepted by legacy tests; evaluation remains cache-based.",
    )
    args = parser.parse_args()

    evaluate(args.config)
