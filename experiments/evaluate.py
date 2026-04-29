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
    build_synthetic_events as _shared_build_synthetic_events,
    filter_events_by_universe,
    summarize_event_coverage,
)
from encoders.sentiment_encoder import aggregate_posts
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


def _build_synthetic_fallback_events() -> list[dict[str, Any]]:
    """Build a tiny year-split event set for evaluation smoke tests."""

    normalized_events: list[dict[str, Any]] = []
    fallback_years = [2021, 2022, 2023, 2024, 2025, 2025]
    for index, event in enumerate(_shared_build_synthetic_events()):
        raw_posts = [str(post) for post in list(event.get("sentiment_posts", []))]
        sentiment_features = aggregate_posts(raw_posts)
        raw_features = event.get("features", {})
        original_date = str(event.get("date", "2023-01-01"))
        adjusted_year = fallback_years[min(index, len(fallback_years) - 1)]
        adjusted_date = f"{adjusted_year}{original_date[4:]}"
        normalized_events.append(
            {
                "ticker": str(event.get("ticker", "")).upper(),
                "date": adjusted_date,
                "transcript": str(event.get("transcript", "")),
                "features": [
                    float(raw_features.get("sue", 0.0)),
                    float(raw_features.get("momentum", 0.0)),
                    float(raw_features.get("implied_vol", 0.0)),
                ],
                "sentiment_raw": raw_posts,
                "sentiment_features": [
                    float(sentiment_features[0].item()),
                    float(sentiment_features[1].item()),
                ],
                "label": float(event.get("label", 0.0)),
                "year": adjusted_year,
            }
        )
    return normalized_events


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
    }


def _explanation_confidence_from_disagreement(
    disagreement: float,
    center: float,
    scale: float,
) -> float:
    """Map modality disagreement to a bounded confidence score."""

    normalized = (float(disagreement) - float(center)) / max(float(scale), 1e-6)
    normalized = min(max(normalized, -50.0), 50.0)
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

    disagreements: list[float] = []
    for full_output, text_output, financial_output, sentiment_output in zip(
        full_outputs,
        text_outputs,
        financial_outputs,
        sentiment_outputs,
    ):
        mu_values = [
            _point_prediction_for_method(text_output, "text_only"),
            _point_prediction_for_method(financial_output, "financial_only"),
            _point_prediction_for_method(sentiment_output, "sentiment_only"),
            _point_prediction_for_method(full_output, "full_multimodal"),
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
        if "llm_confidence" in full_output:
            llm_confidence = min(max(float(full_output["llm_confidence"]), 0.0), 1.0)
        else:
            llm_confidence = model_confidence
        metadata_rows.append(
            {
                **base_metadata,
                "modality_disagreement": float(disagreement),
                "disagreement_confidence": float(disagreement_confidence),
                "model_confidence": model_confidence,
                "llm_confidence": float(llm_confidence),
                "explanation_confidence": float(llm_confidence),
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

    if method == "naive_conformal":
        return _to_float(output.get("quantile_median", output["mu"]))
    if method == "ours" and "ensemble_mu" in output:
        return _to_float(output["ensemble_mu"])
    return _to_float(output.get("point_mu", output["mu"]))


_ENSEMBLE_MODALITIES: tuple[str, ...] = (
    "full_multimodal",
    "text_only",
    "financial_only",
    "sentiment_only",
)


def _fit_modal_ensemble_weights(
    cal_full_outputs: list[dict[str, Any]],
    cal_text_outputs: list[dict[str, Any]],
    cal_financial_outputs: list[dict[str, Any]],
    cal_sentiment_outputs: list[dict[str, Any]],
    cal_labels: list[float],
) -> dict[str, float]:
    """Fit non-negative simplex ensemble weights minimizing |y - sum w_m * mu_m|^2."""

    from scipy.optimize import minimize

    if not (
        len(cal_full_outputs)
        == len(cal_text_outputs)
        == len(cal_financial_outputs)
        == len(cal_sentiment_outputs)
        == len(cal_labels)
    ):
        raise ValueError("Modal ensemble inputs must have matching lengths.")

    mu_full = np.asarray(
        [_point_prediction_for_method(o, "full_multimodal") for o in cal_full_outputs],
        dtype=float,
    )
    mu_text = np.asarray(
        [_point_prediction_for_method(o, "text_only") for o in cal_text_outputs],
        dtype=float,
    )
    mu_financial = np.asarray(
        [_point_prediction_for_method(o, "financial_only") for o in cal_financial_outputs],
        dtype=float,
    )
    mu_sentiment = np.asarray(
        [_point_prediction_for_method(o, "sentiment_only") for o in cal_sentiment_outputs],
        dtype=float,
    )
    design = np.stack([mu_full, mu_text, mu_financial, mu_sentiment], axis=1)
    targets = np.asarray(cal_labels, dtype=float)

    def loss(weights: np.ndarray) -> float:
        return float(np.mean((design @ weights - targets) ** 2))

    constraints = ({"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},)
    bounds = [(0.0, 1.0)] * design.shape[1]
    initial = np.full(design.shape[1], 1.0 / design.shape[1])

    result = minimize(
        loss,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 200, "disp": False},
    )
    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
    weight_sum = float(weights.sum())
    if weight_sum < 1e-9:
        weights = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        weights = weights / weight_sum
    return {name: float(value) for name, value in zip(_ENSEMBLE_MODALITIES, weights)}


def _apply_ensemble_weights(
    full_outputs: list[dict[str, Any]],
    text_outputs: list[dict[str, Any]],
    financial_outputs: list[dict[str, Any]],
    sentiment_outputs: list[dict[str, Any]],
    weights: dict[str, float],
) -> None:
    """Inject ``ensemble_mu`` into each full_multimodal output dict in place."""

    w_full = float(weights.get("full_multimodal", 0.0))
    w_text = float(weights.get("text_only", 0.0))
    w_financial = float(weights.get("financial_only", 0.0))
    w_sentiment = float(weights.get("sentiment_only", 0.0))
    for full, text, financial, sentiment in zip(
        full_outputs, text_outputs, financial_outputs, sentiment_outputs
    ):
        full["ensemble_mu"] = float(
            w_full * _point_prediction_for_method(full, "full_multimodal")
            + w_text * _point_prediction_for_method(text, "text_only")
            + w_financial * _point_prediction_for_method(financial, "financial_only")
            + w_sentiment * _point_prediction_for_method(sentiment, "sentiment_only")
        )


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


def _build_global_thresholds(
    cal_outputs: list[dict[str, Any]],
    cal_labels: list[float],
    coverage_levels: list[float],
) -> dict[float, float]:
    """Build global absolute-error thresholds for a naive conformal baseline."""

    scores: list[float] = []
    for output, label in zip(cal_outputs, cal_labels):
        mu = _point_prediction_for_method(output, "naive_conformal")
        scores.append(abs(float(label) - mu))

    thresholds: dict[float, float] = {}
    n = len(scores)
    for coverage in coverage_levels:
        quantile_level = coverage * (1.0 + 1.0 / n)
        thresholds[float(coverage)] = _conformal_quantile(scores, quantile_level)
    return thresholds


def _confidence_variance_normalizer(
    output: dict[str, Any],
    explanation_confidence: float,
) -> float:
    """Return string_conf * std as the normalized conformal scale.

    Using std (interval half-width) rather than variance keeps calibration
    scores in a ~1-3x range instead of ~10-100x, so q_hat stays reasonable
    and test-time intervals are comparably sized to naive conformal.
    """

    llm_abs_return = float(output.get("llm_predicted_abs_return", 0.0))
    if llm_abs_return > 1e-6:
        return float(max(llm_abs_return, 1e-9))
    confidence = min(max(float(explanation_confidence), 0.05), 0.95)
    variance = max(_prediction_variance_proxy(output), 1e-6)
    std = math.sqrt(variance)
    return float(max(confidence * std, 1e-9))


def _build_confidence_variance_thresholds(
    cal_outputs: list[dict[str, Any]],
    cal_labels: list[float],
    cal_metadata: list[dict[str, float]],
    coverage_levels: list[float],
    reference_explanation_confidence: float,
) -> dict[float, float]:
    """Calibrate residuals normalized by LLM confidence times variance."""

    scores: list[float] = []
    for output, label, event_metadata in zip(cal_outputs, cal_labels, cal_metadata):
        mu = _point_prediction_for_method(output, "ours")
        explanation_confidence = float(
            event_metadata.get(
                "explanation_confidence",
                output.get("introspective_score", reference_explanation_confidence),
            )
        )
        normalizer = _confidence_variance_normalizer(
            output=output,
            explanation_confidence=explanation_confidence,
        )
        scores.append(abs(float(label) - mu) / normalizer)

    thresholds: dict[float, float] = {}
    n = max(len(scores), 1)
    for coverage in coverage_levels:
        quantile_level = coverage * (1.0 + 1.0 / n)
        thresholds[float(coverage)] = _conformal_quantile(scores, quantile_level)
    return thresholds


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
    llm_explainer: Any | None = None,
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

    if llm_explainer is not None:
        llm_items = [
            {
                "ticker": str(events[index].get("ticker", "")),
                "date": str(events[index].get("date", "")),
                "method": method,
                "transcript": transcripts[index] if index < len(transcripts) else "",
                "mu": output["mu"],
                "q_low": output["q_low"],
                "q_high": output["q_high"],
            }
            for index, output in enumerate(outputs)
        ]
        llm_results = llm_explainer.explain_batch(llm_items)
        for output, parsed in zip(outputs, llm_results):
            output["explanation"] = str(parsed.get("explanation", ""))
            output["llm_confidence"] = float(parsed.get("confidence", 0.5))
            output["llm_predicted_abs_return"] = float(parsed.get("abs_return", 0.02))

    return outputs, [float(value) for value in labels.cpu().tolist()], regimes, tickers


def _compute_ticker_volatilities(
    historical_events: list[dict[str, Any]],
    min_events: int = 2,
) -> tuple[dict[str, float], float]:
    """Compute per-ticker historical return std from past events.

    Returns (per_ticker_vol, default_vol) where ``default_vol`` is the median
    used when a ticker has too few historical events to estimate.
    """

    by_ticker: dict[str, list[float]] = {}
    for event in historical_events:
        ticker = str(event.get("ticker", "")).upper()
        label = event.get("label")
        if not ticker or label is None:
            continue
        try:
            value = float(label)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        by_ticker.setdefault(ticker, []).append(value)

    per_ticker: dict[str, float] = {}
    for ticker, values in by_ticker.items():
        if len(values) >= int(min_events):
            sigma = float(np.std(values, ddof=1))
            if math.isfinite(sigma) and sigma > 0.0:
                per_ticker[ticker] = sigma
    default_vol = (
        float(np.median(list(per_ticker.values()))) if per_ticker else 0.02
    )
    return per_ticker, default_vol


def _attach_ticker_volatilities(
    outputs: list[dict[str, Any]],
    events: list[dict[str, Any]],
    per_ticker_vol: dict[str, float],
    default_vol: float,
) -> None:
    """Inject ``ticker_historical_vol`` into each output dict in place."""

    for output, event in zip(outputs, events):
        ticker = str(event.get("ticker", "")).upper()
        sigma = per_ticker_vol.get(ticker, float(default_vol))
        output["ticker_historical_vol"] = float(max(sigma, 1e-6))


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
    """Predict a symmetric globally conformalized interval around the point forecast."""

    mu = _point_prediction_for_method(output, "naive_conformal")
    threshold = global_thresholds[float(coverage)]
    return mu - threshold, mu + threshold


def _predict_interval_confidence_scaled_naive(
    output: dict[str, Any],
    coverage: float,
    global_thresholds: dict[float, float],
    explanation_confidence: float,
    mean_explanation_confidence: float,
) -> tuple[float, float]:
    """Invert confidence-times-variance normalized conformal calibration."""

    mu = _point_prediction_for_method(output, "ours")
    q_hat = float(global_thresholds[float(coverage)])
    normalizer = _confidence_variance_normalizer(
        output=output,
        explanation_confidence=explanation_confidence,
    )
    del mean_explanation_confidence
    adjusted_threshold = q_hat * normalizer
    return mu - adjusted_threshold, mu + adjusted_threshold


def _confidence_adjusted_abs_error(abs_error: float, explanation_confidence: float) -> float:
    """Penalize large errors more when the model reports high confidence."""

    confidence = min(max(float(explanation_confidence), 0.0), 1.0)
    return float(abs_error * (0.5 + confidence))


def _variance_weighted_explanation_error(
    abs_error: float,
    explanation_confidence: float,
    predicted_variance: float,
) -> float:
    """Combine error, confidence, and predicted variance for diagnostics."""

    adjusted_error = _confidence_adjusted_abs_error(abs_error, explanation_confidence)
    return float(adjusted_error / math.sqrt(max(float(predicted_variance), 1e-6)))


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
) -> dict[str, float | str]:
    """Compute evaluation metrics for one method row."""

    coverages = [0.80, 0.90, 0.95]
    interval_hits: dict[float, list[float]] = {coverage: [] for coverage in coverages}
    interval_widths: dict[float, list[float]] = {coverage: [] for coverage in coverages}

    mu_values = [_point_prediction_for_method(output, method) for output in outputs]
    abs_errors = [abs(mu - y) for mu, y in zip(mu_values, labels)]
    mae = float(np.mean(abs_errors))
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
    explanation_confidences = [
        float(
            event_metadata.get(
                "explanation_confidence",
                output.get("introspective_score", 0.0),
            )
        )
        for output, event_metadata in zip(outputs, metadata)
    ]
    predicted_variances = [_prediction_variance_proxy(output) for output in outputs]
    variance_weighted_errors = [
        _variance_weighted_explanation_error(abs_error, confidence, variance)
        for abs_error, confidence, variance in zip(
            abs_errors,
            explanation_confidences,
            predicted_variances,
        )
    ]

    for output, label, regime, event_metadata in zip(outputs, labels, regimes, metadata):
        explanation_confidence = float(
            event_metadata.get(
                "explanation_confidence",
                output.get("introspective_score", 0.0),
            )
        )
        for coverage in coverages:
            if mode == "adaptive":
                try:
                    lower, upper = predictor.predict_interval(
                        output=output,
                        regime=regime,
                        coverage=coverage,
                        metadata=event_metadata,
                    )
                except KeyError:
                    threshold = _fallback_threshold(predictor, coverage)
                    base_lower, base_upper = _output_interval_bounds_for_coverage(output, coverage)
                    lower, upper = base_lower - threshold, base_upper + threshold
            elif mode == "naive":
                lower, upper = _predict_interval_naive(output, coverage, global_thresholds)
            elif mode == "confidence_scaled_naive":
                lower, upper = _predict_interval_confidence_scaled_naive(
                    output=output,
                    coverage=coverage,
                    global_thresholds=global_thresholds,
                    explanation_confidence=explanation_confidence,
                    mean_explanation_confidence=mean_explanation_confidence,
                )
            else:
                lower, upper = _predict_interval_without_adjustment(
                    predictor,
                    output,
                    regime,
                    coverage,
                )

            interval_hits[coverage].append(1.0 if lower <= label <= upper else 0.0)
            interval_widths[coverage].append(upper - lower)

    avg_width_80 = float(np.mean(interval_widths[0.80]))
    avg_width_90 = float(np.mean(interval_widths[0.90]))
    avg_width_95 = float(np.mean(interval_widths[0.95]))

    return {
        "method": method,
        "coverage_80": float(np.mean(interval_hits[0.80])),
        "coverage_90": float(np.mean(interval_hits[0.90])),
        "coverage_95": float(np.mean(interval_hits[0.95])),
        "avg_width": float(np.mean([avg_width_80, avg_width_90, avg_width_95])),
        "avg_width_80": avg_width_80,
        "avg_width_90": avg_width_90,
        "avg_width_95": avg_width_95,
        "MAE": mae,
        "RMSE": rmse,
        "dir_acc": dir_acc,
        "avg_explanation_confidence": float(np.mean(explanation_confidences)),
        "avg_predicted_variance_proxy": float(np.mean(predicted_variances)),
        "avg_variance_weighted_explanation_error": float(np.mean(variance_weighted_errors)),
    }


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
            if mode == "adaptive":
                try:
                    lower, upper = predictor.predict_interval(
                        output=output,
                        regime=regime,
                        coverage=coverage,
                        metadata=event_metadata,
                    )
                except KeyError:
                    threshold = _fallback_threshold(predictor, coverage)
                    base_lower, base_upper = _output_interval_bounds_for_coverage(
                        output,
                        coverage,
                    )
                    lower, upper = base_lower - threshold, base_upper + threshold
            elif mode == "naive":
                lower, upper = _predict_interval_naive(output, coverage, global_thresholds)
            elif mode == "confidence_scaled_naive":
                lower, upper = _predict_interval_confidence_scaled_naive(
                    output=output,
                    coverage=coverage,
                    global_thresholds=global_thresholds,
                    explanation_confidence=float(
                        event_metadata.get(
                            "explanation_confidence",
                            output.get("introspective_score", 0.0),
                        )
                    ),
                    mean_explanation_confidence=mean_explanation_confidence,
                )
            else:
                lower, upper = _predict_interval_without_adjustment(
                    predictor,
                    output,
                    regime,
                    coverage,
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
        explanation_confidence = float(
            event_metadata.get(
                "explanation_confidence",
                output.get("introspective_score", 0.0),
            )
        )
        predicted_variance_proxy = _prediction_variance_proxy(output)
        explanation_adjusted_abs_error = _confidence_adjusted_abs_error(
            abs_error,
            explanation_confidence,
        )
        variance_weighted_explanation_error = _variance_weighted_explanation_error(
            abs_error,
            explanation_confidence,
            predicted_variance_proxy,
        )
        confidence_variance_normalizer = _confidence_variance_normalizer(
            output=output,
            explanation_confidence=explanation_confidence,
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
                "variance_confidence": float(
                    output.get(
                        "variance_confidence",
                        output.get("interval_confidence", 0.0),
                    )
                ),
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
                "llm_confidence": float(
                    event_metadata.get(
                        "llm_confidence",
                        output.get("introspective_score", explanation_confidence),
                    )
                ),
                "llm_predicted_abs_return": float(
                    output.get("llm_predicted_abs_return", 0.0)
                ),
                "ticker_historical_vol": float(
                    output.get("ticker_historical_vol", 0.0)
                ),
                "explanation_confidence": explanation_confidence,
                "confidence_variance_normalizer": confidence_variance_normalizer,
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


def _maybe_load_llm_explainer(
    use_llm: bool,
    model_name: str,
    cache_path: str | Path,
    quantization: str | None,
    gpu_memory_utilization: float,
) -> Any | None:
    """Lazy-load the vLLM explainer when ``--use-llm`` is set."""

    if not use_llm:
        return None
    from encoders.llm_explainer import LLMExplainer

    print(f"Loading LLM explainer: {model_name} (cache={cache_path})")
    return LLMExplainer(
        model_name=model_name,
        cache_path=cache_path,
        quantization=quantization,
        gpu_memory_utilization=gpu_memory_utilization,
    )


def evaluate(
    config_path: str,
    allow_synthetic_fallback: bool = False,
    llm_explainer: Any | None = None,
) -> None:
    """Evaluate the trained model, ablations, and conformal baselines.

    Args:
        config_path: Path to the YAML configuration file.
        allow_synthetic_fallback: Reserved compatibility flag.
        llm_explainer: Optional ``LLMExplainer`` for narrative explanations + confidence.
    """

    config_file = _resolve_path(config_path)
    config = _load_config(config_file)
    data_config = config.get("data", {})
    split_config = data_config.get("split", {})
    use_synthetic_fallback = bool(
        allow_synthetic_fallback
        or data_config.get("split", {}).get("allow_synthetic_fallback", False)
    )
    try:
        all_events = _load_cached_events()
    except FileNotFoundError:
        if not use_synthetic_fallback:
            raise
        print("data/events_cache.pt not found; using synthetic fallback events.")
        all_events = _build_synthetic_fallback_events()
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
    try:
        feature_stats = _load_feature_stats()
    except FileNotFoundError:
        if not use_synthetic_fallback:
            raise
        feature_stats = EarningsDataset.compute_feature_stats(train_events)
    regime_thresholds = _load_regime_thresholds(config, train_events + val_events)
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
    if model_path.is_file():
        state_dict = torch.load(model_path, map_location="cpu")
        load_result = model.load_state_dict(state_dict, strict=False)
        missing_keys = list(getattr(load_result, "missing_keys", []))
        unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
        if missing_keys or unexpected_keys:
            print(
                "Warning: loaded checkpoint with partial compatibility. "
                f"missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}"
            )
    elif use_synthetic_fallback:
        print("Model checkpoint not found; evaluating an untrained model for smoke-test fallback.")
    else:
        raise FileNotFoundError(
            "No model checkpoint found. Run: python3 experiments/train.py first."
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
        llm_explainer=llm_explainer,
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

    coverage_levels = list(config.get("calibration", {}).get("coverage_levels", [0.80, 0.90, 0.95]))
    include_selective_analysis = bool(
        config.get("evaluation", {}).get("include_selective_analysis", False)
    )

    per_ticker_vol, default_ticker_vol = _compute_ticker_volatilities(
        train_events + val_events + cal_events
    )
    print(
        f"Ticker historical-vol coverage: {len(per_ticker_vol)} tickers, "
        f"default={default_ticker_vol:.4f}"
    )
    _attach_ticker_volatilities(cal_outputs, cal_events, per_ticker_vol, default_ticker_vol)

    print("Ours interval rule: ticker_historical_vol_normalized_conformal")
    cal_metadata, disagreement_center, disagreement_scale = _build_explanation_metadata(
        events=cal_events,
        full_outputs=cal_outputs,
        text_outputs=cal_text_outputs,
        financial_outputs=cal_financial_outputs,
        sentiment_outputs=cal_sentiment_outputs,
    )
    calibration_explanation_confidence = _mean_explanation_confidence(cal_metadata)
    ensemble_weights = _fit_modal_ensemble_weights(
        cal_full_outputs=cal_outputs,
        cal_text_outputs=cal_text_outputs,
        cal_financial_outputs=cal_financial_outputs,
        cal_sentiment_outputs=cal_sentiment_outputs,
        cal_labels=cal_labels,
    )
    print(
        "Modal ensemble weights: "
        + ", ".join(f"{name}={ensemble_weights[name]:.3f}" for name in _ENSEMBLE_MODALITIES)
    )
    _apply_ensemble_weights(
        full_outputs=cal_outputs,
        text_outputs=cal_text_outputs,
        financial_outputs=cal_financial_outputs,
        sentiment_outputs=cal_sentiment_outputs,
        weights=ensemble_weights,
    )
    global_thresholds = _build_global_thresholds(cal_outputs, cal_labels, coverage_levels)
    confidence_variance_thresholds = _build_confidence_variance_thresholds(
        cal_outputs=cal_outputs,
        cal_labels=cal_labels,
        cal_metadata=cal_metadata,
        coverage_levels=coverage_levels,
        reference_explanation_confidence=calibration_explanation_confidence,
    )
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
        ("ours", "confidence_scaled_naive", event_conditioned_predictor),
    ]

    for method, mode, _active_predictor in evaluation_specs:
        base_method = "full_multimodal" if method in {"full_multimodal", "naive_conformal", "ours"} else method
        outputs, labels, regimes, _tickers = _compute_model_outputs(
            model=model,
            events=test_events,
            dataset=test_dataset,
            method=base_method,
            device=device,
            regime_thresholds=regime_thresholds,
            llm_explainer=llm_explainer if base_method == "full_multimodal" else None,
        )
        if base_method == "full_multimodal":
            test_full_outputs = outputs
            test_labels = labels
            test_regimes = regimes
            test_tickers = _tickers
            continue
        if base_method == "text_only":
            test_text_outputs = outputs
            continue
        if base_method == "financial_only":
            test_financial_outputs = outputs
            continue
        if base_method == "sentiment_only":
            test_sentiment_outputs = outputs
            continue

    _apply_ensemble_weights(
        full_outputs=test_full_outputs,
        text_outputs=test_text_outputs,
        financial_outputs=test_financial_outputs,
        sentiment_outputs=test_sentiment_outputs,
        weights=ensemble_weights,
    )
    _attach_ticker_volatilities(
        test_full_outputs, test_events, per_ticker_vol, default_ticker_vol
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
        "ours": (test_full_outputs, test_labels, test_regimes, test_tickers),
    }

    for method, mode, active_predictor in evaluation_specs:
        outputs, labels, regimes, _tickers = method_payloads[method]
        thresholds_for_call = (
            confidence_variance_thresholds
            if mode == "confidence_scaled_naive"
            else global_thresholds
        )

        results_rows.append(
            _metric_row(
                method=method,
                outputs=outputs,
                labels=labels,
                regimes=regimes,
                predictor=active_predictor,
                global_thresholds=thresholds_for_call,
                mode=mode,
                metadata=test_metadata,
                reference_explanation_confidence=calibration_explanation_confidence,
            )
        )
        prediction_rows.extend(
            _prediction_rows(
                method=method,
                events=test_events,
                outputs=outputs,
                labels=labels,
                regimes=regimes,
                tickers=_tickers,
                predictor=active_predictor,
                global_thresholds=thresholds_for_call,
                mode=mode,
                metadata=test_metadata,
                financial_lookup=financial_lookup,
                reference_explanation_confidence=calibration_explanation_confidence,
            )
        )
        if include_selective_analysis and method == "ours":
            selective_rows.extend(
                _selective_metric_rows(
                    outputs=outputs,
                    labels=labels,
                    regimes=regimes,
                    predictor=active_predictor,
                    global_thresholds=thresholds_for_call,
                    mode=mode,
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
                global_thresholds=thresholds_for_call,
                mode=mode,
                metadata=test_metadata,
                reference_explanation_confidence=calibration_explanation_confidence,
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
        )
    )

    results_table = pd.DataFrame(results_rows)
    subgroup_results_table = pd.DataFrame(subgroup_rows)
    selective_results_table = pd.DataFrame(selective_rows)
    predictions_table = pd.DataFrame(prediction_rows)
    display_columns = [
        "method",
        "coverage_80",
        "coverage_90",
        "coverage_95",
        "avg_width",
        "MAE",
        "RMSE",
        "dir_acc",
    ]
    print(tabulate(results_table[display_columns], headers="keys", tablefmt="github", showindex=False))
    if not subgroup_results_table.empty:
        subgroup_display_columns = [
            "method",
            "subgroup_type",
            "subgroup",
            "n",
            "coverage_80",
            "coverage_90",
            "coverage_95",
            "avg_width",
        ]
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
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use a vLLM-backed explainer to generate explanations and confidence scores.",
    )
    parser.add_argument(
        "--llm-model",
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="HuggingFace model id for the vLLM explainer.",
    )
    parser.add_argument(
        "--llm-cache",
        default=str(PROJECT_ROOT / "data" / "llm_cache.json"),
        help="JSON cache path for LLM explanations and confidences.",
    )
    parser.add_argument(
        "--llm-quantization",
        default="awq",
        help="vLLM quantization scheme (e.g. 'awq', 'gptq', or empty string for none).",
    )
    parser.add_argument(
        "--llm-gpu-memory",
        type=float,
        default=0.85,
        help="vLLM gpu_memory_utilization fraction.",
    )
    args = parser.parse_args()

    quantization = args.llm_quantization or None
    if isinstance(quantization, str) and quantization.strip().lower() in {"", "none"}:
        quantization = None
    llm_explainer = _maybe_load_llm_explainer(
        use_llm=args.use_llm,
        model_name=args.llm_model,
        cache_path=args.llm_cache,
        quantization=quantization,
        gpu_memory_utilization=args.llm_gpu_memory,
    )
    evaluate(
        args.config,
        allow_synthetic_fallback=args.allow_synthetic_fallback,
        llm_explainer=llm_explainer,
    )
