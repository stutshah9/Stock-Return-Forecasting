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
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
SP500_TICKERS_PATH = PROJECT_ROOT / "data" / "sp500_tickers.csv"


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


def _load_ticker_sector_lookup() -> dict[str, str]:
    """Load optional ticker-sector metadata when it exists locally."""

    if not SP500_TICKERS_PATH.is_file():
        return {}

    try:
        tickers = pd.read_csv(SP500_TICKERS_PATH)
    except Exception:
        return {}
    if "ticker" not in tickers.columns:
        return {}

    sector_column = None
    for candidate in ("sector", "gics_sector", "GICS Sector", "industry", "Industry"):
        if candidate in tickers.columns:
            sector_column = candidate
            break
    if sector_column is None:
        return {}

    lookup: dict[str, str] = {}
    for _, row in tickers.iterrows():
        ticker = str(row.get("ticker", "")).upper()
        sector = str(row.get(sector_column, "")).strip()
        if ticker and sector and sector.lower() != "nan":
            lookup[ticker] = sector
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


def _event_sector(event: dict[str, Any], sector_lookup: dict[str, str]) -> str:
    """Return a stable sector label for subgrouping and sector baselines."""

    for key in ("sector", "gics_sector", "industry"):
        value = str(event.get(key, "")).strip()
        if value and value.lower() != "nan":
            return value
    ticker = str(event.get("ticker", "")).upper()
    return sector_lookup.get(ticker, "unknown")


def _event_feature_row(event: dict[str, Any], sector_lookup: dict[str, str]) -> dict[str, Any]:
    """Build a tabular feature row for non-neural baseline models."""

    sue, momentum, implied_vol = _feature_triplet(event)
    sentiment, message_volume = _sentiment_pair(event)
    ticker = str(event.get("ticker", "")).upper()
    return {
        "ticker": ticker,
        "sector": _event_sector(event, sector_lookup),
        "year": int(event.get("year", 0)),
        "sue": float(sue),
        "abs_sue": abs(float(sue)),
        "momentum": float(momentum),
        "implied_vol": float(implied_vol),
        "avg_sentiment": float(sentiment),
        "message_volume": float(message_volume),
        "transcript_length": float(len(str(event.get("transcript", "") or ""))),
    }


def _events_to_frame(
    events: list[dict[str, Any]],
    sector_lookup: dict[str, str],
) -> pd.DataFrame:
    """Convert cached events into a model-ready baseline feature table."""

    return pd.DataFrame([_event_feature_row(event, sector_lookup) for event in events])


def _event_labels(events: list[dict[str, Any]]) -> np.ndarray:
    """Return event labels as a finite float array."""

    return np.asarray([float(event.get("label", 0.0)) for event in events], dtype=float)


def _explanation_confidence_from_disagreement(
    disagreement: float,
    center: float,
    scale: float,
) -> float:
    """Map modality disagreement to a bounded confidence score."""

    normalized = (float(disagreement) - float(center)) / max(float(scale), 1e-6)
    return float(1.0 / (1.0 + math.exp(normalized)))


def _bounded_unit(value: float) -> float:
    """Clamp a numeric score into the unit interval."""

    if not math.isfinite(float(value)):
        return 0.5
    return float(min(max(float(value), 0.0), 1.0))


def _direction_match_score(reference: float, candidate: float, neutral_band: float = 1e-6) -> float:
    """Score whether two signed signals agree, treating tiny values as neutral."""

    if abs(float(reference)) <= neutral_band or abs(float(candidate)) <= neutral_band:
        return 0.5
    return 1.0 if math.copysign(1.0, float(reference)) == math.copysign(1.0, float(candidate)) else 0.0


def _evidence_direction_alignment(event: dict[str, Any], full_mu: float) -> float:
    """Check whether observable financial/sentiment evidence supports the forecast sign."""

    sue, momentum, _implied_vol = _feature_triplet(event)
    avg_sentiment, _message_volume = _sentiment_pair(event)
    signals = [sue, momentum, avg_sentiment]
    scored_signals = [
        _direction_match_score(full_mu, signal)
        for signal in signals
        if abs(float(signal)) > 1e-6
    ]
    if not scored_signals:
        return 0.5
    return float(np.mean(scored_signals))


def _dominant_modality_alignment(
    full_output: dict[str, float],
    text_output: dict[str, float],
    financial_output: dict[str, float],
    sentiment_output: dict[str, float],
) -> tuple[float, str, float]:
    """Check whether the explanation's dominant modality agrees with the full forecast."""

    modality_names = ("text", "financial", "sentiment")
    strengths = [
        float(full_output.get("text_strength", 0.0)),
        float(full_output.get("financial_strength", 0.0)),
        float(full_output.get("sentiment_strength", 0.0)),
    ]
    if not any(math.isfinite(value) and value > 0.0 for value in strengths):
        return 0.5, "unknown", 0.0

    strength_array = np.asarray(strengths, dtype=float)
    dominant_index = int(np.argmax(strength_array))
    total_strength = float(np.sum(np.maximum(strength_array, 0.0)))
    dominance_margin = 0.0
    if total_strength > 1e-8:
        sorted_strengths = np.sort(strength_array)
        dominance_margin = float((sorted_strengths[-1] - sorted_strengths[-2]) / total_strength)

    branch_outputs = (text_output, financial_output, sentiment_output)
    dominant_mu = float(branch_outputs[dominant_index]["mu"])
    return (
        _direction_match_score(float(full_output["mu"]), dominant_mu),
        modality_names[dominant_index],
        _bounded_unit(dominance_margin),
    )


def _build_explanation_metadata(
    events: list[dict[str, Any]],
    full_outputs: list[dict[str, float]],
    text_outputs: list[dict[str, float]],
    financial_outputs: list[dict[str, float]],
    sentiment_outputs: list[dict[str, float]],
    reference_center: float | None = None,
    reference_scale: float | None = None,
) -> tuple[list[dict[str, float | str]], float, float]:
    """Build metadata with structurally grounded explanation-confidence signals."""

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
            float(text_output["mu"]),
            float(financial_output["mu"]),
            float(sentiment_output["mu"]),
            float(full_output["mu"]),
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

    metadata_rows: list[dict[str, float | str]] = []
    for (
        event,
        disagreement,
        full_output,
        text_output,
        financial_output,
        sentiment_output,
    ) in zip(
        events,
        disagreements,
        full_outputs,
        text_outputs,
        financial_outputs,
        sentiment_outputs,
    ):
        base_metadata = _event_calibration_metadata(event)
        disagreement_confidence = _explanation_confidence_from_disagreement(
            disagreement=disagreement,
            center=center,
            scale=scale,
        )
        model_confidence = _bounded_unit(float(full_output.get("introspective_score", 0.5)))
        modality_consistency = _bounded_unit(float(full_output.get("modality_consistency", 0.5)))
        variance_confidence = _bounded_unit(float(full_output.get("variance_confidence", 0.5)))
        dominant_alignment, dominant_modality, dominance_margin = _dominant_modality_alignment(
            full_output=full_output,
            text_output=text_output,
            financial_output=financial_output,
            sentiment_output=sentiment_output,
        )
        evidence_alignment = _evidence_direction_alignment(
            event=event,
            full_mu=float(full_output["mu"]),
        )
        explanation_grounding_score = _bounded_unit(
            0.30 * model_confidence
            + 0.20 * disagreement_confidence
            + 0.15 * modality_consistency
            + 0.15 * variance_confidence
            + 0.10 * dominant_alignment
            + 0.10 * evidence_alignment
        )
        metadata_rows.append(
            {
                **base_metadata,
                "modality_disagreement": float(disagreement),
                "disagreement_confidence": float(disagreement_confidence),
                "model_confidence": model_confidence,
                "modality_consistency": modality_consistency,
                "variance_confidence": variance_confidence,
                "dominant_modality_alignment": float(dominant_alignment),
                "dominant_modality": dominant_modality,
                "dominance_margin": float(dominance_margin),
                "evidence_direction_alignment": float(evidence_alignment),
                "explanation_grounding_score": float(explanation_grounding_score),
                "explanation_confidence": float(explanation_grounding_score),
            }
        )
    return metadata_rows, center, scale


def _to_float(value: Any) -> float:
    """Convert tensor-like or scalar-like values to float."""

    try:
        return float(value.item())
    except AttributeError:
        return float(value)


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
    """Build global non-regime thresholds for a naive conformal baseline."""

    scores: list[float] = []
    for output, label in zip(cal_outputs, cal_labels):
        mu = _to_float(output["mu"])
        log_sigma = _to_float(output["log_sigma"])
        scores.append(abs(float(label) - mu) / math.exp(log_sigma))

    thresholds: dict[float, float] = {}
    n = len(scores)
    for coverage in coverage_levels:
        quantile_level = coverage * (1.0 + 1.0 / n)
        thresholds[float(coverage)] = _conformal_quantile(scores, quantile_level)
    return thresholds


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
    attention_values = batch_outputs.get("attention_stability")
    variance_values = batch_outputs.get("variance_confidence")
    modality_values = batch_outputs.get("modality_consistency")
    modality_strength_values = batch_outputs.get("modality_strengths")
    attention_list = (
        attention_values.cpu().tolist()
        if attention_values is not None
        else [0.0 for _ in batch_outputs["mu"].cpu().tolist()]
    )
    variance_list = (
        variance_values.cpu().tolist()
        if variance_values is not None
        else [0.0 for _ in batch_outputs["mu"].cpu().tolist()]
    )
    modality_list = (
        modality_values.cpu().tolist()
        if modality_values is not None
        else [0.0 for _ in batch_outputs["mu"].cpu().tolist()]
    )
    modality_strength_list = (
        modality_strength_values.cpu().tolist()
        if modality_strength_values is not None
        else [[0.0, 0.0, 0.0] for _ in batch_outputs["mu"].cpu().tolist()]
    )
    explanations = list(batch_outputs.get("explanations", []))

    for index, (
        mu,
        log_sigma,
        score,
        attention,
        variance_confidence,
        modality_consistency,
        modality_strengths,
    ) in enumerate(zip(
        batch_outputs["mu"].cpu().tolist(),
        batch_outputs["log_sigma"].cpu().tolist(),
        batch_outputs["introspective_score"].cpu().tolist(),
        attention_list,
        variance_list,
        modality_list,
        modality_strength_list,
    )):
        strength_values = [float(value) for value in list(modality_strengths)[:3]]
        while len(strength_values) < 3:
            strength_values.append(0.0)
        outputs.append(
            {
                "mu": float(mu),
                "log_sigma": float(log_sigma),
                "introspective_score": float(score),
                "attention_stability": float(attention),
                "variance_confidence": float(variance_confidence),
                "modality_consistency": float(modality_consistency),
                "text_strength": strength_values[0],
                "financial_strength": strength_values[1],
                "sentiment_strength": strength_values[2],
                "explanation": str(explanations[index]) if index < len(explanations) else "",
            }
        )
    return outputs, [float(value) for value in labels.cpu().tolist()], regimes, tickers


def _outputs_from_mu_sigma(
    mu_values: np.ndarray,
    sigma: float | np.ndarray,
    confidence: float = 0.5,
) -> list[dict[str, float]]:
    """Serialize baseline predictions into the common Gaussian output shape."""

    sigma_values = (
        np.full(len(mu_values), float(sigma), dtype=float)
        if np.isscalar(sigma)
        else np.asarray(sigma, dtype=float)
    )
    outputs: list[dict[str, float]] = []
    for mu, sigma_value in zip(mu_values, sigma_values):
        outputs.append(
            {
                "mu": float(mu),
                "log_sigma": float(math.log(max(float(sigma_value), 1e-6))),
                "introspective_score": float(confidence),
            }
        )
    return outputs


def _residual_sigma(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Estimate a robust residual scale for baseline intervals."""

    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    if residuals.size < 2:
        return 0.02
    sigma = float(np.std(residuals, ddof=1))
    return sigma if math.isfinite(sigma) and sigma > 1e-6 else 0.02


def _history_group_mean_predictions(
    train_events: list[dict[str, Any]],
    target_events: list[dict[str, Any]],
    sector_lookup: dict[str, str],
    group_key: str | None,
) -> tuple[np.ndarray, float]:
    """Predict target returns with historical global, sector, or ticker means."""

    train_labels = _event_labels(train_events)
    global_mu = float(np.mean(train_labels)) if len(train_labels) else 0.0
    grouped: dict[str, list[float]] = {}

    for event in train_events:
        if group_key == "ticker":
            key = str(event.get("ticker", "")).upper()
        elif group_key == "sector":
            key = _event_sector(event, sector_lookup)
        else:
            key = "market"
        grouped.setdefault(key, []).append(float(event.get("label", 0.0)))

    target_mu: list[float] = []
    for event in target_events:
        if group_key == "ticker":
            key = str(event.get("ticker", "")).upper()
        elif group_key == "sector":
            key = _event_sector(event, sector_lookup)
        else:
            key = "market"
        values = grouped.get(key, [])
        target_mu.append(float(np.mean(values)) if values else global_mu)

    train_mu = []
    for event in train_events:
        if group_key == "ticker":
            key = str(event.get("ticker", "")).upper()
        elif group_key == "sector":
            key = _event_sector(event, sector_lookup)
        else:
            key = "market"
        values = grouped.get(key, [])
        train_mu.append(float(np.mean(values)) if values else global_mu)

    return np.asarray(target_mu, dtype=float), _residual_sigma(train_labels, np.asarray(train_mu))


def _make_tabular_pipeline(
    method: str,
    numeric_features: list[str],
    categorical_features: list[str] | None = None,
) -> Pipeline:
    """Create a tabular baseline estimator with preprocessing."""

    categorical_features = list(categorical_features or [])
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        transformers.append(("numeric", StandardScaler(), numeric_features))
    if categorical_features:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # pragma: no cover - older scikit-learn compatibility
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(
            (
                "categorical",
                encoder,
                categorical_features,
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    if method == "gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            max_iter=160,
            learning_rate=0.04,
            l2_regularization=0.01,
            random_state=42,
        )
    else:
        estimator = Ridge(alpha=5.0)
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def _fit_predict_tabular_baseline(
    method: str,
    train_events: list[dict[str, Any]],
    target_events: list[dict[str, Any]],
    sector_lookup: dict[str, str],
    numeric_features: list[str],
    categorical_features: list[str] | None = None,
) -> tuple[np.ndarray, float]:
    """Fit a tabular baseline on historical events and predict target events."""

    train_frame = _events_to_frame(train_events, sector_lookup)
    target_frame = _events_to_frame(target_events, sector_lookup)
    y_train = _event_labels(train_events)
    if len(train_frame) < 8:
        fallback_mu = float(np.mean(y_train)) if len(y_train) else 0.0
        return np.full(len(target_events), fallback_mu), 0.02

    pipeline = _make_tabular_pipeline(
        method=method,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    pipeline.fit(train_frame, y_train)
    train_pred = np.asarray(pipeline.predict(train_frame), dtype=float)
    target_pred = np.asarray(pipeline.predict(target_frame), dtype=float)
    return target_pred, _residual_sigma(y_train, train_pred)


def _compute_strong_baseline(
    method: str,
    train_events: list[dict[str, Any]],
    target_events: list[dict[str, Any]],
    regime_thresholds: dict[str, float],
    sector_lookup: dict[str, str],
) -> tuple[list[dict[str, float]], list[float], list[str], list[str]]:
    """Compute one paper-oriented non-neural baseline."""

    if method == "market_return":
        mu_values, sigma = _history_group_mean_predictions(
            train_events, target_events, sector_lookup, group_key=None
        )
    elif method == "sector_return":
        mu_values, sigma = _history_group_mean_predictions(
            train_events, target_events, sector_lookup, group_key="sector"
        )
    elif method == "post_earnings_drift":
        mu_values, sigma = _history_group_mean_predictions(
            train_events, target_events, sector_lookup, group_key="ticker"
        )
    elif method == "analyst_surprise":
        mu_values, sigma = _fit_predict_tabular_baseline(
            method="ridge",
            train_events=train_events,
            target_events=target_events,
            sector_lookup=sector_lookup,
            numeric_features=["sue", "abs_sue"],
        )
    elif method == "implied_vol_only":
        mu_values, sigma = _fit_predict_tabular_baseline(
            method="ridge",
            train_events=train_events,
            target_events=target_events,
            sector_lookup=sector_lookup,
            numeric_features=["implied_vol"],
        )
    elif method == "linear_ridge":
        mu_values, sigma = _fit_predict_tabular_baseline(
            method="ridge",
            train_events=train_events,
            target_events=target_events,
            sector_lookup=sector_lookup,
            numeric_features=[
                "sue",
                "abs_sue",
                "momentum",
                "implied_vol",
                "avg_sentiment",
                "message_volume",
                "transcript_length",
            ],
        )
    elif method == "ticker_fixed_effects":
        mu_values, sigma = _fit_predict_tabular_baseline(
            method="ridge",
            train_events=train_events,
            target_events=target_events,
            sector_lookup=sector_lookup,
            numeric_features=["sue", "momentum", "implied_vol"],
            categorical_features=["ticker"],
        )
    elif method == "xgboost_lightgbm_proxy":
        mu_values, sigma = _fit_predict_tabular_baseline(
            method="gradient_boosting",
            train_events=train_events,
            target_events=target_events,
            sector_lookup=sector_lookup,
            numeric_features=[
                "sue",
                "abs_sue",
                "momentum",
                "implied_vol",
                "avg_sentiment",
                "message_volume",
                "transcript_length",
            ],
            categorical_features=["sector"],
        )
    else:
        raise ValueError(f"Unsupported baseline method: {method}")

    labels = [float(event.get("label", 0.0)) for event in target_events]
    regimes = [
        assign_regime(
            sue=_event_sue(event),
            implied_vol=_event_implied_vol(event),
            thresholds=regime_thresholds,
        )
        for event in target_events
    ]
    tickers = [str(event.get("ticker", "")).upper() for event in target_events]
    return _outputs_from_mu_sigma(mu_values, sigma), labels, regimes, tickers


def _baseline_global_thresholds(
    outputs: list[dict[str, float]],
    labels: list[float],
    coverage_levels: list[float],
) -> dict[float, float]:
    """Build per-baseline conformal thresholds from calibration residuals."""

    return _build_global_thresholds(outputs, labels, coverage_levels)


def _predict_interval_without_adjustment(
    predictor: EventConditionedConformalPredictor,
    output: dict[str, Any],
    regime: str,
    coverage: float,
) -> tuple[float, float]:
    """Predict a regime-aware interval without introspective widening."""

    normalized_regime = regime
    threshold = predictor.thresholds.get((normalized_regime, float(coverage)))
    if threshold is None:
        threshold = _fallback_threshold(predictor, coverage)
    mu = _to_float(output["mu"])
    sigma = math.exp(_to_float(output["log_sigma"]))
    half_width = threshold * sigma
    return mu - half_width, mu + half_width


def _predict_interval_naive(
    output: dict[str, Any],
    coverage: float,
    global_thresholds: dict[float, float],
) -> tuple[float, float]:
    """Predict a global conformal interval without introspective widening."""

    mu = _to_float(output["mu"])
    sigma = math.exp(_to_float(output["log_sigma"]))
    half_width = global_thresholds[float(coverage)] * sigma
    return mu - half_width, mu + half_width


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
) -> dict[str, float | str]:
    """Compute evaluation metrics for one method row."""

    coverages = [0.80, 0.90, 0.95]
    interval_hits: dict[float, list[float]] = {coverage: [] for coverage in coverages}
    interval_widths: dict[float, list[float]] = {coverage: [] for coverage in coverages}

    mu_values = [float(output["mu"]) for output in outputs]
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

    for output, label, regime, event_metadata in zip(outputs, labels, regimes, metadata):
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
                    mu = _to_float(output["mu"])
                    sigma = math.exp(_to_float(output["log_sigma"]))
                    half_width = threshold * sigma
                    lower, upper = mu - half_width, mu + half_width
            elif mode == "naive":
                lower, upper = _predict_interval_naive(output, coverage, global_thresholds)
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
        "coverage_error_80": float(np.mean(interval_hits[0.80]) - 0.80),
        "coverage_error_90": float(np.mean(interval_hits[0.90]) - 0.90),
        "coverage_error_95": float(np.mean(interval_hits[0.95]) - 0.95),
        "mean_abs_coverage_error": float(
            np.mean(
                [
                    abs(np.mean(interval_hits[0.80]) - 0.80),
                    abs(np.mean(interval_hits[0.90]) - 0.90),
                    abs(np.mean(interval_hits[0.95]) - 0.95),
                ]
            )
        ),
        "avg_width": float(np.mean([avg_width_80, avg_width_90, avg_width_95])),
        "avg_width_80": avg_width_80,
        "avg_width_90": avg_width_90,
        "avg_width_95": avg_width_95,
        "MAE": mae,
        "RMSE": rmse,
        "dir_acc": dir_acc,
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
    events: list[dict[str, Any]] | None = None,
    sector_lookup: dict[str, str] | None = None,
) -> list[dict[str, float | str | int]]:
    """Compute subgroup metrics by surprise, volatility, attention, and sector."""

    subgroup_members: dict[tuple[str, str], list[int]] = {}
    for index, regime in enumerate(regimes):
        surprise_band, volatility_band = _regime_components(regime)
        subgroup_members.setdefault(("surprise_band", surprise_band), []).append(index)
        subgroup_members.setdefault(("volatility_band", volatility_band), []).append(index)
    if events is not None:
        sector_lookup = sector_lookup or {}
        for index, event in enumerate(events):
            subgroup_members.setdefault(
                ("ticker_sector", _event_sector(event, sector_lookup)),
                [],
            ).append(index)

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
) -> list[dict[str, float | str | int]]:
    """Build per-event prediction rows for export and lightweight inspection."""

    rows: list[dict[str, float | str | int]] = []
    if metadata is None:
        metadata = [_event_calibration_metadata(event) for event in events]
    if financial_lookup is None:
        financial_lookup = {}

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
                lower, upper = predictor.predict_interval(
                    output=output,
                    regime=regime,
                    coverage=coverage,
                    metadata=event_metadata,
                )
            elif mode == "naive":
                lower, upper = _predict_interval_naive(output, coverage, global_thresholds)
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
        mu = float(output["mu"])
        sigma = float(math.exp(float(output["log_sigma"])))
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
                "prediction_error": mu - float(label),
                "sigma": sigma,
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
                "explanation_confidence": float(
                    event_metadata.get(
                        "explanation_confidence",
                        output.get("introspective_score", 0.0),
                    )
                ),
                "explanation_grounding_score": float(
                    event_metadata.get(
                        "explanation_grounding_score",
                        event_metadata.get(
                            "explanation_confidence",
                            output.get("introspective_score", 0.0),
                        ),
                    )
                ),
                "modality_disagreement": float(
                    event_metadata.get("modality_disagreement", 0.0)
                ),
                "disagreement_confidence": float(
                    event_metadata.get("disagreement_confidence", 0.0)
                ),
                "modality_consistency": float(
                    event_metadata.get(
                        "modality_consistency",
                        output.get("modality_consistency", 0.0),
                    )
                ),
                "variance_confidence": float(
                    event_metadata.get(
                        "variance_confidence",
                        output.get("variance_confidence", 0.0),
                    )
                ),
                "dominant_modality": str(event_metadata.get("dominant_modality", "")),
                "dominant_modality_alignment": float(
                    event_metadata.get("dominant_modality_alignment", 0.0)
                ),
                "dominance_margin": float(event_metadata.get("dominance_margin", 0.0)),
                "evidence_direction_alignment": float(
                    event_metadata.get("evidence_direction_alignment", 0.0)
                ),
                "generated_explanation": str(output.get("explanation", "")),
                "direction_match": int(np.sign(mu) == np.sign(float(label))),
                "transcript_preview": transcript_preview,
            }
        )
    return rows


def _selective_metric_rows(
    outputs: list[dict[str, float]],
    labels: list[float],
    regimes: list[str],
    predictor: EventConditionedConformalPredictor,
    min_scores: list[float],
    coverage: float = 0.90,
    metadata: list[dict[str, float]] | None = None,
) -> list[dict[str, float | str]]:
    """Compute a lightweight selective prediction risk-coverage table."""

    rows: list[dict[str, float | str]] = []
    if metadata is None:
        metadata = [{} for _ in outputs]
    for min_score in min_scores:
        kept_indices: list[int] = []
        interval_widths: list[float] = []
        interval_hits: list[float] = []
        for index, (output, label, regime, event_metadata) in enumerate(
            zip(outputs, labels, regimes, metadata)
        ):
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
        mu_values = [float(output["mu"]) for output in kept_outputs]
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


def _wilson_interval(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson confidence interval for a binomial proportion."""

    if n <= 0:
        return float("nan"), float("nan")
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n))) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def _bootstrap_metric_ci(
    errors: np.ndarray,
    metric: str,
    rng: np.random.Generator,
    n_bootstrap: int = 500,
) -> tuple[float, float]:
    """Bootstrap a confidence interval for MAE or RMSE from per-event errors."""

    if errors.size == 0:
        return float("nan"), float("nan")
    values: list[float] = []
    indices = np.arange(errors.size)
    for _ in range(max(int(n_bootstrap), 20)):
        sample = errors[rng.choice(indices, size=errors.size, replace=True)]
        if metric == "rmse":
            values.append(float(np.sqrt(np.mean(np.square(sample)))))
        else:
            values.append(float(np.mean(np.abs(sample))))
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def _diebold_mariano_pvalue(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
) -> float:
    """Compute a simple paired Diebold-Mariano p-value with squared-error loss."""

    if errors_a.size != errors_b.size or errors_a.size < 3:
        return float("nan")
    loss_diff = np.square(errors_a) - np.square(errors_b)
    diff_std = float(np.std(loss_diff, ddof=1))
    if diff_std <= 1e-12:
        return 1.0
    dm_stat = float(np.mean(loss_diff) / (diff_std / math.sqrt(loss_diff.size)))
    return float(2.0 * stats.t.sf(abs(dm_stat), df=loss_diff.size - 1))


def _significance_rows(
    prediction_rows: list[dict[str, float | str | int]],
    baseline_method: str,
    n_bootstrap: int = 500,
) -> list[dict[str, float | str | int]]:
    """Build bootstrap, paired-comparison, and direction-accuracy statistics."""

    table = pd.DataFrame(prediction_rows)
    required = {"method", "ticker", "date", "actual_return", "predicted_return"}
    if table.empty or not required.issubset(table.columns):
        return []

    rows: list[dict[str, float | str | int]] = []
    rng = np.random.default_rng(42)
    baseline = table.loc[table["method"] == baseline_method].copy()
    baseline_keyed = baseline.set_index(["ticker", "date"]) if not baseline.empty else pd.DataFrame()

    for method, method_frame in table.groupby("method"):
        method_frame = method_frame.copy()
        errors = (
            method_frame["predicted_return"].astype(float).to_numpy()
            - method_frame["actual_return"].astype(float).to_numpy()
        )
        mae_low, mae_high = _bootstrap_metric_ci(errors, "mae", rng, n_bootstrap)
        rmse_low, rmse_high = _bootstrap_metric_ci(errors, "rmse", rng, n_bootstrap)
        direction_hits = (
            np.sign(method_frame["predicted_return"].astype(float).to_numpy())
            == np.sign(method_frame["actual_return"].astype(float).to_numpy())
        )
        dir_low, dir_high = _wilson_interval(int(direction_hits.sum()), int(direction_hits.size))

        paired_mae_delta = float("nan")
        dm_pvalue = float("nan")
        if not baseline_keyed.empty and method != baseline_method:
            joined = method_frame.set_index(["ticker", "date"]).join(
                baseline_keyed[["actual_return", "predicted_return"]],
                rsuffix="_baseline",
                how="inner",
            )
            if not joined.empty:
                method_errors = (
                    joined["predicted_return"].astype(float).to_numpy()
                    - joined["actual_return"].astype(float).to_numpy()
                )
                baseline_errors = (
                    joined["predicted_return_baseline"].astype(float).to_numpy()
                    - joined["actual_return"].astype(float).to_numpy()
                )
                paired_mae_delta = float(
                    np.mean(np.abs(method_errors) - np.abs(baseline_errors))
                )
                dm_pvalue = _diebold_mariano_pvalue(method_errors, baseline_errors)

        rows.append(
            {
                "method": str(method),
                "baseline_method": baseline_method,
                "n": int(errors.size),
                "mae_ci_low": mae_low,
                "mae_ci_high": mae_high,
                "rmse_ci_low": rmse_low,
                "rmse_ci_high": rmse_high,
                "dir_acc_ci_low": dir_low,
                "dir_acc_ci_high": dir_high,
                "paired_mae_delta_vs_baseline": paired_mae_delta,
                "dm_pvalue_vs_baseline": dm_pvalue,
            }
        )
    return rows


def _explanation_diagnostic_rows(
    outputs: list[dict[str, float]],
    labels: list[float],
    metadata: list[dict[str, float | str]],
    predictor: EventConditionedConformalPredictor,
    regimes: list[str],
) -> list[dict[str, float | str | int]]:
    """Score generated explanations and test confidence-residual relationships."""

    if not outputs:
        return []
    residuals = np.asarray(
        [abs(float(output["mu"]) - float(label)) for output, label in zip(outputs, labels)],
        dtype=float,
    )
    confidences = np.asarray(
        [
            float(event_metadata.get("explanation_confidence", output.get("introspective_score", 0.5)))
            for output, event_metadata in zip(outputs, metadata)
        ],
        dtype=float,
    )
    rubric_scores: list[float] = []
    for output, event_metadata in zip(outputs, metadata):
        explanation = str(output.get("explanation", ""))
        score = 0.0
        score += 0.25 if ("positive" in explanation or "negative" in explanation) else 0.0
        score += 0.25 if ("uncertainty" in explanation or "variance" in explanation) else 0.0
        score += 0.25 if any(token in explanation for token in ("transcript", "financial", "sentiment")) else 0.0
        score += 0.25 * float(event_metadata.get("evidence_direction_alignment", 0.5))
        rubric_scores.append(float(score))

    corr = float(stats.spearmanr(confidences, residuals).correlation) if len(outputs) >= 3 else float("nan")
    rubric_corr = (
        float(stats.spearmanr(np.asarray(rubric_scores), residuals).correlation)
        if len(outputs) >= 3
        else float("nan")
    )

    shuffled_metadata = [dict(row) for row in metadata]
    rng = np.random.default_rng(42)
    shuffled_confidence = np.asarray(confidences, dtype=float).copy()
    rng.shuffle(shuffled_confidence)
    for row, value in zip(shuffled_metadata, shuffled_confidence):
        row["explanation_confidence"] = float(value)
        row["explanation_grounding_score"] = float(value)

    real_widths: list[float] = []
    shuffled_widths: list[float] = []
    for output, regime, real_metadata, permuted_metadata in zip(
        outputs, regimes, metadata, shuffled_metadata
    ):
        lo, hi = predictor.predict_interval(output, regime, coverage=0.90, metadata=real_metadata)
        real_widths.append(float(hi - lo))
        lo_perm, hi_perm = predictor.predict_interval(
            output,
            regime,
            coverage=0.90,
            metadata=permuted_metadata,
        )
        shuffled_widths.append(float(hi_perm - lo_perm))

    low_cut = float(np.quantile(confidences, 0.25)) if len(confidences) else 0.0
    high_cut = float(np.quantile(confidences, 0.75)) if len(confidences) else 1.0
    low_mask = confidences <= low_cut
    high_mask = confidences >= high_cut

    return [
        {
            "diagnostic": "confidence_residual_spearman",
            "n": int(len(outputs)),
            "value": corr,
            "expected_direction": "negative",
        },
        {
            "diagnostic": "rubric_residual_spearman",
            "n": int(len(outputs)),
            "value": rubric_corr,
            "expected_direction": "negative",
        },
        {
            "diagnostic": "low_confidence_mae",
            "n": int(np.sum(low_mask)),
            "value": float(np.mean(residuals[low_mask])) if np.any(low_mask) else float("nan"),
            "expected_direction": "higher_than_high_confidence",
        },
        {
            "diagnostic": "high_confidence_mae",
            "n": int(np.sum(high_mask)),
            "value": float(np.mean(residuals[high_mask])) if np.any(high_mask) else float("nan"),
            "expected_direction": "lower_than_low_confidence",
        },
        {
            "diagnostic": "real_minus_permuted_interval_width_90",
            "n": int(len(outputs)),
            "value": float(np.mean(real_widths) - np.mean(shuffled_widths)),
            "expected_direction": "nonzero_if_explanation_signal_matters",
        },
        {
            "diagnostic": "avg_generated_explanation_rubric",
            "n": int(len(outputs)),
            "value": float(np.mean(rubric_scores)),
            "expected_direction": "higher_is_better",
        },
    ]


def _rolling_backtest_rows(
    events: list[dict[str, Any]],
    config: dict[str, Any],
    sector_lookup: dict[str, str],
    baseline_methods: list[str],
) -> list[dict[str, float | str | int]]:
    """Run rolling yearly backtests for strong tabular/historical baselines."""

    rolling_config = config.get("evaluation", {}).get("rolling_backtest", {})
    if not bool(rolling_config.get("enabled", True)):
        return []
    years = sorted({int(event.get("year", 0)) for event in events if int(event.get("year", 0)) > 0})
    requested_years = rolling_config.get("test_years")
    if requested_years:
        years = [int(year) for year in requested_years if int(year) in years]

    min_train_years = int(rolling_config.get("min_train_years", 2))
    rows: list[dict[str, float | str | int]] = []
    for test_year in years:
        cal_year = test_year - 1
        train_events = [event for event in events if int(event.get("year", 0)) < cal_year]
        cal_events = [event for event in events if int(event.get("year", 0)) == cal_year]
        test_events = [event for event in events if int(event.get("year", 0)) == test_year]
        if len({int(event.get("year", 0)) for event in train_events}) < min_train_years:
            continue
        if not train_events or not cal_events or not test_events:
            continue
        thresholds = fit_regime_thresholds(train_events)
        coverage_levels = list(config.get("calibration", {}).get("coverage_levels", [0.80, 0.90, 0.95]))
        for method in baseline_methods:
            outputs, labels, regimes, _tickers = _compute_strong_baseline(
                method=method,
                train_events=train_events,
                target_events=test_events,
                regime_thresholds=thresholds,
                sector_lookup=sector_lookup,
            )
            cal_outputs, cal_labels, _cal_regimes, _cal_tickers = _compute_strong_baseline(
                method=method,
                train_events=train_events,
                target_events=cal_events,
                regime_thresholds=thresholds,
                sector_lookup=sector_lookup,
            )
            global_thresholds = _baseline_global_thresholds(cal_outputs, cal_labels, coverage_levels)
            row = _metric_row(
                method=method,
                outputs=outputs,
                labels=labels,
                regimes=regimes,
                predictor=EventConditionedConformalPredictor(coverage_levels=coverage_levels),
                global_thresholds=global_thresholds,
                mode="naive",
            )
            row["test_year"] = int(test_year)
            row["train_years"] = ",".join(str(year) for year in sorted({int(e.get("year", 0)) for e in train_events}))
            row["calibration_year"] = int(cal_year)
            row["n_test"] = int(len(test_events))
            rows.append(row)
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
    cal_dataset = EarningsDataset(cal_events, feature_stats=feature_stats)
    test_dataset = EarningsDataset(test_events, feature_stats=feature_stats)
    financial_lookup = _load_financial_lookup()
    sector_lookup = _load_ticker_sector_lookup()

    device = _select_device(config)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    model_path = PROJECT_ROOT / "experiments" / "model_best.pt" if (PROJECT_ROOT / "experiments" / "model_best.pt").exists() else PROJECT_ROOT / "experiments" / "model.pt"
    if not model_path.is_file():
        model_path = PROJECT_ROOT / "model.pt"

    model = MultimodalForecastModel.load_from_config(str(config_file))
    state_dict = torch.load(model_path, map_location="cpu")
    load_result = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(getattr(load_result, "missing_keys", []))
    unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
    if missing_keys or unexpected_keys:
        print(
            "Warning: loaded checkpoint with partial compatibility. "
            f"missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}"
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

    coverage_levels = list(config.get("calibration", {}).get("coverage_levels", [0.80, 0.90, 0.95]))
    event_conditioned_predictor = EventConditionedConformalPredictor(
        coverage_levels=coverage_levels,
        use_attention_conditioning=True,
        use_explanation_adjustment=False,
    )
    explanation_augmented_predictor = EventConditionedConformalPredictor(
        coverage_levels=coverage_levels,
        use_attention_conditioning=True,
        use_explanation_adjustment=True,
    )
    cal_metadata, disagreement_center, disagreement_scale = _build_explanation_metadata(
        events=cal_events,
        full_outputs=cal_outputs,
        text_outputs=cal_text_outputs,
        financial_outputs=cal_financial_outputs,
        sentiment_outputs=cal_sentiment_outputs,
    )
    event_conditioned_predictor.calibrate(
        cal_outputs=cal_outputs,
        cal_labels=cal_labels,
        cal_regimes=cal_regimes,
        cal_metadata=cal_metadata,
    )
    explanation_augmented_predictor.calibrate(
        cal_outputs=cal_outputs,
        cal_labels=cal_labels,
        cal_regimes=cal_regimes,
        cal_metadata=cal_metadata,
    )
    global_thresholds = _build_global_thresholds(cal_outputs, cal_labels, coverage_levels)
    historical_events = train_events + val_events + cal_events

    results_rows: list[dict[str, float | str]] = []
    subgroup_rows: list[dict[str, float | str | int]] = []
    selective_rows: list[dict[str, float | str]] = []
    prediction_rows: list[dict[str, float | str | int]] = []
    significance_rows: list[dict[str, float | str | int]] = []
    explanation_diagnostic_rows: list[dict[str, float | str | int]] = []
    rolling_rows: list[dict[str, float | str | int]] = []
    include_selective_analysis = bool(
        config.get("evaluation", {}).get("include_selective_analysis", False)
    )
    include_modality_ablations = bool(
        config.get("evaluation", {}).get("include_modality_ablations", False)
    )
    baseline_methods = list(
        config.get("evaluation", {}).get(
            "strong_baselines",
            [
                "market_return",
                "sector_return",
                "post_earnings_drift",
                "analyst_surprise",
                "implied_vol_only",
                "linear_ridge",
                "ticker_fixed_effects",
                "xgboost_lightgbm_proxy",
            ],
        )
    )

    evaluation_specs: list[tuple[str, str, EventConditionedConformalPredictor]] = []
    if include_modality_ablations:
        evaluation_specs.extend(
            [
                ("text_only", "regime_no_introspection", event_conditioned_predictor),
                ("financial_only", "regime_no_introspection", event_conditioned_predictor),
                ("sentiment_only", "regime_no_introspection", event_conditioned_predictor),
            ]
        )
    evaluation_specs.extend([
        ("full_multimodal", "regime_no_introspection", event_conditioned_predictor),
        ("naive_conformal", "naive", event_conditioned_predictor),
        ("ours", "adaptive", event_conditioned_predictor),
        ("ours_explanation_augmented", "adaptive", explanation_augmented_predictor),
    ])

    for method, mode, _active_predictor in evaluation_specs:
        base_method = "full_multimodal" if method in {"full_multimodal", "naive_conformal", "ours"} else method
        if method == "ours_explanation_augmented":
            base_method = "full_multimodal"
        outputs, labels, regimes, _tickers = _compute_model_outputs(
            model=model,
            events=test_events,
            dataset=test_dataset,
            method=base_method,
            device=device,
            regime_thresholds=regime_thresholds,
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

    if "test_text_outputs" not in locals():
        test_text_outputs, _unused_labels, _unused_regimes, _unused_tickers = _compute_model_outputs(
            model=model,
            events=test_events,
            dataset=test_dataset,
            method="text_only",
            device=device,
            regime_thresholds=regime_thresholds,
        )
    if "test_financial_outputs" not in locals():
        test_financial_outputs, _unused_labels, _unused_regimes, _unused_tickers = _compute_model_outputs(
            model=model,
            events=test_events,
            dataset=test_dataset,
            method="financial_only",
            device=device,
            regime_thresholds=regime_thresholds,
        )
    if "test_sentiment_outputs" not in locals():
        test_sentiment_outputs, _unused_labels, _unused_regimes, _unused_tickers = _compute_model_outputs(
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
        "full_multimodal": (test_full_outputs, test_labels, test_regimes, test_tickers),
        "naive_conformal": (test_full_outputs, test_labels, test_regimes, test_tickers),
        "ours": (test_full_outputs, test_labels, test_regimes, test_tickers),
        "ours_explanation_augmented": (test_full_outputs, test_labels, test_regimes, test_tickers),
    }
    if include_modality_ablations:
        method_payloads.update(
            {
                "text_only": (test_text_outputs, test_labels, test_regimes, test_tickers),
                "financial_only": (test_financial_outputs, test_labels, test_regimes, test_tickers),
                "sentiment_only": (test_sentiment_outputs, test_labels, test_regimes, test_tickers),
            }
        )

    for method, mode, active_predictor in evaluation_specs:
        outputs, labels, regimes, _tickers = method_payloads[method]

        results_rows.append(
            _metric_row(
                method=method,
                outputs=outputs,
                labels=labels,
                regimes=regimes,
                predictor=active_predictor,
                global_thresholds=global_thresholds,
                mode=mode,
                metadata=test_metadata,
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
                global_thresholds=global_thresholds,
                mode=mode,
                metadata=test_metadata,
                financial_lookup=financial_lookup,
            )
        )
        if include_selective_analysis and method == "ours_explanation_augmented":
            selective_rows.extend(
                _selective_metric_rows(
                    outputs=outputs,
                    labels=labels,
                    regimes=regimes,
                    predictor=active_predictor,
                    min_scores=[
                        float(score)
                        for score in config.get("calibration", {}).get(
                            "selective_min_scores",
                            [0.55, 0.70, 0.85],
                        )
                    ],
                    metadata=test_metadata,
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
                mode=mode,
                metadata=test_metadata,
                events=test_events,
                sector_lookup=sector_lookup,
            )
        )

    for baseline_method in baseline_methods:
        cal_baseline_outputs, cal_baseline_labels, _cal_baseline_regimes, _ = _compute_strong_baseline(
            method=baseline_method,
            train_events=train_events + val_events,
            target_events=cal_events,
            regime_thresholds=regime_thresholds,
            sector_lookup=sector_lookup,
        )
        baseline_thresholds = _baseline_global_thresholds(
            cal_baseline_outputs,
            cal_baseline_labels,
            coverage_levels,
        )
        baseline_outputs, baseline_labels, baseline_regimes, baseline_tickers = _compute_strong_baseline(
            method=baseline_method,
            train_events=historical_events,
            target_events=test_events,
            regime_thresholds=regime_thresholds,
            sector_lookup=sector_lookup,
        )
        results_rows.append(
            _metric_row(
                method=baseline_method,
                outputs=baseline_outputs,
                labels=baseline_labels,
                regimes=baseline_regimes,
                predictor=event_conditioned_predictor,
                global_thresholds=baseline_thresholds,
                mode="naive",
            )
        )
        prediction_rows.extend(
            _prediction_rows(
                method=baseline_method,
                events=test_events,
                outputs=baseline_outputs,
                labels=baseline_labels,
                regimes=baseline_regimes,
                tickers=baseline_tickers,
                predictor=event_conditioned_predictor,
                global_thresholds=baseline_thresholds,
                mode="naive",
                financial_lookup=financial_lookup,
            )
        )
        subgroup_rows.extend(
            _subgroup_metric_rows(
                method=baseline_method,
                outputs=baseline_outputs,
                labels=baseline_labels,
                regimes=baseline_regimes,
                predictor=event_conditioned_predictor,
                global_thresholds=baseline_thresholds,
                mode="naive",
                events=test_events,
                sector_lookup=sector_lookup,
            )
        )

    explanation_diagnostic_rows = _explanation_diagnostic_rows(
        outputs=test_full_outputs,
        labels=test_labels,
        metadata=test_metadata,
        predictor=explanation_augmented_predictor,
        regimes=test_regimes,
    )
    significance_rows = _significance_rows(
        prediction_rows=prediction_rows,
        baseline_method=str(config.get("evaluation", {}).get("primary_baseline", "linear_ridge")),
        n_bootstrap=int(config.get("evaluation", {}).get("bootstrap_samples", 500)),
    )
    rolling_rows = _rolling_backtest_rows(
        events=all_events,
        config=config,
        sector_lookup=sector_lookup,
        baseline_methods=baseline_methods,
    )

    results_table = pd.DataFrame(results_rows)
    subgroup_results_table = pd.DataFrame(subgroup_rows)
    selective_results_table = pd.DataFrame(selective_rows)
    predictions_table = pd.DataFrame(prediction_rows)
    significance_table = pd.DataFrame(significance_rows)
    explanation_diagnostics_table = pd.DataFrame(explanation_diagnostic_rows)
    rolling_results_table = pd.DataFrame(rolling_rows)
    display_columns = [
        "method",
        "coverage_80",
        "coverage_90",
        "coverage_95",
        "mean_abs_coverage_error",
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
    if not significance_table.empty:
        print(
            tabulate(
                significance_table.head(12),
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
    experiments_significance_path = PROJECT_ROOT / "experiments" / "results_significance.csv"
    root_significance_path = PROJECT_ROOT / "results_significance.csv"
    experiments_explanation_diagnostics_path = PROJECT_ROOT / "experiments" / "explanation_diagnostics.csv"
    root_explanation_diagnostics_path = PROJECT_ROOT / "explanation_diagnostics.csv"
    experiments_rolling_results_path = PROJECT_ROOT / "experiments" / "results_rolling.csv"
    root_rolling_results_path = PROJECT_ROOT / "results_rolling.csv"
    results_table.to_csv(experiments_results_path, index=False)
    results_table.to_csv(root_results_path, index=False)
    subgroup_results_table.to_csv(experiments_subgroup_results_path, index=False)
    subgroup_results_table.to_csv(root_subgroup_results_path, index=False)
    predictions_table.to_csv(experiments_predictions_path, index=False)
    predictions_table.to_csv(root_predictions_path, index=False)
    significance_table.to_csv(experiments_significance_path, index=False)
    significance_table.to_csv(root_significance_path, index=False)
    explanation_diagnostics_table.to_csv(experiments_explanation_diagnostics_path, index=False)
    explanation_diagnostics_table.to_csv(root_explanation_diagnostics_path, index=False)
    rolling_results_table.to_csv(experiments_rolling_results_path, index=False)
    rolling_results_table.to_csv(root_rolling_results_path, index=False)
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
