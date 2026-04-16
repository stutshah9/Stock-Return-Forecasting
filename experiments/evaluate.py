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

from calibration.conformal import EventConditionedConformalPredictor, assign_regime
from data.dataset import EarningsDataset
from data.event_utils import (
    filter_events_by_universe,
    summarize_event_coverage,
)
from models.fusion_model import MultimodalForecastModel


EVENTS_CACHE_PATH = PROJECT_ROOT / "data" / "events_cache.pt"
FEATURE_STATS_PATH = PROJECT_ROOT / "experiments" / "feature_stats.json"


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


def _split_events_by_year(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split cached events into fixed year-based train/calibration/test partitions."""

    sorted_events = sorted(events, key=lambda event: (int(event["year"]), str(event["date"])))
    train_events = [event for event in sorted_events if int(event["year"]) <= 2022]
    cal_events = [
        event for event in sorted_events if int(event["year"]) in {2023, 2024}
    ]
    test_events = [event for event in sorted_events if int(event["year"]) == 2025]
    if not train_events or not cal_events or not test_events:
        raise ValueError(
            "Year-based cache split produced an empty partition. "
            "Expected train years <= 2022, calibration years 2023-2024, "
            "and test year 2025."
        )
    return train_events, cal_events, test_events


def _assert_disjoint_splits(
    train_events: list[dict[str, Any]],
    cal_events: list[dict[str, Any]],
    test_events: list[dict[str, Any]],
) -> None:
    """Ensure train/calibration/test events are strictly disjoint."""

    def _keys(events: list[dict[str, Any]]) -> set[tuple[str, str]]:
        return {
            (str(event.get("ticker", "")).upper(), str(event.get("date", "")))
            for event in events
        }

    train_keys = _keys(train_events)
    cal_keys = _keys(cal_events)
    test_keys = _keys(test_events)
    if train_keys & cal_keys:
        raise ValueError("Training and calibration splits overlap.")
    if train_keys & test_keys:
        raise ValueError("Training and test splits overlap.")
    if cal_keys & test_keys:
        raise ValueError("Calibration and test splits overlap.")


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
) -> tuple[list[dict[str, float]], list[float], list[str], list[str]]:
    """Run a model-based evaluation variant and serialize its outputs."""

    transcripts, financial, sentiment, labels, regimes, tickers = _prepare_modal_inputs(
        events,
        dataset,
        method,
        device,
    )
    model.eval()
    with torch.no_grad():
        batch_outputs = model(
            transcripts=transcripts,
            financial=financial,
            sentiment=sentiment,
        )

    outputs: list[dict[str, float]] = []
    for mu, log_sigma, score in zip(
        batch_outputs["mu"].cpu().tolist(),
        batch_outputs["log_sigma"].cpu().tolist(),
        batch_outputs["introspective_score"].cpu().tolist(),
    ):
        outputs.append(
            {
                "mu": float(mu),
                "log_sigma": float(log_sigma),
                "introspective_score": float(score),
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
                "mu": float(mu),
                "log_sigma": float(math.log(max(sigma, 1e-6))),
                "introspective_score": 1.0,
            }
        )
        labels.append(float(event.get("label", 0.0)))
        regimes.append(
            assign_regime(
                sue=_event_sue(event),
                implied_vol=_event_implied_vol(event),
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


def _metric_row(
    method: str,
    outputs: list[dict[str, float]],
    labels: list[float],
    regimes: list[str],
    predictor: EventConditionedConformalPredictor,
    global_thresholds: dict[float, float],
    mode: str,
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

    for output, label, regime in zip(outputs, labels, regimes):
        for coverage in coverages:
            if mode == "ours":
                try:
                    lower, upper = predictor.predict_interval(
                        output=output,
                        regime=regime,
                        coverage=coverage,
                    )
                except KeyError:
                    threshold = _fallback_threshold(predictor, coverage)
                    mu = _to_float(output["mu"])
                    sigma = math.exp(_to_float(output["log_sigma"]))
                    adjustment = 1.0 + 0.5 * (
                        1.0 - _to_float(output["introspective_score"])
                    )
                    half_width = threshold * sigma * adjustment
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
) -> list[dict[str, float | str | int]]:
    """Compute subgroup metrics by surprise band and volatility regime."""

    subgroup_members: dict[tuple[str, str], list[int]] = {}
    for index, regime in enumerate(regimes):
        surprise_band, volatility_band = _regime_components(regime)
        subgroup_members.setdefault(("surprise_band", surprise_band), []).append(index)
        subgroup_members.setdefault(("volatility_band", volatility_band), []).append(index)

    rows: list[dict[str, float | str | int]] = []
    for (subgroup_type, subgroup_name), indices in sorted(subgroup_members.items()):
        subgroup_outputs = [outputs[index] for index in indices]
        subgroup_labels = [labels[index] for index in indices]
        subgroup_regimes = [regimes[index] for index in indices]
        row = _metric_row(
            method=method,
            outputs=subgroup_outputs,
            labels=subgroup_labels,
            regimes=subgroup_regimes,
            predictor=predictor,
            global_thresholds=global_thresholds,
            mode=mode,
        )
        row["subgroup_type"] = subgroup_type
        row["subgroup"] = subgroup_name
        row["n"] = len(indices)
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
    feature_stats = _load_feature_stats()
    all_events = _load_cached_events()
    all_events = filter_events_by_universe(all_events, data_config.get("universe"))
    if not all_events:
        raise ValueError(
            "No usable cached events were found for evaluation. "
            f"Coverage summary: {summarize_event_coverage(all_events)}"
        )

    try:
        train_events, cal_events, test_events = _split_events_by_year(all_events)
    except Exception as exc:
        raise ValueError(f"Could not build the requested evaluation split. {exc}") from exc
    _assert_disjoint_splits(train_events, cal_events, test_events)
    cal_dataset = EarningsDataset(cal_events, feature_stats=feature_stats)
    test_dataset = EarningsDataset(test_events, feature_stats=feature_stats)

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
    )

    coverage_levels = list(config.get("calibration", {}).get("coverage_levels", [0.80, 0.90, 0.95]))
    predictor = EventConditionedConformalPredictor(coverage_levels=coverage_levels)
    predictor.calibrate(
        cal_outputs=cal_outputs,
        cal_labels=cal_labels,
        cal_regimes=cal_regimes,
    )
    global_thresholds = _build_global_thresholds(cal_outputs, cal_labels, coverage_levels)
    historical_events = train_events + cal_events

    results_rows: list[dict[str, float | str]] = []
    subgroup_rows: list[dict[str, float | str | int]] = []

    for method, mode in (
        ("text_only", "regime_no_introspection"),
        ("financial_only", "regime_no_introspection"),
        ("sentiment_only", "regime_no_introspection"),
        ("full_multimodal", "regime_no_introspection"),
        ("naive_conformal", "naive"),
        ("ours", "ours"),
    ):
        base_method = "full_multimodal" if method in {"full_multimodal", "naive_conformal", "ours"} else method
        outputs, labels, regimes, _tickers = _compute_model_outputs(
            model=model,
            events=test_events,
            dataset=test_dataset,
            method=base_method,
            device=device,
        )
        results_rows.append(
            _metric_row(
                method=method,
                outputs=outputs,
                labels=labels,
                regimes=regimes,
                predictor=predictor,
                global_thresholds=global_thresholds,
                mode=mode,
            )
        )
        subgroup_rows.extend(
            _subgroup_metric_rows(
                method=method,
                outputs=outputs,
                labels=labels,
                regimes=regimes,
                predictor=predictor,
                global_thresholds=global_thresholds,
                mode=mode,
            )
        )

    baseline_outputs, baseline_labels, baseline_regimes, _baseline_tickers = _compute_same_ticker_baseline(
        test_events=test_events,
        training_events=train_events,
    )
    results_rows.append(
        _metric_row(
            method="same_ticker_baseline",
            outputs=baseline_outputs,
            labels=baseline_labels,
            regimes=baseline_regimes,
            predictor=predictor,
            global_thresholds=global_thresholds,
            mode="naive",
        )
    )
    subgroup_rows.extend(
        _subgroup_metric_rows(
            method="same_ticker_baseline",
            outputs=baseline_outputs,
            labels=baseline_labels,
            regimes=baseline_regimes,
            predictor=predictor,
            global_thresholds=global_thresholds,
            mode="naive",
        )
    )

    results_table = pd.DataFrame(results_rows)
    subgroup_results_table = pd.DataFrame(subgroup_rows)
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

    experiments_results_path = PROJECT_ROOT / "experiments" / "results.csv"
    root_results_path = PROJECT_ROOT / "results.csv"
    experiments_subgroup_results_path = PROJECT_ROOT / "experiments" / "results_by_subgroup.csv"
    root_subgroup_results_path = PROJECT_ROOT / "results_by_subgroup.csv"
    results_table.to_csv(experiments_results_path, index=False)
    results_table.to_csv(root_results_path, index=False)
    subgroup_results_table.to_csv(experiments_subgroup_results_path, index=False)
    subgroup_results_table.to_csv(root_subgroup_results_path, index=False)


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
