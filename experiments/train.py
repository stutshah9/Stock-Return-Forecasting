"""Training loop for the multimodal earnings forecasting model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

try:
    from torch.amp import GradScaler as TorchGradScaler
    from torch.amp import autocast as torch_autocast

    def _make_grad_scaler(device_type: str, enabled: bool) -> TorchGradScaler:
        return TorchGradScaler(device_type, enabled=enabled)

    def _autocast(device_type: str, enabled: bool):
        return torch_autocast(device_type=device_type, enabled=enabled)

except ImportError:  # pragma: no cover - older torch fallback
    from torch.cuda.amp import GradScaler as TorchGradScaler
    from torch.cuda.amp import autocast as torch_autocast

    def _make_grad_scaler(device_type: str, enabled: bool) -> TorchGradScaler:
        del device_type
        return TorchGradScaler(enabled=enabled)

    def _autocast(device_type: str, enabled: bool):
        del device_type
        return torch_autocast(enabled=enabled)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration.conformal import assign_regime, fit_regime_thresholds
from data.dataset import EarningsDataset
from data.event_utils import (
    build_synthetic_events as _shared_build_synthetic_events,
    filter_events_by_universe,
    summarize_event_coverage,
)
from encoders.sentiment_encoder import aggregate_posts
from models.fusion_model import MultimodalForecastModel


_CLI_DRY_RUN = False
EVENTS_CACHE_PATH = PROJECT_ROOT / "data" / "events_cache.pt"
FEATURE_STATS_PATH = PROJECT_ROOT / "experiments" / "feature_stats.json"
REGIME_THRESHOLDS_PATH = PROJECT_ROOT / "experiments" / "regime_thresholds.json"
BEST_MODEL_PATH = PROJECT_ROOT / "experiments" / "model_best.pt"
TRAINING_SUMMARY_PATH = PROJECT_ROOT / "experiments" / "training_summary.json"


def _resolve_path(path_str: str) -> Path:
    """Resolve a possibly relative path against the project root."""

    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration values from disk."""

    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def _set_seed(seed: int) -> None:
    """Set random seeds for more reproducible training runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_dry_run_events() -> list[dict[str, Any]]:
    """Build a small synthetic dataset for end-to-end dry-run verification."""

    normalized_events: list[dict[str, Any]] = []
    dry_run_years = [2021, 2022, 2023, 2024, 2025, 2025]
    for index, event in enumerate(_shared_build_synthetic_events()):
        raw_posts = [str(post) for post in list(event.get("sentiment_posts", []))]
        sentiment_features = aggregate_posts(raw_posts)
        raw_features = event.get("features", {})
        original_date = str(event.get("date", "2023-01-01"))
        adjusted_year = dry_run_years[min(index, len(dry_run_years) - 1)]
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


def _collate_batch(
    batch: list[dict[str, Any]],
    regime_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Collate dataset samples into model-ready tensors and metadata."""

    transcripts = [str(item["transcript"]) for item in batch]
    financial = torch.stack([item["features"].float() for item in batch], dim=0)
    sentiment = torch.stack([item["sentiment"].float() for item in batch], dim=0)
    labels = torch.stack([item["label"].float() for item in batch], dim=0).view(-1)
    regimes = [
        assign_regime(
            sue=float(item["raw_features"][0].item()),
            implied_vol=float(item["raw_features"][2].item()),
            thresholds=regime_thresholds,
        )
        for item in batch
    ]
    return {
        "transcripts": transcripts,
        "financial": financial,
        "sentiment": sentiment,
        "labels": labels,
        "regimes": regimes,
    }


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor batch fields to the target device."""

    non_blocking = device.type == "cuda"
    return {
        "transcripts": batch["transcripts"],
        "financial": batch["financial"].to(device, non_blocking=non_blocking),
        "sentiment": batch["sentiment"].to(device, non_blocking=non_blocking),
        "labels": batch["labels"].to(device, non_blocking=non_blocking),
        "regimes": batch["regimes"],
    }


def _select_device(config: dict[str, Any]) -> torch.device:
    """Select the requested training device with sensible GPU fallbacks.

    The selection order for ``auto`` is CUDA, then Apple Metal (MPS), then CPU.
    """

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
    tqdm.write(f"Model parameter device: {parameter_device}")

    if device.type != "cuda":
        return

    tqdm.write(f"torch.version.cuda: {torch.version.cuda}")
    tqdm.write(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    tqdm.write(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    tqdm.write(f"CUDA current device index: {torch.cuda.current_device()}")

    warmup_tensor = torch.empty((1024, 1024), device=device)
    warmup_tensor.zero_()
    torch.cuda.synchronize(device)

    allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
    tqdm.write(
        f"CUDA memory after warmup: allocated={allocated_mb:.2f} MB, "
        f"reserved={reserved_mb:.2f} MB"
    )
    del warmup_tensor


def _serialize_output_batch(outputs: dict[str, Tensor]) -> list[dict[str, float]]:
    """Convert batched model outputs into a list of Python dictionaries."""

    serialized: list[dict[str, float]] = []
    quantile_levels = [float(value) for value in outputs["quantile_levels"].detach().cpu().tolist()]
    quantile_predictions = outputs["quantile_predictions"].detach().cpu().tolist()
    base_intervals_raw = outputs.get("base_intervals", {})
    q_low_values = outputs["q_low"].detach().cpu().tolist()
    q_high_values = outputs["q_high"].detach().cpu().tolist()
    mu_values = outputs["mu"].detach().cpu().tolist()
    point_mu_values = outputs.get("point_mu", outputs["mu"]).detach().cpu().tolist()
    quantile_median_values = outputs.get("quantile_median", outputs["mu"]).detach().cpu().tolist()
    score_values = outputs["introspective_score"].detach().cpu().tolist()
    attention_values = outputs.get("attention_stability")
    interval_confidence_values = outputs.get("interval_confidence")
    modality_values = outputs.get("modality_consistency")
    attention_list = (
        attention_values.detach().cpu().tolist()
        if attention_values is not None
        else [0.0 for _ in mu_values]
    )
    interval_confidence_list = (
        interval_confidence_values.detach().cpu().tolist()
        if interval_confidence_values is not None
        else [0.0 for _ in mu_values]
    )
    modality_list = (
        modality_values.detach().cpu().tolist()
        if modality_values is not None
        else [0.0 for _ in mu_values]
    )

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
        q_low_values,
        q_high_values,
        mu_values,
        point_mu_values,
        quantile_median_values,
        score_values,
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
        serialized.append(
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
            }
        )
    return serialized


def _point_prediction(output: dict[str, float]) -> float:
    """Read the point forecast used for directional test metrics."""

    return float(output.get("point_mu", output["mu"]))


def _binary_direction_metrics(
    predictions: list[float],
    labels: list[float],
) -> dict[str, float | int]:
    """Compute binary up/down test metrics from return signs."""

    predicted_positive = [float(value) >= 0.0 for value in predictions]
    actual_positive = [float(value) >= 0.0 for value in labels]

    tp = sum(1 for pred, actual in zip(predicted_positive, actual_positive) if pred and actual)
    tn = sum(1 for pred, actual in zip(predicted_positive, actual_positive) if not pred and not actual)
    fp = sum(1 for pred, actual in zip(predicted_positive, actual_positive) if pred and not actual)
    fn = sum(1 for pred, actual in zip(predicted_positive, actual_positive) if not pred and actual)
    total = len(labels)

    accuracy = (tp + tn) / total if total else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support": int(total),
        "positive_support": int(sum(actual_positive)),
        "negative_support": int(total - sum(actual_positive)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def _run_inference(
    model: MultimodalForecastModel,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[list[dict[str, float]], list[float], list[str]]:
    """Run no-grad inference and collect serialized outputs, labels, and regimes."""

    outputs_list: list[dict[str, float]] = []
    labels_list: list[float] = []
    regimes_list: list[str] = []

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch = _move_batch_to_device(batch, device)
            outputs = model(
                transcripts=batch["transcripts"],
                financial=batch["financial"],
                sentiment=batch["sentiment"],
            )
            outputs_list.extend(_serialize_output_batch(outputs))
            labels_list.extend(float(value) for value in batch["labels"].cpu().tolist())
            regimes_list.extend(batch["regimes"])
    return outputs_list, labels_list, regimes_list


def train(config_path: str) -> None:
    """Train the multimodal forecast model and save calibration artifacts.

    Args:
        config_path: Path to the YAML configuration file.
    """

    config_file = _resolve_path(config_path)
    config = _load_config(config_file)
    training_config = config.get("training", {})
    data_config = config.get("data", {})
    split_config = data_config.get("split", {})
    batch_size = int(training_config.get("batch_size", 32))
    epochs = int(training_config.get("epochs", 20))
    seed = int(training_config.get("seed", 42))
    use_amp = bool(training_config.get("use_amp", True))
    grad_clip_norm = float(training_config.get("grad_clip_norm", 1.0))
    early_stop_metric = str(training_config.get("early_stop_metric", "val_rmse"))
    pinball_quantiles = [
        float(value)
        for value in training_config.get("pinball_quantiles", [0.10, 0.90])
        if 0.0 < float(value) < 1.0
    ]
    if len(pinball_quantiles) < 2:
        pinball_quantiles = [0.10, 0.90]

    _set_seed(seed)

    if _CLI_DRY_RUN:
        events = _build_dry_run_events()
        batch_size = 2
        epochs = 1
        max_train_batches = 2
    else:
        tqdm.write("Loading cached events...")
        events = _load_cached_events()
        max_train_batches = None

    try:
        events = filter_events_by_universe(events, data_config.get("universe"))
    except Exception as exc:
        raise ValueError(f"Failed to apply ticker universe filter: {exc}") from exc

    if len(events) < 3:
        raise ValueError(
            "Insufficient usable events after filtering. "
            f"Coverage summary: {summarize_event_coverage(events)}. "
            "Build the offline cache first with "
            "`python3 data/build_cache.py`, or use --dry-run to verify the pipeline."
        )

    device = _select_device(config)
    tqdm.write(f"Using device: {device}")
    quantile_text = ", ".join(f"{quantile:.2f}" for quantile in pinball_quantiles)
    tqdm.write(f"Regression loss: pinball quantile loss [{quantile_text}]")
    if device.type == "cuda":
        tqdm.write(f"CUDA device: {torch.cuda.get_device_name(0)}")

    model = MultimodalForecastModel.load_from_config(str(config_file))
    model.to(device)
    _log_device_diagnostics(device, model)

    try:
        train_events, val_events, cal_events, test_events = _split_events_by_year(
            events,
            split_config,
        )
    except Exception as exc:
        raise ValueError(f"Could not build the requested split. {exc}") from exc
    _assert_disjoint_splits(train_events, val_events, cal_events, test_events)

    feature_stats = EarningsDataset.compute_feature_stats(train_events)
    FEATURE_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FEATURE_STATS_PATH.open("w", encoding="utf-8") as stats_file:
        json.dump(feature_stats, stats_file)
    regime_fit_config = config.get("calibration", {}).get("regime_fit_quantiles", {})
    regime_thresholds = fit_regime_thresholds(
        train_events + val_events,
        low_quantile=float(regime_fit_config.get("low_surprise", 0.60)),
        high_quantile=float(regime_fit_config.get("high_surprise", 0.90)),
        vol_quantile=float(regime_fit_config.get("high_vol", 0.60)),
    )
    with REGIME_THRESHOLDS_PATH.open("w", encoding="utf-8") as thresholds_file:
        json.dump(regime_thresholds, thresholds_file)

    train_dataset = EarningsDataset(train_events, feature_stats=feature_stats)
    val_dataset = EarningsDataset(val_events, feature_stats=feature_stats)
    cal_dataset = EarningsDataset(cal_events, feature_stats=feature_stats)
    test_dataset = EarningsDataset(test_events, feature_stats=feature_stats)
    collate_fn = lambda batch: _collate_batch(batch, regime_thresholds=regime_thresholds)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    cal_loader = DataLoader(
        cal_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config.get("lr", 1e-4)),
        weight_decay=1e-4,
    )
    scaler = _make_grad_scaler(
        device_type=device.type,
        enabled=device.type == "cuda" and use_amp,
    )

    best_val_score = float("inf")
    best_epoch = 0
    best_val_metrics: dict[str, float] = {}
    patience_counter = 0
    PATIENCE = 5
    validation_history: list[dict[str, float | int]] = []

    for epoch_index in range(epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch_index + 1}/{epochs}",
            leave=False,
        )
        for batch_index, batch in enumerate(progress):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break

            batch = _move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with _autocast(device_type=device.type, enabled=device.type == "cuda" and use_amp):
                outputs = model(
                    transcripts=batch["transcripts"],
                    financial=batch["financial"],
                    sentiment=batch["sentiment"],
                )
                loss = model.loss(outputs, batch["labels"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            batch_count += 1
            progress.set_postfix(train_loss=f"{loss.item():.4f}")

        if batch_count == 0:
            raise RuntimeError("Training loop saw zero batches.")

        epoch_loss = running_loss / batch_count
        tqdm.write(f"Epoch {epoch_index + 1}/{epochs} train_loss={epoch_loss:.4f}")

        model.eval()
        val_loss_total = 0.0
        val_predictions: list[float] = []
        val_labels: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = _move_batch_to_device(batch, device)
                with _autocast(device_type=device.type, enabled=device.type == "cuda" and use_amp):
                    out = model(
                        transcripts=batch["transcripts"],
                        financial=batch["financial"],
                        sentiment=batch["sentiment"],
                    )
                    val_loss_total += model.loss(out, batch["labels"]).item()
                val_predictions.extend(float(value) for value in out["mu"].detach().cpu().tolist())
                val_labels.extend(float(value) for value in batch["labels"].detach().cpu().tolist())
        val_loss_avg = val_loss_total / len(val_loader)
        val_mae = float(np.mean([abs(pred - label) for pred, label in zip(val_predictions, val_labels)]))
        val_rmse = float(
            np.sqrt(np.mean([(pred - label) ** 2 for pred, label in zip(val_predictions, val_labels)]))
        )
        print(f"  val_loss={val_loss_avg:.4f} val_mae={val_mae:.4f} val_rmse={val_rmse:.4f}")
        validation_history.append(
            {
                "epoch": int(epoch_index + 1),
                "train_loss": float(epoch_loss),
                "val_loss": float(val_loss_avg),
                "val_mae": float(val_mae),
                "val_rmse": float(val_rmse),
            }
        )
        model.train()

        current_score = {
            "val_loss": float(val_loss_avg),
            "val_mae": float(val_mae),
            "val_rmse": float(val_rmse),
        }.get(early_stop_metric, float(val_rmse))

        if current_score < best_val_score:
            best_val_score = current_score
            best_epoch = epoch_index + 1
            best_val_metrics = {
                "val_loss": float(val_loss_avg),
                "val_mae": float(val_mae),
                "val_rmse": float(val_rmse),
            }
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch_index + 1}")
                break

    if BEST_MODEL_PATH.is_file():
        best_state_dict = torch.load(BEST_MODEL_PATH, map_location="cpu")
        model.load_state_dict(best_state_dict, strict=False)
        model.to(device)

    cal_outputs, cal_labels, cal_regimes = _run_inference(model, cal_loader, device)
    test_outputs, test_labels, _test_regimes = _run_inference(model, test_loader, device)
    test_predictions = [_point_prediction(output) for output in test_outputs]
    test_mae = float(
        np.mean([abs(prediction - label) for prediction, label in zip(test_predictions, test_labels)])
    )
    test_rmse = float(
        np.sqrt(
            np.mean(
                [
                    (prediction - label) ** 2
                    for prediction, label in zip(test_predictions, test_labels)
                ]
            )
        )
    )
    test_metrics = {
        "mae": test_mae,
        "rmse": test_rmse,
        **_binary_direction_metrics(test_predictions, test_labels),
    }
    print(
        "Test metrics: "
        f"n={test_metrics['support']} "
        f"mae={test_metrics['mae']:.4f} "
        f"rmse={test_metrics['rmse']:.4f} "
        f"accuracy={test_metrics['accuracy']:.4f} "
        f"precision={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f} "
        f"f1={test_metrics['f1']:.4f}"
    )

    data_output_path = PROJECT_ROOT / "data" / "cal_outputs.pt"
    model_output_path = PROJECT_ROOT / "experiments" / "model.pt"
    legacy_cal_output_path = PROJECT_ROOT / "cal_outputs.pt"
    legacy_model_output_path = PROJECT_ROOT / "model.pt"

    saved_calibration = {
        "outputs": cal_outputs,
        "labels": cal_labels,
        "regimes": cal_regimes,
    }
    training_summary = {
        "early_stop_metric": early_stop_metric,
        "best_val_score": float(best_val_score),
        "best_epoch": int(best_epoch),
        "best_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "history": validation_history,
    }
    torch.save(saved_calibration, data_output_path)
    torch.save(model.state_dict(), model_output_path)
    with TRAINING_SUMMARY_PATH.open("w", encoding="utf-8") as summary_file:
        json.dump(training_summary, summary_file, indent=2)

    # Keep compatibility with existing local tests while also saving to spec paths.
    torch.save(saved_calibration, legacy_cal_output_path)
    torch.save(model.state_dict(), legacy_model_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the multimodal forecast model.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run two batches of two synthetic samples to verify the pipeline.",
    )
    args = parser.parse_args()

    _CLI_DRY_RUN = args.dry_run
    train(args.config)
