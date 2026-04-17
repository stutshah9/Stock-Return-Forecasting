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
    from torch.amp import autocast as _autocast
except ImportError:  # pragma: no cover - older torch fallback
    from torch.cuda.amp import autocast as _autocast

try:
    from torch.amp import GradScaler as _GradScaler

    def _make_grad_scaler(device_type: str, enabled: bool) -> Any:
        return _GradScaler(device_type, enabled=enabled)

except ImportError:  # pragma: no cover - older torch fallback
    from torch.cuda.amp import GradScaler as _GradScaler

    def _make_grad_scaler(device_type: str, enabled: bool) -> Any:
        return _GradScaler(enabled=enabled)


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
    mu_values = outputs["mu"].detach().cpu().tolist()
    log_sigma_values = outputs["log_sigma"].detach().cpu().tolist()
    score_values = outputs["introspective_score"].detach().cpu().tolist()

    for mu, log_sigma, score in zip(mu_values, log_sigma_values, score_values):
        serialized.append(
            {
                "mu": float(mu),
                "log_sigma": float(log_sigma),
                "introspective_score": float(score),
            }
        )
    return serialized


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
        device.type,
        enabled=device.type == "cuda" and use_amp,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    PATIENCE = 5

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

            with _autocast(
                device_type=device.type,
                enabled=device.type == "cuda" and use_amp,
            ):
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
        with torch.no_grad():
            for batch in val_loader:
                batch = _move_batch_to_device(batch, device)
                with _autocast(
                    device_type=device.type,
                    enabled=device.type == "cuda" and use_amp,
                ):
                    out = model(
                        transcripts=batch["transcripts"],
                        financial=batch["financial"],
                        sentiment=batch["sentiment"],
                    )
                    val_loss_total += model.loss(out, batch["labels"]).item()
        val_loss_avg = val_loss_total / len(val_loader)
        print(f"  val_loss={val_loss_avg:.4f}")
        model.train()

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
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
    _run_inference(model, test_loader, device)

    data_output_path = PROJECT_ROOT / "data" / "cal_outputs.pt"
    model_output_path = PROJECT_ROOT / "experiments" / "model.pt"
    legacy_cal_output_path = PROJECT_ROOT / "cal_outputs.pt"
    legacy_model_output_path = PROJECT_ROOT / "model.pt"

    saved_calibration = {
        "outputs": cal_outputs,
        "labels": cal_labels,
        "regimes": cal_regimes,
    }
    torch.save(saved_calibration, data_output_path)
    torch.save(model.state_dict(), model_output_path)

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
