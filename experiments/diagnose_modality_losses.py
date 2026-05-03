"""Compare full-model and unimodal losses from a trained checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import EarningsDataset
from data.event_utils import filter_events_by_universe
from experiments.evaluate import (
    FEATURE_STATS_PATH,
    _assert_disjoint_splits,
    _compute_model_outputs,
    _load_cached_events,
    _load_regime_thresholds,
    _prepare_modal_inputs,
    _split_events_by_year,
)
from models.fusion_model import MultimodalForecastModel


METHODS = ("full_multimodal", "text_only", "financial_only", "sentiment_only")


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _select_device(config: dict[str, Any], override: str | None) -> torch.device:
    requested = (override or str(config.get("training", {}).get("device", "auto"))).lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def _load_feature_stats() -> dict[str, list[float]]:
    with FEATURE_STATS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_model(config_path: Path, device: torch.device) -> MultimodalForecastModel:
    model_path = PROJECT_ROOT / "experiments" / "model_best.pt"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "experiments" / "model.pt"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "model.pt"
    model = MultimodalForecastModel.load_from_config(str(config_path))
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    model.to(device)
    model.eval()
    return model


def _loss_for_method(
    model: MultimodalForecastModel,
    events: list[dict[str, Any]],
    dataset: EarningsDataset,
    method: str,
    device: torch.device,
    regime_thresholds: dict[str, float],
) -> float:
    transcripts, financial, sentiment, labels, _regimes, _tickers = _prepare_modal_inputs(
        events=events,
        dataset=dataset,
        method=method,
        device=device,
        regime_thresholds=regime_thresholds,
    )
    with torch.no_grad():
        outputs = model(transcripts=transcripts, financial=financial, sentiment=sentiment)
        return float(model.loss(outputs, labels).detach().cpu().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare losses across modality ablations.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default=None, choices=[None, "auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = _load_config(config_path)
    data_config = config.get("data", {})
    split_config = data_config.get("split", {})
    device = _select_device(config, args.device)

    events = filter_events_by_universe(_load_cached_events(), data_config.get("universe"))
    train_events, val_events, cal_events, test_events = _split_events_by_year(events, split_config)
    _assert_disjoint_splits(train_events, val_events, cal_events, test_events)
    feature_stats = _load_feature_stats()
    regime_thresholds = _load_regime_thresholds(config, train_events + val_events)
    model = _load_model(config_path, device)

    splits = {
        "validation": val_events,
        "calibration": cal_events,
        "test": test_events,
    }
    rows: list[dict[str, float | str | int]] = []
    for split_name, split_events in splits.items():
        dataset = EarningsDataset(split_events, feature_stats=feature_stats)
        for method in METHODS:
            outputs, labels, _regimes, _tickers = _compute_model_outputs(
                model=model,
                events=split_events,
                dataset=dataset,
                method=method,
                device=device,
                regime_thresholds=regime_thresholds,
            )
            loss = _loss_for_method(model, split_events, dataset, method, device, regime_thresholds)
            mae = sum(abs(float(output["point_mu"]) - label) for output, label in zip(outputs, labels))
            mae = mae / max(len(labels), 1)
            rows.append(
                {
                    "split": split_name,
                    "method": method,
                    "n": len(labels),
                    "loss": loss,
                    "mae": float(mae),
                }
            )

    output_path = PROJECT_ROOT / "experiments" / "modality_loss_diagnostics.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "method", "n", "loss", "mae"])
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['split']:>11} {row['method']:<16} "
            f"loss={float(row['loss']):.6f} mae={float(row['mae']):.6f} n={row['n']}"
        )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
