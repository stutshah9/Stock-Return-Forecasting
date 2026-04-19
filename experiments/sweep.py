"""Hyperparameter sweep runner for proposal-oriented model selection."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import itertools
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
TRAIN_RESULTS_PATH = EXPERIMENTS_DIR / "training_summary.json"
EVAL_RESULTS_PATH = EXPERIMENTS_DIR / "results.csv"
SUBGROUP_RESULTS_PATH = EXPERIMENTS_DIR / "results_by_subgroup.csv"
PREDICTIONS_PATH = EXPERIMENTS_DIR / "predictions.csv"
SELECTIVE_RESULTS_PATH = EXPERIMENTS_DIR / "results_selective.csv"
MODEL_PATH = EXPERIMENTS_DIR / "model.pt"
BEST_MODEL_PATH = EXPERIMENTS_DIR / "model_best.pt"
CAL_OUTPUTS_PATH = PROJECT_ROOT / "cal_outputs.pt"


@dataclass(frozen=True)
class SweepRun:
    index: int
    name: str
    config: dict[str, Any]
    params: dict[str, float]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _build_default_grid(base_config: dict[str, Any]) -> dict[str, list[float]]:
    training = base_config.get("training", {})
    base_lr = float(training.get("lr", 5e-5))
    base_uncertainty = float(training.get("uncertainty_alignment_weight", 0.05))
    base_confidence = float(training.get("confidence_calibration_weight", 0.10))
    base_attention = float(training.get("attention_alignment_weight", 0.05))
    base_explanation = float(training.get("explanation_alignment_weight", 0.05))

    def _choices(base: float, multipliers: list[float]) -> list[float]:
        values = sorted({round(base * multiplier, 8) for multiplier in multipliers if base > 0.0})
        return [float(value) for value in values]

    return {
        "lr": _choices(base_lr, [0.6, 1.0, 1.4]),
        "uncertainty_alignment_weight": _choices(base_uncertainty, [1.0]),
        "confidence_calibration_weight": _choices(base_confidence, [0.75, 1.5]),
        "attention_alignment_weight": _choices(base_attention, [0.5, 1.5]),
        "explanation_alignment_weight": _choices(base_explanation, [0.5, 1.0]),
    }


def _comma_floats(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    values = []
    for item in raw.split(","):
        stripped = item.strip()
        if stripped:
            values.append(float(stripped))
    return values or None


def _relative_distance(value: float, base: float) -> float:
    if base > 0.0 and value > 0.0:
        return abs(math.log(value / base))
    return abs(value - base)


def _sweep_runs(
    base_config: dict[str, Any],
    grid: dict[str, list[float]],
    max_runs: int | None = None,
) -> list[SweepRun]:
    training = base_config.setdefault("training", {})
    keys = list(grid.keys())
    value_product = itertools.product(*(grid[key] for key in keys))

    candidates: list[tuple[float, dict[str, float]]] = []
    for combination in value_product:
        params = {key: float(value) for key, value in zip(keys, combination)}
        distance = 0.0
        for key, value in params.items():
            distance += _relative_distance(value, float(training.get(key, value)))
        candidates.append((distance, params))

    candidates.sort(key=lambda item: (item[0], tuple(item[1].values())))
    if max_runs is not None:
        candidates = candidates[: max(max_runs, 1)]

    runs: list[SweepRun] = []
    for index, (_distance, params) in enumerate(candidates, start=1):
        run_config = json.loads(json.dumps(base_config))
        run_training = run_config.setdefault("training", {})
        for key, value in params.items():
            run_training[key] = value
        name = f"run_{index:03d}"
        runs.append(SweepRun(index=index, name=name, config=run_config, params=params))
    return runs


def _run_command(command: list[str], cwd: Path, log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(command)}")


def _extract_method_row(rows: list[dict[str, str]], method: str) -> dict[str, float]:
    for row in rows:
        if row.get("method") == method:
            return {
                key: _parse_float(value) if key != "method" else value
                for key, value in row.items()
            }
    return {}


def _interval_objective(row: dict[str, float]) -> float:
    if not row:
        return float("inf")
    return (
        abs(float(row.get("coverage_80", float("nan"))) - 0.80)
        + abs(float(row.get("coverage_90", float("nan"))) - 0.90)
        + abs(float(row.get("coverage_95", float("nan"))) - 0.95)
        + 0.5 * float(row.get("avg_width", float("inf")))
    )


def _build_run_summary(
    run: SweepRun,
    training_summary: dict[str, Any],
    results_rows: list[dict[str, str]],
) -> dict[str, float | int | str]:
    best_metrics = training_summary.get("best_metrics", {})
    full_row = _extract_method_row(results_rows, "full_multimodal")
    ours_row = _extract_method_row(results_rows, "ours")
    naive_row = _extract_method_row(results_rows, "naive_conformal")

    summary: dict[str, float | int | str] = {
        "run_name": run.name,
        "best_epoch": int(training_summary.get("best_epoch", 0)),
        "best_val_score": _parse_float(training_summary.get("best_val_score")),
        "best_val_rmse": _parse_float(best_metrics.get("val_rmse")),
        "best_val_mae": _parse_float(best_metrics.get("val_mae")),
        "best_val_dir_acc": _parse_float(best_metrics.get("val_dir_acc")),
        "best_val_confidence_brier": _parse_float(best_metrics.get("val_confidence_brier")),
        "best_val_selective_mae": _parse_float(best_metrics.get("val_selective_mae")),
        "best_val_proposal_score": _parse_float(best_metrics.get("val_proposal_score")),
        "full_multimodal_rmse": _parse_float(full_row.get("RMSE")),
        "full_multimodal_mae": _parse_float(full_row.get("MAE")),
        "full_multimodal_dir_acc": _parse_float(full_row.get("dir_acc")),
        "ours_coverage_80": _parse_float(ours_row.get("coverage_80")),
        "ours_coverage_90": _parse_float(ours_row.get("coverage_90")),
        "ours_coverage_95": _parse_float(ours_row.get("coverage_95")),
        "ours_avg_width": _parse_float(ours_row.get("avg_width")),
        "ours_interval_score": _interval_objective(ours_row),
        "naive_avg_width": _parse_float(naive_row.get("avg_width")),
        "lr": float(run.params["lr"]),
        "uncertainty_alignment_weight": float(run.params["uncertainty_alignment_weight"]),
        "confidence_calibration_weight": float(run.params["confidence_calibration_weight"]),
        "attention_alignment_weight": float(run.params["attention_alignment_weight"]),
        "explanation_alignment_weight": float(run.params["explanation_alignment_weight"]),
    }
    return summary


def _write_summary_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _best_run(rows: list[dict[str, float | int | str]], selection_metric: str) -> dict[str, float | int | str]:
    eligible_rows = [row for row in rows if not math.isnan(_parse_float(row.get(selection_metric)))]
    if not eligible_rows:
        raise RuntimeError(f"No completed sweep runs contained selection metric: {selection_metric}")
    return min(eligible_rows, key=lambda row: _parse_float(row.get(selection_metric), float("inf")))


def _materialize_best_artifacts(
    best_run: dict[str, float | int | str],
    output_dir: Path,
    rerun_best: bool,
) -> None:
    best_run_name = str(best_run["run_name"])
    best_run_dir = output_dir / "runs" / best_run_name
    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    _copy_if_exists(best_run_dir / "config.yaml", best_dir / "config.yaml")
    _copy_if_exists(best_run_dir / "train.log", best_dir / "train.log")
    _copy_if_exists(best_run_dir / "eval.log", best_dir / "eval.log")
    _copy_if_exists(best_run_dir / "training_summary.json", best_dir / "training_summary.json")
    _copy_if_exists(best_run_dir / "results.csv", best_dir / "results.csv")
    _copy_if_exists(best_run_dir / "results_by_subgroup.csv", best_dir / "results_by_subgroup.csv")

    if not rerun_best:
        return

    config_path = best_run_dir / "config.yaml"
    _run_command(
        ["python3", "experiments/train.py", "--config", str(config_path)],
        cwd=PROJECT_ROOT,
        log_path=best_dir / "train_rerun.log",
    )
    _run_command(
        ["python3", "experiments/evaluate.py", "--config", str(config_path)],
        cwd=PROJECT_ROOT,
        log_path=best_dir / "eval_rerun.log",
    )
    _copy_if_exists(TRAIN_RESULTS_PATH, best_dir / "training_summary_rerun.json")
    _copy_if_exists(EVAL_RESULTS_PATH, best_dir / "results_rerun.csv")
    _copy_if_exists(SUBGROUP_RESULTS_PATH, best_dir / "results_by_subgroup_rerun.csv")
    _copy_if_exists(PREDICTIONS_PATH, best_dir / "predictions_rerun.csv")
    _copy_if_exists(SELECTIVE_RESULTS_PATH, best_dir / "results_selective_rerun.csv")
    _copy_if_exists(MODEL_PATH, best_dir / "model.pt")
    _copy_if_exists(BEST_MODEL_PATH, best_dir / "model_best.pt")
    _copy_if_exists(CAL_OUTPUTS_PATH, best_dir / "cal_outputs.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a proposal-oriented hyperparameter sweep.")
    parser.add_argument("--config", default="config.yaml", help="Base configuration file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for sweep outputs. Defaults to experiments/sweeps/<timestamp>.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on the number of candidate runs after sorting near the base config.",
    )
    parser.add_argument(
        "--selection-metric",
        default="best_val_proposal_score",
        choices=[
            "best_val_proposal_score",
            "best_val_rmse",
            "best_val_mae",
            "full_multimodal_rmse",
            "ours_interval_score",
        ],
        help="Metric used to select the best sweep run.",
    )
    parser.add_argument("--lr-values", default=None, help="Comma-separated learning rates.")
    parser.add_argument(
        "--uncertainty-values",
        default=None,
        help="Comma-separated uncertainty alignment weights.",
    )
    parser.add_argument(
        "--confidence-values",
        default=None,
        help="Comma-separated confidence calibration weights.",
    )
    parser.add_argument(
        "--attention-values",
        default=None,
        help="Comma-separated attention alignment weights.",
    )
    parser.add_argument(
        "--explanation-values",
        default=None,
        help="Comma-separated explanation alignment weights.",
    )
    parser.add_argument(
        "--no-rerun-best",
        action="store_true",
        help="Do not rerun the selected best config to preserve final artifacts.",
    )
    args = parser.parse_args()

    base_config_path = (PROJECT_ROOT / args.config).resolve()
    base_config = _load_yaml(base_config_path)
    default_grid = _build_default_grid(base_config)
    grid = {
        "lr": _comma_floats(args.lr_values) or default_grid["lr"],
        "uncertainty_alignment_weight": _comma_floats(args.uncertainty_values)
        or default_grid["uncertainty_alignment_weight"],
        "confidence_calibration_weight": _comma_floats(args.confidence_values)
        or default_grid["confidence_calibration_weight"],
        "attention_alignment_weight": _comma_floats(args.attention_values)
        or default_grid["attention_alignment_weight"],
        "explanation_alignment_weight": _comma_floats(args.explanation_values)
        or default_grid["explanation_alignment_weight"],
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else (EXPERIMENTS_DIR / "sweeps" / timestamp).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = _sweep_runs(base_config=base_config, grid=grid, max_runs=args.max_runs)
    manifest = {
        "base_config": str(base_config_path),
        "selection_metric": args.selection_metric,
        "grid": grid,
        "run_count": len(runs),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    summaries: list[dict[str, float | int | str]] = []
    for run in runs:
        run_dir = output_dir / "runs" / run.name
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.yaml"
        _write_yaml(config_path, run.config)

        status: dict[str, Any] = {
            "run_name": run.name,
            "params": run.params,
            "status": "running",
        }
        with (run_dir / "status.json").open("w", encoding="utf-8") as handle:
            json.dump(status, handle, indent=2)

        try:
            _run_command(
                ["python3", "experiments/train.py", "--config", str(config_path)],
                cwd=PROJECT_ROOT,
                log_path=run_dir / "train.log",
            )
            _run_command(
                ["python3", "experiments/evaluate.py", "--config", str(config_path)],
                cwd=PROJECT_ROOT,
                log_path=run_dir / "eval.log",
            )
            _copy_if_exists(TRAIN_RESULTS_PATH, run_dir / "training_summary.json")
            _copy_if_exists(EVAL_RESULTS_PATH, run_dir / "results.csv")
            _copy_if_exists(SUBGROUP_RESULTS_PATH, run_dir / "results_by_subgroup.csv")
            _copy_if_exists(PREDICTIONS_PATH, run_dir / "predictions.csv")
            _copy_if_exists(SELECTIVE_RESULTS_PATH, run_dir / "results_selective.csv")

            training_summary = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
            results_rows = _read_csv_rows(run_dir / "results.csv")
            summary = _build_run_summary(run, training_summary, results_rows)
            summaries.append(summary)
            status["status"] = "completed"
            status["summary"] = summary
        except Exception as exc:
            status["status"] = "failed"
            status["error"] = str(exc)

        with (run_dir / "status.json").open("w", encoding="utf-8") as handle:
            json.dump(status, handle, indent=2)

    _write_summary_csv(output_dir / "sweep_summary.csv", summaries)
    if not summaries:
        raise RuntimeError("Sweep finished with no successful runs.")

    best_run = _best_run(summaries, args.selection_metric)
    with (output_dir / "best_run.json").open("w", encoding="utf-8") as handle:
        json.dump(best_run, handle, indent=2)

    _materialize_best_artifacts(
        best_run=best_run,
        output_dir=output_dir,
        rerun_best=not args.no_rerun_best,
    )

    print(f"Sweep completed: {output_dir}")
    print(
        "Best run "
        f"{best_run['run_name']} with {args.selection_metric}="
        f"{_parse_float(best_run.get(args.selection_metric)):.6f}"
    )


if __name__ == "__main__":
    main()
