import os
import subprocess
import sys
import tempfile

import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def main() -> None:
    root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    config_path = os.path.join(root, "config.yaml")
    eval_config_path = config_path

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if (
        str(config.get("training", {}).get("device", "auto")).lower() == "cuda"
        and not torch.cuda.is_available()
    ):
        config.setdefault("training", {})["device"] = "cpu"
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
            encoding="utf-8",
        ) as temp_config:
            yaml.safe_dump(config, temp_config)
            eval_config_path = temp_config.name

    subprocess.run(
        [
            "python3",
            "experiments/evaluate.py",
            "--allow-synthetic-fallback",
            "--config",
            eval_config_path,
        ],
        cwd=root,
        check=True,
    )

    results_path = os.path.join(root, "results.csv")
    assert os.path.isfile(results_path), "results.csv must exist after evaluation"
    subgroup_results_path = os.path.join(root, "results_by_subgroup.csv")
    assert os.path.isfile(
        subgroup_results_path
    ), "results_by_subgroup.csv must exist after evaluation"
    predictions_path = os.path.join(root, "predictions.csv")
    assert os.path.isfile(predictions_path), "predictions.csv must exist after evaluation"
    results = pd.read_csv(results_path)
    subgroup_results = pd.read_csv(subgroup_results_path)
    predictions = pd.read_csv(predictions_path)
    assert "method" in results.columns, "results.csv missing method column"
    assert len(results) >= 7, "results.csv must contain at least 7 method rows"

    methods = [str(x) for x in results["method"].tolist()]
    assert (
        "normalized_conformal_modality" in methods
    ), "Missing required normalized_conformal_modality method row"
    assert len(set(methods)) >= 7, "There must be at least 7 distinct method rows"

    required_columns = [
        "method",
        "coverage_80",
        "coverage_90",
        "coverage_95",
        "avg_width_80",
        "avg_width_90",
        "avg_width_95",
        "dir_acc",
        "avg_explanation_confidence",
        "avg_predicted_variance_proxy",
        "avg_variance_weighted_explanation_error",
    ]
    for column in required_columns:
        assert column in results.columns, "results.csv missing required column"

    for cov_col in ("coverage_80", "coverage_90", "coverage_95"):
        assert (results[cov_col] >= 0.0).all(), "Coverage must be >= 0"
        assert (results[cov_col] <= 1.0).all(), "Coverage must be <= 1"

    normalized = results.loc[results["method"] == "normalized_conformal_modality"]
    assert len(normalized) == 1, "There must be exactly one normalized conformal row"
    normalized_row = normalized.iloc[0]
    assert (
        normalized_row["coverage_95"] >= normalized_row["coverage_80"]
    ), "normalized coverage_95 must be >= coverage_80"

    assert (results["dir_acc"] >= 0.0).all(), "dir_acc must be >= 0"
    assert (results["dir_acc"] <= 1.0).all(), "dir_acc must be <= 1"

    subgroup_required_columns = [
        "method",
        "subgroup_type",
        "subgroup",
        "n",
        "coverage_80",
        "coverage_90",
        "coverage_95",
        "avg_width_80",
        "avg_width_90",
        "avg_width_95",
    ]
    for column in subgroup_required_columns:
        assert (
            column in subgroup_results.columns
        ), "results_by_subgroup.csv missing required column"

    subgroup_types = set(str(value) for value in subgroup_results["subgroup_type"].tolist())
    assert "surprise_band" in subgroup_types, "Missing surprise-band subgroup metrics"
    assert "volatility_band" in subgroup_types, "Missing volatility-band subgroup metrics"
    assert "attention_volume_band" in subgroup_types, "Missing attention-volume subgroup metrics"
    assert (subgroup_results["n"] > 0).all(), "Subgroup counts must be positive"
    for cov_col in ("coverage_80", "coverage_90", "coverage_95"):
        assert (subgroup_results[cov_col] >= 0.0).all(), "Subgroup coverage must be >= 0"
        assert (subgroup_results[cov_col] <= 1.0).all(), "Subgroup coverage must be <= 1"

    prediction_required_columns = [
        "method",
        "ticker",
        "date",
        "year",
        "actual_return",
        "expected_return",
        "predicted_return",
        "abs_error",
        "prediction_error",
        "predicted_variance_proxy",
        "variance_confidence",
        "explanation_confidence",
        "explanation_adjusted_abs_error",
        "variance_weighted_explanation_error",
        "coverage_90_lower",
        "coverage_90_upper",
        "regime",
        "explanation",
    ]
    for column in prediction_required_columns:
        assert column in predictions.columns, "predictions.csv missing required column"
    assert len(predictions) > 0, "predictions.csv must contain at least one row"
    assert predictions["explanation"].fillna("").astype(str).str.len().gt(0).all(), "Each prediction row must include a non-empty explanation"

    print("All evaluation tests passed.")


if __name__ == "__main__":
    main()
