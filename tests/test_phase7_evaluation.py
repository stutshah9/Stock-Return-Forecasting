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

    results = pd.read_csv(results_path)
    subgroup_results = pd.read_csv(subgroup_results_path)
    assert "method" in results.columns, "results.csv missing method column"
    assert len(results) >= 7, "results.csv must contain at least 7 method rows"

    methods = [str(x) for x in results["method"].tolist()]
    assert "ours" in methods, "Missing required ours method row"
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
    ]
    for column in required_columns:
        assert column in results.columns, "results.csv missing required column"

    for cov_col in ("coverage_80", "coverage_90", "coverage_95"):
        assert (results[cov_col] >= 0.0).all(), "Coverage must be >= 0"
        assert (results[cov_col] <= 1.0).all(), "Coverage must be <= 1"

    ours = results.loc[results["method"] == "ours"]
    assert len(ours) == 1, "There must be exactly one ours row"
    ours_row = ours.iloc[0]
    assert ours_row["coverage_95"] >= ours_row["coverage_80"], "ours coverage_95 must be >= coverage_80"

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
    assert (subgroup_results["n"] > 0).all(), "Subgroup counts must be positive"
    for cov_col in ("coverage_80", "coverage_90", "coverage_95"):
        assert (subgroup_results[cov_col] >= 0.0).all(), "Subgroup coverage must be >= 0"
        assert (subgroup_results[cov_col] <= 1.0).all(), "Subgroup coverage must be <= 1"

    print("All evaluation tests passed.")


if __name__ == "__main__":
    main()
