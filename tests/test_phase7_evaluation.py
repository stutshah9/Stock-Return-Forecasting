import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def main() -> None:
    root = os.path.dirname(os.path.dirname(__file__))

    subprocess.run(
        ["python3", "experiments/evaluate.py", "--allow-synthetic-fallback"],
        cwd=root,
        check=True,
    )

    results_path = os.path.join(root, "results.csv")
    assert os.path.isfile(results_path), "results.csv must exist after evaluation"

    results = pd.read_csv(results_path)
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

    print("All evaluation tests passed.")


if __name__ == "__main__":
    main()
