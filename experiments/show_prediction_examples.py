"""Show a few per-event prediction examples from the exported test predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_predictions() -> pd.DataFrame:
    for candidate in (
        PROJECT_ROOT / "experiments" / "predictions.csv",
        PROJECT_ROOT / "predictions.csv",
    ):
        if candidate.is_file():
            return pd.read_csv(candidate)
    raise FileNotFoundError(
        "predictions.csv not found. Run `python3 experiments/evaluate.py` first."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Show per-event prediction examples.")
    parser.add_argument(
        "--method",
        default="all",
        help="Method to filter, e.g. ours. Use `all` to show every method.",
    )
    parser.add_argument("--year", type=int, default=None, help="Optional year filter.")
    parser.add_argument("--ticker", default=None, help="Optional ticker filter.")
    parser.add_argument("--date", default=None, help="Optional event date filter.")
    parser.add_argument("--limit", type=int, default=5, help="Number of rows to show.")
    args = parser.parse_args()

    predictions = _load_predictions()
    filtered = predictions.copy()
    if str(args.method).lower() != "all":
        filtered = filtered.loc[filtered["method"] == str(args.method)]
    if args.year is not None:
        filtered = filtered.loc[filtered["year"] == int(args.year)]
    if args.ticker:
        filtered = filtered.loc[
            filtered["ticker"].astype(str).str.upper() == str(args.ticker).upper()
        ]
    if args.date:
        filtered = filtered.loc[filtered["date"].astype(str) == str(args.date)]

    display_columns = [
        "method",
        "ticker",
        "date",
        "actual_return",
        "predicted_return",
        "prediction_error",
        "interval_90" if "interval_90" in filtered.columns else "coverage_90_lower",
        "regime",
    ]
    if filtered.empty:
        print("No prediction rows matched the requested filters.")
        return
    filtered = filtered.sort_values(["year", "ticker", "date", "method"]).reset_index(drop=True)
    print(filtered[display_columns].head(int(args.limit)).to_string(index=False))


if __name__ == "__main__":
    main()
