"""Build an offline multimodal event cache for training and evaluation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.loader import load_earnings_event
from encoders.sentiment_encoder import aggregate_posts


DATA_DIR = PROJECT_ROOT / "data"
TRANSCRIPTS_PATH = DATA_DIR / "transcripts.csv"
FINANCIALS_PATH = DATA_DIR / "financials.csv"
EVENTS_CACHE_PATH = DATA_DIR / "events_cache.pt"
ERROR_LOG_PATH = DATA_DIR / "build_cache_errors.log"
DEFAULT_YEARS = [2021, 2022, 2023, 2024, 2025]


def _canonicalize_ticker(ticker: str) -> str:
    """Normalize ticker symbols for consistent cross-file matching."""

    return str(ticker).upper().replace(".", "-").strip()


def _normalize_date(date_value: Any) -> str:
    """Normalize a date-like value to ``YYYY-MM-DD``."""

    return pd.Timestamp(date_value).strftime("%Y-%m-%d")


def _load_source_index(csv_path: Path) -> pd.DataFrame:
    """Load and normalize a source CSV into ticker/date index rows."""

    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing required source file: {csv_path}")

    data_frame = pd.read_csv(csv_path)
    required_columns = {"ticker", "date"}
    if not required_columns.issubset(data_frame.columns):
        raise ValueError(
            f"{csv_path.name} must contain columns {sorted(required_columns)}."
        )

    index_frame = data_frame.loc[:, ["ticker", "date"]].copy()
    index_frame["ticker"] = index_frame["ticker"].map(_canonicalize_ticker)
    index_frame["date"] = pd.to_datetime(index_frame["date"], errors="coerce")
    index_frame = index_frame.dropna(subset=["date"])
    index_frame["date"] = index_frame["date"].dt.strftime("%Y-%m-%d")
    index_frame["year"] = pd.to_datetime(index_frame["date"]).dt.year.astype(int)
    return index_frame.drop_duplicates(subset=["ticker", "date"])


def _load_existing_cache() -> list[dict[str, Any]]:
    """Load a previously-built cache file when available."""

    if not EVENTS_CACHE_PATH.is_file():
        return []

    try:
        cached_payload = torch.load(EVENTS_CACHE_PATH, map_location="cpu")
    except Exception:
        return []

    if not isinstance(cached_payload, list):
        return []

    normalized_events: list[dict[str, Any]] = []
    for event in cached_payload:
        if not isinstance(event, dict):
            continue
        try:
            normalized_events.append(
                {
                    **event,
                    "ticker": _canonicalize_ticker(str(event.get("ticker", ""))),
                    "date": _normalize_date(event.get("date")),
                }
            )
        except Exception:
            continue
    return normalized_events


def _cache_key(event: dict[str, Any]) -> tuple[str, str]:
    """Return the ticker/date key for one cached event."""

    return (
        _canonicalize_ticker(str(event.get("ticker", ""))),
        _normalize_date(event.get("date")),
    )


def _pairs_to_process(
    tickers: list[str] | None,
    years: list[int],
) -> list[tuple[str, str]]:
    """Build the matched transcript/financial ticker-date universe to cache."""

    transcripts_index = _load_source_index(TRANSCRIPTS_PATH)
    financials_index = _load_source_index(FINANCIALS_PATH)

    transcript_pairs = {
        (_canonicalize_ticker(ticker), date)
        for ticker, date in zip(transcripts_index["ticker"], transcripts_index["date"])
    }
    financial_pairs = {
        (_canonicalize_ticker(ticker), date)
        for ticker, date in zip(financials_index["ticker"], financials_index["date"])
    }

    matched_pairs = transcript_pairs & financial_pairs
    target_years = set(int(year) for year in years)
    target_tickers = (
        {_canonicalize_ticker(ticker) for ticker in tickers}
        if tickers
        else None
    )

    filtered_pairs: list[tuple[str, str]] = []
    for ticker, date_str in matched_pairs:
        event_year = int(pd.Timestamp(date_str).year)
        if event_year not in target_years:
            continue
        if target_tickers is not None and ticker not in target_tickers:
            continue
        filtered_pairs.append((ticker, date_str))

    return sorted(filtered_pairs, key=lambda pair: (pair[1], pair[0]))


def _log_error(ticker: str, date_str: str, reason: str) -> None:
    """Append a cache-build failure or skip reason to the log file."""

    ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ERROR_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{ticker},{date_str}: {reason}\n")


def _feature_list(event: dict[str, Any]) -> list[float]:
    """Convert the loader's feature dict into a cached ordered feature list."""

    raw_features = event.get("features", {})
    if not isinstance(raw_features, dict):
        raw_features = {}

    return [
        float(raw_features.get("sue", math.nan)),
        float(raw_features.get("momentum", math.nan)),
        float(raw_features.get("implied_vol", math.nan)),
    ]


def _has_nan(values: list[float]) -> bool:
    """Return whether a sequence of floats contains at least one NaN."""

    return any(math.isnan(float(value)) for value in values)


def _merge_events(
    existing_events: list[dict[str, Any]],
    new_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge newly cached events into an existing cache without duplicates."""

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for event in existing_events + new_events:
        try:
            merged[_cache_key(event)] = event
        except Exception:
            continue
    return sorted(
        merged.values(),
        key=lambda event: (
            _normalize_date(event.get("date")),
            _canonicalize_ticker(str(event.get("ticker", ""))),
        ),
    )


def build_cache(
    tickers: list[str] | None = None,
    years: list[int] | None = None,
) -> None:
    """Precompute all matched events and save them to ``events_cache.pt``."""

    target_years = [int(year) for year in (years or DEFAULT_YEARS)]
    existing_events = _load_existing_cache()
    existing_pairs = {_cache_key(event) for event in existing_events}
    matched_pairs = _pairs_to_process(tickers=tickers, years=target_years)

    new_events: list[dict[str, Any]] = []
    skipped_nan = 0

    for index, (ticker, date_str) in enumerate(matched_pairs, start=1):
        if (ticker, date_str) in existing_pairs:
            continue

        print(f"Caching {ticker} {date_str} ({index}/{len(matched_pairs)})...")
        try:
            loaded_event = load_earnings_event(ticker=ticker, earnings_date=date_str)
        except Exception as exc:
            _log_error(ticker, date_str, f"load failure: {exc}")
            continue

        try:
            label = float(loaded_event.get("label", math.nan))
            features = _feature_list(loaded_event)
            if math.isnan(label):
                skipped_nan += 1
                _log_error(ticker, date_str, "skipped: label is NaN")
                continue
            if _has_nan(features):
                skipped_nan += 1
                _log_error(ticker, date_str, "skipped: feature list contains NaN")
                continue

            transcript = str(loaded_event.get("transcript", ""))
            raw_posts = loaded_event.get("sentiment_posts", [])
            if not isinstance(raw_posts, list):
                raw_posts = []
            sentiment_raw = [str(post) for post in raw_posts]
            sentiment_tensor = aggregate_posts(sentiment_raw)
            sentiment_features = [
                float(sentiment_tensor[0].item()),
                float(sentiment_tensor[1].item()),
            ]
            event_year = int(pd.Timestamp(date_str).year)

            new_events.append(
                {
                    "ticker": ticker,
                    "date": date_str,
                    "transcript": transcript,
                    "features": features,
                    "sentiment_raw": sentiment_raw,
                    "sentiment_features": sentiment_features,
                    "label": label,
                    "year": event_year,
                }
            )
        except Exception as exc:
            _log_error(ticker, date_str, f"serialization failure: {exc}")
            continue

    merged_events = _merge_events(existing_events, new_events)
    EVENTS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_events, EVENTS_CACHE_PATH)

    years_covered = [int(event.get("year", 0)) for event in merged_events]
    tickers_covered = {
        _canonicalize_ticker(str(event.get("ticker", "")))
        for event in merged_events
    }

    print(f"total events cached: {len(merged_events)}")
    print(f"train (year <= 2023): {sum(1 for year in years_covered if year <= 2023)}")
    print(f"calibration (2024):   {sum(1 for year in years_covered if year == 2024)}")
    print(f"test (2025):          {sum(1 for year in years_covered if year == 2025)}")
    print(f"tickers covered:      {len(tickers_covered)}")
    print(f"skipped (NaN):        {skipped_nan}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build an offline event cache for training and evaluation.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Optional subset of tickers to cache.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=DEFAULT_YEARS,
        help="Optional subset of years to cache.",
    )
    args = parser.parse_args()

    build_cache(tickers=args.tickers, years=args.years)
