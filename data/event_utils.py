"""Shared utilities for event discovery, universe filtering, and dataset splits."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
import pandas as pd
import requests
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SP500_CACHE_PATH = PROJECT_ROOT / "data" / "sp500_tickers.csv"
EVENTS_CACHE_PATH = PROJECT_ROOT / "data" / "events_cache.pt"
SP500_SOURCE_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HTTP_USER_AGENT = "earnings-forecast/1.0"


def canonicalize_ticker(ticker: str) -> str:
    """Normalize a ticker symbol for cross-source compatibility."""

    return str(ticker).upper().replace(".", "-").strip()


def discover_event_keys() -> list[tuple[str, str]]:
    """Discover available ticker/date pairs from local transcript and financial files."""

    candidate_pairs: set[tuple[str, str]] = set()
    for csv_path in (
        PROJECT_ROOT / "data" / "transcripts.csv",
        PROJECT_ROOT / "data" / "financials.csv",
    ):
        if not csv_path.is_file():
            continue
        try:
            data_frame = pd.read_csv(csv_path)
        except Exception:
            continue
        if "ticker" not in data_frame.columns or "date" not in data_frame.columns:
            continue
        for ticker, date_value in zip(data_frame["ticker"], data_frame["date"]):
            candidate_pairs.add(
                (
                    canonicalize_ticker(str(ticker)),
                    pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                )
            )
    return sorted(candidate_pairs, key=lambda pair: pair[1])


def _normalize_cached_features(raw_features: Any) -> dict[str, float]:
    """Normalize cached feature storage into the training-time dict schema."""

    if isinstance(raw_features, dict):
        return {
            "sue": float(raw_features.get("sue", 0.0) or 0.0),
            "momentum": float(raw_features.get("momentum", 0.0) or 0.0),
            "implied_vol": float(raw_features.get("implied_vol", 0.0) or 0.0),
        }

    if isinstance(raw_features, (list, tuple)) and len(raw_features) >= 3:
        return {
            "sue": float(raw_features[0]),
            "momentum": float(raw_features[1]),
            "implied_vol": float(raw_features[2]),
        }

    return {
        "sue": 0.0,
        "momentum": 0.0,
        "implied_vol": 0.0,
    }


def _normalize_cached_posts(raw_posts: Any) -> list[str]:
    """Normalize cached Reddit post storage into a list of strings."""

    if not isinstance(raw_posts, list):
        return []
    return [str(post) for post in raw_posts]


def _load_events_from_cache() -> list[dict[str, Any]] | None:
    """Load cached events when an offline event cache is available."""

    if not EVENTS_CACHE_PATH.is_file():
        return None

    try:
        cached_events = torch.load(EVENTS_CACHE_PATH, map_location="cpu")
    except Exception:
        return None

    if not isinstance(cached_events, list):
        return None

    normalized_events: list[dict[str, Any]] = []
    for cached_event in cached_events:
        if not isinstance(cached_event, dict):
            continue

        try:
            event_date = pd.Timestamp(cached_event.get("date")).strftime("%Y-%m-%d")
            label = float(cached_event.get("label", math.nan))
        except Exception:
            continue

        if pd.isna(label):
            continue

        features = _normalize_cached_features(cached_event.get("features", {}))
        if any(pd.isna(feature_value) for feature_value in features.values()):
            continue

        raw_posts = cached_event.get(
            "sentiment_raw",
            cached_event.get("sentiment_posts", []),
        )
        normalized_events.append(
            {
                "ticker": canonicalize_ticker(str(cached_event.get("ticker", ""))),
                "date": event_date,
                "transcript": str(cached_event.get("transcript", "")),
                "sentiment_posts": _normalize_cached_posts(raw_posts),
                "features": features,
                "label": label,
                "year": int(cached_event.get("year", pd.Timestamp(event_date).year)),
                "sentiment_features": cached_event.get("sentiment_features"),
            }
        )

    return normalized_events


def load_real_events() -> list[dict[str, Any]]:
    """Load all usable local events via the shared event loader."""

    cached_events = _load_events_from_cache()
    if cached_events is not None:
        return cached_events

    from data.loader import load_earnings_event

    events: list[dict[str, Any]] = []
    for ticker, event_date in discover_event_keys():
        try:
            event = load_earnings_event(ticker=ticker, earnings_date=event_date)
        except Exception:
            continue
        try:
            label = float(event.get("label", float("nan")))
        except Exception:
            continue
        if pd.isna(label):
            continue
        events.append(event)
    return events


def build_synthetic_events() -> list[dict[str, Any]]:
    """Build a small synthetic dataset for smoke tests and dry-runs."""

    synthetic_rows = [
        ("AAPL", "2023-01-20", 0.2, 0.05, 0.22, 0.010),
        ("MSFT", "2023-01-27", 0.4, 0.03, 0.24, -0.006),
        ("NVDA", "2023-02-10", 0.8, 0.07, 0.38, 0.021),
        ("AMZN", "2023-02-24", 1.2, -0.02, 0.31, -0.012),
        ("META", "2023-03-10", 1.7, 0.11, 0.29, 0.028),
        ("GOOG", "2023-03-24", 2.1, 0.04, 0.26, 0.014),
    ]
    events: list[dict[str, Any]] = []
    for ticker, event_date, sue, momentum, implied_vol, label in synthetic_rows:
        events.append(
            {
                "ticker": ticker,
                "date": event_date,
                "transcript": (
                    f"{ticker} management discussed earnings on {event_date} with "
                    "commentary on demand, margins, and guidance."
                ),
                "sentiment_posts": [
                    f"{ticker} beat expectations and market sentiment improved.",
                    f"Investors debated {ticker} guidance and valuation after earnings.",
                ],
                "features": {
                    "sue": float(sue),
                    "momentum": float(momentum),
                    "implied_vol": float(implied_vol),
                },
                "label": float(label),
            }
        )
    return events


def _read_cached_sp500_tickers() -> set[str]:
    """Read a cached S&P 500 ticker list when present."""

    if not SP500_CACHE_PATH.is_file():
        return set()
    try:
        data_frame = pd.read_csv(SP500_CACHE_PATH)
    except Exception:
        return set()
    if "ticker" not in data_frame.columns:
        return set()
    return {canonicalize_ticker(ticker) for ticker in data_frame["ticker"].tolist()}


def _fetch_sp500_tickers_from_web() -> set[str]:
    """Fetch the current S&P 500 constituent list from Wikipedia and cache it."""

    response = requests.get(
        SP500_SOURCE_URL,
        headers={"User-Agent": HTTP_USER_AGENT},
        timeout=20,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", attrs={"id": "constituents"})
    if table is None:
        table = soup.find("table", class_="wikitable")
    if table is None:
        raise ValueError("Could not locate the S&P 500 constituents table.")

    tickers: list[str] = []
    body = table.find("tbody")
    if body is None:
        raise ValueError("S&P 500 constituents table has no body.")

    for row in body.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        symbol = cells[0].get_text(strip=True)
        if symbol.lower() == "symbol":
            continue
        if symbol:
            tickers.append(canonicalize_ticker(symbol))

    unique_tickers = sorted(set(tickers))
    if not unique_tickers:
        raise ValueError("No S&P 500 tickers were parsed from the source table.")

    pd.DataFrame({"ticker": unique_tickers}).to_csv(SP500_CACHE_PATH, index=False)
    return set(unique_tickers)


def load_sp500_tickers() -> set[str]:
    """Load the S&P 500 ticker universe from cache or a live web fetch."""

    cached_tickers = _read_cached_sp500_tickers()
    if cached_tickers:
        return cached_tickers
    return _fetch_sp500_tickers_from_web()


def filter_events_by_universe(
    events: list[dict[str, Any]],
    universe: str | None,
) -> list[dict[str, Any]]:
    """Filter events to a configured ticker universe."""

    if universe is None or universe.lower() in {"", "all"}:
        return events
    if universe.lower() != "sp500":
        raise ValueError(f"Unsupported universe: {universe}")

    sp500_tickers = load_sp500_tickers()
    return [
        event
        for event in events
        if canonicalize_ticker(str(event.get("ticker", ""))) in sp500_tickers
    ]


def summarize_event_coverage(events: list[dict[str, Any]]) -> str:
    """Summarize event coverage by count, ticker count, and year frequency."""

    if not events:
        return "events=0, tickers=0, years=[]"

    years = [int(pd.Timestamp(event["date"]).year) for event in events]
    year_counts = Counter(years)
    year_summary = ", ".join(
        f"{year}:{year_counts[year]}" for year in sorted(year_counts)
    )
    ticker_count = len({canonicalize_ticker(str(event.get("ticker", ""))) for event in events})
    return f"events={len(events)}, tickers={ticker_count}, years={{ {year_summary} }}"


def split_events(
    events: list[dict[str, Any]],
    split_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split events according to config, using year-based or chronological logic."""

    strategy = str(split_config.get("strategy", "chronological")).lower()
    if strategy == "year":
        return _split_events_by_year(events, split_config)
    return _split_events_chronologically(events)


def _split_events_chronologically(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split events into chronological train/cal/test partitions."""

    sorted_events = sorted(events, key=lambda event: pd.Timestamp(event["date"]))
    num_events = len(sorted_events)
    if num_events < 3:
        raise ValueError("At least 3 events are required for a chronological split.")

    train_end = max(int(num_events * 0.70), 1)
    cal_size = max(int(num_events * 0.15), 1)
    cal_end = train_end + cal_size

    if cal_end >= num_events:
        cal_end = num_events - 1
    if train_end >= cal_end:
        train_end = max(1, cal_end - 1)

    train_events = sorted_events[:train_end]
    cal_events = sorted_events[train_end:cal_end]
    test_events = sorted_events[cal_end:]

    if not train_events or not cal_events or not test_events:
        raise ValueError("Chronological split produced an empty partition.")
    return train_events, cal_events, test_events


def _split_events_by_year(
    events: list[dict[str, Any]],
    split_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split events by explicit year partitions."""

    if not events:
        raise ValueError("No events are available for year-based splitting.")

    sorted_events = sorted(events, key=lambda event: pd.Timestamp(event["date"]))
    test_years = [int(year) for year in split_config.get("test_years", [])]
    calibration_years = [int(year) for year in split_config.get("calibration_years", [])]
    if not test_years:
        raise ValueError("Year-based split requires data.split.test_years in config.")

    if not calibration_years:
        calibration_years = [min(test_years) - 1]

    train_cutoff_year = min(calibration_years + test_years)
    train_events = [
        event
        for event in sorted_events
        if int(pd.Timestamp(event["date"]).year) < train_cutoff_year
        and int(pd.Timestamp(event["date"]).year) not in set(calibration_years + test_years)
    ]
    cal_events = [
        event
        for event in sorted_events
        if int(pd.Timestamp(event["date"]).year) in set(calibration_years)
    ]
    test_events = [
        event
        for event in sorted_events
        if int(pd.Timestamp(event["date"]).year) in set(test_years)
    ]

    if not train_events or not cal_events or not test_events:
        raise ValueError(
            "Year-based split produced an empty partition. "
            f"Requested calibration_years={calibration_years}, test_years={test_years}. "
            f"Available coverage: {summarize_event_coverage(events)}"
        )

    return train_events, cal_events, test_events
