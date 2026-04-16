"""Utilities for loading multimodal earnings event data."""

from __future__ import annotations

import math
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import requests
except Exception:  # pragma: no cover - environment-dependent optional import
    requests = None

try:
    import yfinance as yf
except Exception:  # pragma: no cover - environment-dependent optional import
    yf = None


DEFAULT_IMPLIED_VOL = 0.0
PRICE_LOOKBACK_DAYS = 45
PRICE_LOOKAHEAD_DAYS = 7
MOMENTUM_LOOKBACK_DAYS = 21
REDDIT_POST_LIMIT = 100
REDDIT_REQUEST_TIMEOUT = 20
REDDIT_USER_AGENT = "earnings-research-bot/1.0"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


def _data_file_path(filename: str) -> Path:
    """Return the absolute path to a local data artifact."""

    return Path(__file__).resolve().parent / filename


def _normalize_date(date_value: str | pd.Timestamp) -> pd.Timestamp:
    """Convert a date value into a normalized pandas timestamp."""

    return pd.Timestamp(date_value).normalize()


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float and fall back to a default on failure."""

    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _to_unix_timestamp(date_value: pd.Timestamp) -> int:
    """Convert a pandas timestamp into a Unix timestamp in UTC seconds."""

    timestamp = pd.Timestamp(date_value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.timestamp())


def _reddit_window_bounds(event_date: pd.Timestamp) -> tuple[int, int]:
    """Return a non-leaky Reddit feature window around an earnings event.

    We include the day before the earnings date and the earnings date itself,
    but exclude the following calendar day so sentiment features cannot absorb
    obviously post-event discussion.
    """

    normalized_event_date = _normalize_date(event_date)
    window_start = _to_unix_timestamp(normalized_event_date) - 86400
    window_end = _to_unix_timestamp(normalized_event_date + timedelta(days=1)) - 1
    return window_start, window_end


def _fetch_price_history(ticker: str, earnings_date: pd.Timestamp) -> pd.DataFrame:
    """Fetch OHLCV history around an earnings date using yfinance."""

    start_date = (earnings_date - timedelta(days=PRICE_LOOKBACK_DAYS)).strftime(
        "%Y-%m-%d"
    )
    end_date = (earnings_date + timedelta(days=PRICE_LOOKAHEAD_DAYS)).strftime(
        "%Y-%m-%d"
    )

    if yf is not None:
        try:
            history = yf.Ticker(ticker).history(
                start=start_date,
                end=end_date,
                auto_adjust=False,
            )
            if not history.empty:
                history = history.copy()
                normalized_index = pd.to_datetime(history.index)
                if getattr(normalized_index, "tz", None) is not None:
                    normalized_index = normalized_index.tz_localize(None)
                history.index = normalized_index.normalize()
                return history
        except Exception:
            pass

    return _fetch_price_history_from_yahoo_chart(
        ticker=ticker,
        start_date=pd.Timestamp(start_date),
        end_date=pd.Timestamp(end_date),
    )


def _fetch_price_history_from_yahoo_chart(
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch OHLCV history from Yahoo's chart JSON endpoint as a fallback."""

    if requests is None:
        return pd.DataFrame()

    period1 = _to_unix_timestamp(start_date.normalize())
    period2 = _to_unix_timestamp((end_date.normalize() + timedelta(days=1)))
    url = YAHOO_CHART_URL.format(ticker=ticker)
    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "includePrePost": "false",
        "events": "div,splits",
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers={"User-Agent": REDDIT_USER_AGENT},
            timeout=REDDIT_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        result = payload.get("chart", {}).get("result", [])
        if not result:
            return pd.DataFrame()

        result_payload = result[0]
        timestamps = result_payload.get("timestamp", [])
        indicators = result_payload.get("indicators", {}).get("quote", [])
        if not timestamps or not indicators:
            return pd.DataFrame()

        quote_payload = indicators[0]
        history = pd.DataFrame(
            {
                "Open": quote_payload.get("open", []),
                "High": quote_payload.get("high", []),
                "Low": quote_payload.get("low", []),
                "Close": quote_payload.get("close", []),
                "Volume": quote_payload.get("volume", []),
            },
            index=pd.to_datetime(timestamps, unit="s"),
        )
        history = history.dropna(subset=["Close"], how="all")
        history.index = history.index.tz_localize(None).normalize()
        return history
    except Exception:
        return pd.DataFrame()


def _compute_label_and_momentum(
    history: pd.DataFrame,
    earnings_date: pd.Timestamp,
) -> tuple[float, float]:
    """Compute next-day log return label and trailing 21-day momentum."""

    if history.empty or "Close" not in history.columns:
        return math.nan, 0.0

    trading_days = history.index.sort_values()
    event_candidates = trading_days[trading_days >= earnings_date]
    if len(event_candidates) < 2:
        return math.nan, 0.0

    event_day = event_candidates[0]
    next_day = event_candidates[1]

    try:
        event_close = _safe_float(history.loc[event_day, "Close"], default=math.nan)
        next_close = _safe_float(history.loc[next_day, "Close"], default=math.nan)
        label = math.log(next_close / event_close)
    except Exception:
        label = math.nan

    try:
        event_position = history.index.get_loc(event_day)
        if isinstance(event_position, slice):
            event_position = event_position.start
        lookback_position = int(event_position) - MOMENTUM_LOOKBACK_DAYS
        if lookback_position < 0:
            momentum = 0.0
        else:
            lookback_close = _safe_float(
                history.iloc[lookback_position]["Close"],
                default=math.nan,
            )
            momentum = (event_close / lookback_close) - 1.0
    except Exception:
        momentum = 0.0

    return float(label), float(momentum)


def _load_transcript(ticker: str, earnings_date: pd.Timestamp) -> str:
    """Load an earnings call transcript from the local transcripts CSV."""

    transcript_path = _data_file_path("transcripts.csv")
    transcripts = pd.read_csv(transcript_path)
    transcripts["ticker"] = transcripts["ticker"].astype(str).str.upper()
    transcripts["date"] = pd.to_datetime(transcripts["date"]).dt.normalize()

    mask = (
        (transcripts["ticker"] == ticker.upper())
        & (transcripts["date"] == earnings_date)
    )
    if not mask.any():
        return ""

    text_value = transcripts.loc[mask, "text"].iloc[0]
    if pd.isna(text_value):
        return ""
    return str(text_value)


def _load_financial_features(
    ticker: str,
    earnings_date: pd.Timestamp,
) -> dict[str, float]:
    """Load structured financial features from the local financials CSV."""

    financials_path = _data_file_path("financials.csv")
    if not financials_path.is_file():
        return {
            "sue": 0.0,
            "momentum": 0.0,
            "implied_vol": DEFAULT_IMPLIED_VOL,
        }

    financials = pd.read_csv(financials_path)
    financials["ticker"] = financials["ticker"].astype(str).str.upper()
    financials["date"] = pd.to_datetime(financials["date"]).dt.normalize()

    mask = (
        (financials["ticker"] == ticker.upper())
        & (financials["date"] == earnings_date)
    )
    if not mask.any():
        return {
            "sue": 0.0,
            "momentum": 0.0,
            "implied_vol": DEFAULT_IMPLIED_VOL,
        }

    row = financials.loc[mask].iloc[0]
    earnings_surprise = _safe_float(row.get("earnings_surprise", math.nan), default=math.nan)
    std_dev_surprise = _safe_float(row.get("std_dev_surprise", 0.0))
    momentum = _safe_float(row.get("momentum", math.nan), default=math.nan)
    implied_vol = _safe_float(row.get("implied_vol", DEFAULT_IMPLIED_VOL))

    prior_mask = (
        (financials["ticker"] == ticker.upper())
        & (financials["date"] < earnings_date)
    )
    prior_surprises = (
        pd.to_numeric(
            financials.loc[prior_mask, "earnings_surprise"],
            errors="coerce",
        )
        .dropna()
        .tolist()
    )
    if not math.isnan(earnings_surprise) and len(prior_surprises) >= 2:
        prior_std = float(pd.Series(prior_surprises, dtype="float64").std(ddof=1))
        if not math.isnan(prior_std) and prior_std > 0.0:
            sue = earnings_surprise / prior_std
        else:
            sue = 0.0
    elif std_dev_surprise > 0.0 and not math.isnan(earnings_surprise):
        sue = earnings_surprise / std_dev_surprise
    else:
        sue = 0.0

    if math.isnan(momentum):
        momentum = 0.0

    return {
        "sue": float(sue),
        "momentum": float(momentum),
        "implied_vol": float(implied_vol),
    }


def _reddit_json(url: str) -> dict[str, Any]:
    """Fetch and decode a Reddit JSON endpoint for reuse across nested calls."""

    if requests is None:
        raise RuntimeError("The 'requests' package is required for Reddit scraping.")

    try:
        response = requests.get(
            url,
            headers={"User-Agent": REDDIT_USER_AGENT},
            timeout=REDDIT_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Reddit JSON response must decode to a dictionary.")
        return payload
    finally:
        time.sleep(1)


def fetch_reddit_posts(ticker: str, date: str | pd.Timestamp) -> list[str]:
    """Fetch Reddit posts from the day before through the earnings date only."""

    event_date = _normalize_date(date)
    lower_bound, upper_bound = _reddit_window_bounds(event_date)
    url = (
        f"https://www.reddit.com/search.json?q={ticker}&sort=new&limit=100&type=link"
    )
    posts: list[str] = []

    try:
        data = _reddit_json(url)
        children = data.get("data", {}).get("children", [])
        for post in children:
            payload = post.get("data", {})
            created_utc = _safe_float(payload.get("created_utc"), default=math.nan)
            if math.isnan(created_utc):
                continue
            if created_utc < lower_bound or created_utc > upper_bound:
                continue

            title = str(payload.get("title", "") or "").strip()
            selftext = str(payload.get("selftext", "") or "").strip()
            post_text = "\n".join(part for part in (title, selftext) if part).strip()
            if post_text:
                posts.append(post_text)
    except Exception:
        return []

    return posts


def load_earnings_event(ticker: str, earnings_date: str) -> dict[str, Any]:
    """Load transcript, Reddit posts, structured features, and the return label.

    Args:
        ticker: Public ticker symbol for the company.
        earnings_date: Earnings event date parseable by ``pandas.Timestamp``.

    Returns:
        A dictionary containing the event metadata, transcript text, Reddit post
        texts, numeric features, and next-day log return label.
    """

    normalized_ticker = ticker.upper()
    event_date = _normalize_date(earnings_date)

    transcript = ""
    sentiment_posts: list[str] = []
    sue = 0.0
    momentum = 0.0
    implied_vol = DEFAULT_IMPLIED_VOL
    label = math.nan

    try:
        price_history = _fetch_price_history(normalized_ticker, event_date)
        label, momentum = _compute_label_and_momentum(price_history, event_date)
    except Exception:
        label = math.nan
        momentum = 0.0

    try:
        transcript = _load_transcript(normalized_ticker, event_date)
    except Exception:
        transcript = ""

    try:
        sentiment_posts = fetch_reddit_posts(normalized_ticker, event_date)
    except Exception:
        sentiment_posts = []

    try:
        financial_features = _load_financial_features(normalized_ticker, event_date)
        sue = financial_features["sue"]
        if not math.isnan(financial_features["momentum"]):
            momentum = financial_features["momentum"]
        implied_vol = financial_features["implied_vol"]
    except Exception:
        sue = 0.0
        implied_vol = DEFAULT_IMPLIED_VOL

    return {
        "ticker": normalized_ticker,
        "date": event_date.strftime("%Y-%m-%d"),
        "transcript": transcript,
        "sentiment_posts": sentiment_posts,
        "features": {
            "sue": float(sue),
            "momentum": float(momentum),
            "implied_vol": float(implied_vol),
        },
        "label": float(label),
    }
