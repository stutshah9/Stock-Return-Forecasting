"""Load Yahoo Finance price history and derive event-level market features."""

from __future__ import annotations

import inspect
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.data.normalize import normalize_ticker
from src.utils.io_utils import ensure_dir, save_dataframe
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

try:  # pragma: no cover - networked dependency
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


PRICE_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume"]


@dataclass(slots=True)
class YFinanceLoadConfig:
    """Configuration for Yahoo Finance extraction."""

    raw_price_dir: Path = Path("data/raw/yfinance_prices")
    processed_output_path: Path = Path("data/processed/market_features.parquet")
    start_date: str = "2023-01-01"
    end_date: str = "2025-12-31"
    refresh: bool = False


def _yahoo_symbol_candidates(ticker: str) -> list[str]:
    """Return Yahoo-compatible symbol candidates for a logical ticker."""
    normalized = normalize_ticker(ticker) or ticker
    candidates = [normalized]
    dashed = normalized.replace(".", "-")
    if dashed not in candidates:
        candidates.append(dashed)
    return candidates


def _standardize_history(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize a Yahoo Finance history frame into a stable OHLCV schema."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    history = frame.copy()
    if isinstance(history.columns, pd.MultiIndex):
        history.columns = [column[0] if isinstance(column, tuple) else column for column in history.columns]
    history = history.reset_index().rename(
        columns={
            "index": "date",
            "Date": "date",
            "Datetime": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    history["ticker"] = normalize_ticker(ticker)
    history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.normalize()
    for column in ["open", "high", "low", "close", "volume"]:
        history[column] = pd.to_numeric(history[column], errors="coerce")
    history = history[PRICE_COLUMNS].dropna(subset=["ticker", "date", "close"])
    return history.sort_values("date").reset_index(drop=True)


def _download_kwargs(
    ticker: str,
    start_date: str,
    end_date: str,
    proxy: str | None,
) -> dict[str, object]:
    """Build kwargs compatible with the installed yfinance.download signature."""
    kwargs: dict[str, object] = {
        "tickers": ticker,
        "start": start_date,
        "end": (pd.Timestamp(end_date) + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        "progress": False,
        "auto_adjust": False,
        "group_by": "column",
        "threads": False,
    }
    try:
        signature = inspect.signature(yf.download)
        supported = set(signature.parameters)
    except Exception:  # pragma: no cover - defensive fallback
        supported = set()

    if "tickers" not in supported and "tickers" in kwargs:
        kwargs["ticker"] = kwargs.pop("tickers")
    if proxy and "proxy" in supported:
        kwargs["proxy"] = proxy
    return kwargs


def fetch_price_history(
    ticker: str,
    start_date: str,
    end_date: str,
    raw_price_dir: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch and cache daily OHLCV history for a single ticker."""
    cache_dir = ensure_dir(raw_price_dir)
    cache_path = cache_dir / f"{ticker}.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)
    if yf is None:
        raise RuntimeError("yfinance is required to fetch market data.")
    proxy = os.getenv("YFINANCE_PROXY") or None
    last_exception: Exception | None = None
    for yahoo_symbol in _yahoo_symbol_candidates(ticker):
        try:
            history = yf.download(**_download_kwargs(yahoo_symbol, start_date, end_date, proxy))
        except TypeError as exc:
            # Older yfinance releases do not accept `proxy`; retry without it.
            if proxy and "proxy" in str(exc):
                try:
                    history = yf.download(**_download_kwargs(yahoo_symbol, start_date, end_date, None))
                except Exception as retry_exc:  # pragma: no cover - networked dependency
                    last_exception = retry_exc
                    continue
            else:
                last_exception = exc
                continue
        except Exception as exc:  # pragma: no cover - networked dependency
            last_exception = exc
            continue

        standardized = _standardize_history(history, ticker=ticker)
        if standardized.empty:
            continue
        standardized.to_parquet(cache_path, index=False)
        LOGGER.info(
            "Cached %s price rows for %s using Yahoo symbol %s to %s",
            len(standardized),
            ticker,
            yahoo_symbol,
            cache_path,
        )
        return standardized

    if last_exception is not None:
        raise RuntimeError(f"Yahoo history fetch failed for {ticker}: {last_exception}") from last_exception
    LOGGER.warning("Yahoo returned no price history for %s across candidates %s", ticker, _yahoo_symbol_candidates(ticker))
    return pd.DataFrame(columns=PRICE_COLUMNS)


def fetch_company_metadata(ticker: str) -> dict[str, object]:
    """Fetch basic company metadata from Yahoo Finance when available."""
    if yf is None:  # pragma: no cover
        return {"ticker": ticker}
    info: dict[str, object] = {}
    last_exception: Exception | None = None
    used_symbol = ticker
    for yahoo_symbol in _yahoo_symbol_candidates(ticker):
        ticker_obj = yf.Ticker(yahoo_symbol)
        try:  # pragma: no cover - networked dependency
            info = ticker_obj.get_info()
        except Exception as exc:
            last_exception = exc
            continue
        if info:
            used_symbol = yahoo_symbol
            break
    if not info and last_exception is not None:
        LOGGER.warning("Yahoo metadata fetch failed for %s: %s", ticker, last_exception)
    return {
        "ticker": ticker,
        "yf_symbol_used": used_symbol if info else None,
        "yf_long_name": info.get("longName") or info.get("shortName"),
        "yf_sector": info.get("sector"),
        "yf_industry": info.get("industry"),
        "yf_exchange": info.get("exchange"),
        "yf_currency": info.get("currency"),
        "yf_country": info.get("country"),
        "yf_market_cap": info.get("marketCap"),
        "yf_quote_type": info.get("quoteType"),
    }


def _select_event_rows(history: pd.DataFrame, event_date: pd.Timestamp) -> tuple[pd.Series | None, pd.Series | None, pd.Series | None]:
    """Return previous-close row, event-close row, and next-close row."""
    if history.empty:
        return None, None, None
    event_candidates = history.index[history["date"] >= event_date]
    if len(event_candidates) == 0:
        return None, None, None
    event_position = int(event_candidates[0])
    previous_row = history.iloc[event_position - 1] if event_position >= 1 else None
    event_row = history.iloc[event_position]
    next_row = history.iloc[event_position + 1] if event_position + 1 < len(history) else None
    return previous_row, event_row, next_row


def _log_return(current_price: float | int | None, base_price: float | int | None) -> float | None:
    """Compute a log return safely."""
    if current_price in {None, 0} or base_price in {None, 0}:
        return None
    if pd.isna(current_price) or pd.isna(base_price):
        return None
    return float(math.log(float(current_price) / float(base_price)))


def _rolling_pre_event_features(history: pd.DataFrame, previous_position: int | None) -> tuple[float | None, float | None, float | None, float | None]:
    """Compute pre-event returns, volatility, and volume using only prior sessions."""
    if previous_position is None or previous_position < 0:
        return None, None, None, None
    pre_history = history.iloc[: previous_position + 1].copy()
    if pre_history.empty:
        return None, None, None, None
    previous_close = pre_history["close"].iloc[-1]

    def trailing_return(window: int) -> float | None:
        if len(pre_history) < 2:
            return None
        lookback_index = max(0, len(pre_history) - 1 - window)
        base_close = pre_history["close"].iloc[lookback_index]
        return _log_return(previous_close, base_close)

    close_returns = pre_history["close"].pct_change().dropna()
    rolling_volatility = None
    if len(close_returns) >= 2:
        rolling_volatility = float(close_returns.tail(20).std(ddof=0) * (252 ** 0.5))
    average_volume = float(pre_history["volume"].tail(20).mean()) if "volume" in pre_history.columns else None
    return trailing_return(5), trailing_return(20), rolling_volatility, average_volume


def build_market_features(events_df: pd.DataFrame, price_history_df: pd.DataFrame) -> pd.DataFrame:
    """Build event-aligned market features from cached daily price history."""
    if events_df.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "ticker",
                "event_date",
                "event_trading_date",
                "previous_close",
                "event_close",
                "next_trading_close",
                "log_return_target",
                "pre_return_5d",
                "pre_return_20d",
                "rolling_volatility_20d",
                "average_volume_20d",
            ]
        )

    histories = {
        ticker: frame.sort_values("date").reset_index(drop=True)
        for ticker, frame in price_history_df.groupby("ticker", sort=False)
    }
    records: list[dict[str, object]] = []
    for _, event in events_df.iterrows():
        ticker = event["ticker"]
        event_date = pd.Timestamp(event["event_date"]).normalize()
        history = histories.get(ticker, pd.DataFrame(columns=PRICE_COLUMNS))
        previous_row, event_row, next_row = _select_event_rows(history, event_date)
        previous_close = previous_row["close"] if previous_row is not None else None
        previous_position = None if previous_row is None else int(previous_row.name)
        pre_return_5d, pre_return_20d, rolling_volatility, average_volume = _rolling_pre_event_features(
            history,
            previous_position=previous_position,
        )
        records.append(
            {
                "event_id": event["event_id"],
                "ticker": ticker,
                "event_date": event_date,
                "event_trading_date": event_row["date"] if event_row is not None else pd.NaT,
                "previous_close": previous_close,
                "event_close": event_row["close"] if event_row is not None else None,
                "next_trading_close": next_row["close"] if next_row is not None else None,
                "log_return_target": (
                    _log_return(next_row["close"], event_row["close"])
                    if event_row is not None and next_row is not None
                    else None
                ),
                "pre_return_5d": pre_return_5d,
                "pre_return_20d": pre_return_20d,
                "rolling_volatility_20d": rolling_volatility,
                "average_volume_20d": average_volume,
            }
        )
    return pd.DataFrame(records).sort_values(["event_date", "ticker"]).reset_index(drop=True)


def load_yfinance_data(
    transcripts_df: pd.DataFrame,
    config: YFinanceLoadConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch price history, derive market features, and collect company metadata."""
    tickers = sorted({ticker for ticker in transcripts_df["ticker"].dropna().unique() if ticker})
    all_histories: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, object]] = []
    for ticker in tickers:
        try:
            history = fetch_price_history(
                ticker=ticker,
                start_date=config.start_date,
                end_date=config.end_date,
                raw_price_dir=config.raw_price_dir,
                refresh=config.refresh,
            )
        except Exception as exc:
            LOGGER.warning("Yahoo price fetch failed for %s: %s", ticker, exc)
            history = pd.DataFrame(columns=PRICE_COLUMNS)
        if not history.empty:
            all_histories.append(history)
        metadata_rows.append(fetch_company_metadata(ticker))

    price_history_df = (
        pd.concat(all_histories, ignore_index=True)
        if all_histories
        else pd.DataFrame(columns=PRICE_COLUMNS)
    )
    market_features_df = build_market_features(transcripts_df, price_history_df)
    save_dataframe(market_features_df, config.processed_output_path)
    metadata_df = pd.DataFrame(metadata_rows).drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    LOGGER.info("Built market features for %s events", len(market_features_df))
    return price_history_df, market_features_df, metadata_df
