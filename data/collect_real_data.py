"""Collect real structured financial event data into local CSV artifacts."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import pandas as pd
import requests


def _install_multitasking_shim() -> None:
    """Install a minimal ``multitasking`` shim for Python 3.8 compatibility."""

    shim = types.ModuleType("multitasking")

    def cpu_count() -> int:
        return os.cpu_count() or 1

    def set_max_threads(_max_threads: int) -> None:
        return None

    def task(func: Any) -> Any:
        return func

    shim.cpu_count = cpu_count
    shim.set_max_threads = set_max_threads
    shim.task = task
    sys.modules["multitasking"] = shim


try:
    import yfinance as yf
except Exception:  # pragma: no cover - environment-dependent optional import
    _install_multitasking_shim()
    sys.modules.pop("yfinance", None)
    try:
        import yfinance as yf
    except Exception:  # pragma: no cover - environment-dependent optional import
        yf = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FINANCIALS_PATH = DATA_DIR / "financials.csv"
SP500_TICKERS_PATH = DATA_DIR / "sp500_tickers.csv"
ERROR_LOG_PATH = DATA_DIR / "collect_errors.log"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
HTTP_TIMEOUT = 20
YFINANCE_SLEEP_SECONDS = 0.5
TRAILING_WINDOW_DAYS = 21
PRICE_LOOKBACK_DAYS = 60
DEFAULT_YEARS = [2021, 2022, 2023, 2024, 2025]
FINANCIAL_COLUMNS = [
    "ticker",
    "date",
    "earnings_surprise",
    "estimated_earnings",
    "actual_earnings",
    "sue",
    "implied_vol",
    "momentum",
]


def _canonicalize_ticker(ticker: str) -> str:
    """Normalize ticker symbols for file and API consistency."""

    return str(ticker).upper().replace(".", "-").strip()


def _log_error(ticker: str, error_message: str) -> None:
    """Append a ticker-specific error line to the collection log."""

    ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ERROR_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{ticker}: {error_message}\n")


def _safe_float(value: Any) -> float:
    """Convert values to float, returning ``nan`` on failure."""

    try:
        if pd.isna(value):
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def _load_existing_financials() -> pd.DataFrame:
    """Load and normalize the financials CSV into the current schema."""

    if FINANCIALS_PATH.is_file():
        financials = pd.read_csv(FINANCIALS_PATH)
    else:
        financials = pd.DataFrame(columns=FINANCIAL_COLUMNS)

    if "sue" not in financials.columns:
        if {"earnings_surprise", "std_dev_surprise"}.issubset(financials.columns):
            std_dev = pd.to_numeric(financials["std_dev_surprise"], errors="coerce")
            surprise = pd.to_numeric(financials["earnings_surprise"], errors="coerce")
            financials["sue"] = (surprise / std_dev.where(std_dev != 0.0)).fillna(0.0)
        else:
            financials["sue"] = pd.NA

    if "estimated_earnings" not in financials.columns:
        financials["estimated_earnings"] = pd.NA
    if "actual_earnings" not in financials.columns:
        financials["actual_earnings"] = pd.NA
    if "momentum" not in financials.columns:
        financials["momentum"] = pd.NA
    if "implied_vol" not in financials.columns:
        financials["implied_vol"] = pd.NA
    if "earnings_surprise" not in financials.columns:
        financials["earnings_surprise"] = pd.NA
    if "date" not in financials.columns:
        financials["date"] = pd.NA
    if "ticker" not in financials.columns:
        financials["ticker"] = pd.NA

    financials["ticker"] = financials["ticker"].astype(str).str.upper()
    financials["date"] = financials["date"].astype(str)
    for column in [
        "earnings_surprise",
        "estimated_earnings",
        "actual_earnings",
        "sue",
        "implied_vol",
        "momentum",
    ]:
        financials[column] = pd.to_numeric(financials[column], errors="coerce")

    financials = financials[FINANCIAL_COLUMNS].drop_duplicates(
        subset=["ticker", "date"],
        keep="last",
    )
    return financials


def _load_default_tickers() -> list[str]:
    """Load the default ticker universe from the cached S&P 500 list."""

    if not SP500_TICKERS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing cached S&P 500 list at {SP500_TICKERS_PATH}."
        )
    tickers = pd.read_csv(SP500_TICKERS_PATH)["ticker"].tolist()
    return [_canonicalize_ticker(ticker) for ticker in tickers]

def _ticker_has_financial_coverage(
    ticker: str,
    financials_df: pd.DataFrame,
    target_years: list[int],
) -> bool:
    """Check whether a ticker already has financial rows for all target years."""

    financial_years = set(
        pd.to_datetime(
            financials_df.loc[financials_df["ticker"] == ticker, "date"],
            errors="coerce",
        )
        .dt.year.dropna()
        .astype(int)
        .tolist()
    )
    return set(int(year) for year in target_years).issubset(financial_years)


def _fetch_earnings_dates_with_yfinance(ticker: str) -> pd.DataFrame:
    """Fetch historical earnings dates and EPS estimates from yfinance."""

    if yf is None:
        raise RuntimeError("yfinance is required for financial collection.")

    earnings_dates = yf.Ticker(ticker).get_earnings_dates(limit=40)
    if earnings_dates is None or earnings_dates.empty:
        return pd.DataFrame()

    earnings_dates = earnings_dates.copy()
    index = pd.to_datetime(earnings_dates.index, errors="coerce")
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    earnings_dates.index = index.normalize()
    return earnings_dates


def _download_close_history_for_event(
    ticker: str,
    event_date: pd.Timestamp,
) -> pd.Series:
    """Download daily close prices for the 60-day window ending on an event date."""

    if yf is None:
        return pd.Series(dtype="float64")

    start_date = (event_date - pd.Timedelta(days=PRICE_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end_date = (event_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    history = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if history.empty or "Close" not in history.columns:
        if not isinstance(history.columns, pd.MultiIndex):
            return pd.Series(dtype="float64")
        if ("Close", ticker) not in history.columns:
            return pd.Series(dtype="float64")
        closes = history[("Close", ticker)].copy()
    else:
        closes = history["Close"].copy()
        if isinstance(closes, pd.DataFrame):
            closes = closes.iloc[:, 0]

    close_index = pd.to_datetime(closes.index, errors="coerce")
    if getattr(close_index, "tz", None) is not None:
        close_index = close_index.tz_localize(None)
    closes.index = close_index.normalize()
    return closes.dropna()


def _compute_yfinance_financial_metrics(
    ticker: str,
    event_date: pd.Timestamp,
) -> tuple[float, float]:
    """Compute trailing momentum and annualized volatility from yfinance closes."""

    closes = _download_close_history_for_event(ticker, event_date)
    closes = closes.loc[closes.index <= event_date].dropna()
    if len(closes) < TRAILING_WINDOW_DAYS + 1:
        return math.nan, math.nan

    momentum = float(closes.iloc[-1] / closes.iloc[-22] - 1.0)
    implied_vol = float(
        closes.pct_change().rolling(TRAILING_WINDOW_DAYS).std().iloc[-1] * math.sqrt(252)
    )
    return implied_vol, momentum


def _fetch_surprises_for_ticker(
    ticker: str,
    target_years: list[int],
) -> pd.DataFrame:
    """Fetch earnings surprise rows for one ticker from yfinance."""

    earnings_dates = _fetch_earnings_dates_with_yfinance(ticker)
    if earnings_dates.empty:
        return pd.DataFrame(columns=FINANCIAL_COLUMNS)

    rows: list[dict[str, Any]] = []
    target_year_set = set(target_years)

    for event_date, row in earnings_dates.iterrows():
        if pd.isna(event_date) or int(event_date.year) not in target_year_set:
            continue

        estimated = _safe_float(row.get("EPS Estimate"))
        actual = _safe_float(row.get("Reported EPS"))
        if math.isnan(estimated) or math.isnan(actual):
            continue

        earnings_surprise = float(actual - estimated)

        implied_vol, momentum = _compute_yfinance_financial_metrics(
            ticker=ticker,
            event_date=pd.Timestamp(event_date).normalize(),
        )

        rows.append(
            {
                "ticker": _canonicalize_ticker(ticker),
                "date": pd.Timestamp(event_date).strftime("%Y-%m-%d"),
                "earnings_surprise": earnings_surprise,
                "estimated_earnings": float(estimated),
                "actual_earnings": float(actual),
                # SUE is recomputed later from each ticker's historical surprise
                # dispersion so it remains standardized and leakage-safe.
                "sue": math.nan,
                "implied_vol": implied_vol,
                "momentum": momentum,
            }
        )

    if not rows:
        return pd.DataFrame(columns=FINANCIAL_COLUMNS)

    financials = pd.DataFrame(rows)
    return financials[FINANCIAL_COLUMNS].drop_duplicates(
        subset=["ticker", "date"],
        keep="last",
    )


def _save_financials(financials_df: pd.DataFrame) -> None:
    """Persist the normalized financial dataframe to disk."""

    financials_df = financials_df[FINANCIAL_COLUMNS].sort_values(by=["ticker", "date"])
    financials_df.to_csv(FINANCIALS_PATH, index=False)


def _merge_new_rows(existing_df: pd.DataFrame, new_df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    """Merge new rows into an existing dataframe and drop duplicates."""

    if new_df.empty:
        return existing_df
    merged = pd.concat([existing_df, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=subset, keep="last")
    return merged


def _fetch_price_history_with_yfinance(
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch daily close prices with yfinance when available."""

    if yf is None:
        return pd.DataFrame()
    try:
        history = yf.Ticker(ticker).history(
            start=start_date.strftime("%Y-%m-%d"),
            end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=False,
        )
    except Exception:
        return pd.DataFrame()
    if history.empty or "Close" not in history.columns:
        return pd.DataFrame()
    history = history[["Close"]].copy()
    index = pd.to_datetime(history.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    history.index = index.normalize()
    return history


def _fetch_price_history_from_yahoo(
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch daily close prices from Yahoo's chart JSON endpoint as a fallback."""

    period1 = int(start_date.tz_localize("UTC").timestamp())
    period2 = int((end_date + pd.Timedelta(days=1)).tz_localize("UTC").timestamp())
    response = requests.get(
        YAHOO_CHART_URL.format(ticker=ticker),
        params={
            "period1": period1,
            "period2": period2,
            "interval": "1d",
            "includePrePost": "false",
            "events": "div,splits",
        },
        headers={"User-Agent": "earnings-forecast/1.0"},
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    result = payload.get("chart", {}).get("result", [])
    if not result:
        return pd.DataFrame()

    quote = result[0].get("indicators", {}).get("quote", [])
    timestamps = result[0].get("timestamp", [])
    if not quote or not timestamps:
        return pd.DataFrame()

    history = pd.DataFrame(
        {"Close": quote[0].get("close", [])},
        index=pd.to_datetime(timestamps, unit="s"),
    )
    history = history.dropna(subset=["Close"], how="all")
    history.index = history.index.tz_localize(None).normalize()
    return history


def _fetch_close_history(
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch close-price history, preferring yfinance but falling back if needed."""

    history = _fetch_price_history_with_yfinance(ticker, start_date, end_date)
    if not history.empty:
        return history
    try:
        return _fetch_price_history_from_yahoo(ticker, start_date, end_date)
    except Exception:
        return pd.DataFrame()


def _compute_trailing_metrics(
    close_history: pd.DataFrame,
    event_date: pd.Timestamp,
) -> tuple[float, float]:
    """Compute annualized 21-day return volatility and 21-day momentum."""

    history = close_history.loc[close_history.index <= event_date, "Close"].dropna()
    if len(history) < TRAILING_WINDOW_DAYS + 1:
        return math.nan, math.nan

    trailing_closes = history.iloc[-(TRAILING_WINDOW_DAYS + 1) :]
    daily_returns = trailing_closes.pct_change().dropna()
    if len(daily_returns) < TRAILING_WINDOW_DAYS:
        return math.nan, math.nan

    implied_vol = float(daily_returns.std(ddof=1) * math.sqrt(252))
    momentum = float(trailing_closes.iloc[-1] / trailing_closes.iloc[0] - 1.0)
    return implied_vol, momentum


def _update_financial_volatility(financials_df: pd.DataFrame) -> pd.DataFrame:
    """Populate implied volatility and momentum for each financial row."""

    if financials_df.empty:
        return financials_df

    updated = financials_df.copy()
    updated["date"] = updated["date"].astype(str)
    for ticker, ticker_rows in updated.groupby("ticker"):
        ticker_indices = ticker_rows.index.tolist()
        date_series = pd.to_datetime(ticker_rows["date"], errors="coerce").dropna()
        if date_series.empty:
            continue

        start_date = date_series.min() - pd.Timedelta(days=PRICE_LOOKBACK_DAYS)
        end_date = date_series.max()
        history = _fetch_close_history(str(ticker), start_date.normalize(), end_date.normalize())
        if history.empty:
            updated.loc[ticker_indices, "implied_vol"] = pd.to_numeric(
                updated.loc[ticker_indices, "implied_vol"],
                errors="coerce",
            )
            updated.loc[ticker_indices, "momentum"] = pd.to_numeric(
                updated.loc[ticker_indices, "momentum"],
                errors="coerce",
            )
            continue

        for row_index in ticker_indices:
            event_date = pd.to_datetime(updated.at[row_index, "date"], errors="coerce")
            if pd.isna(event_date):
                updated.at[row_index, "implied_vol"] = math.nan
                updated.at[row_index, "momentum"] = math.nan
                continue

            implied_vol, momentum = _compute_trailing_metrics(history, event_date.normalize())
            updated.at[row_index, "implied_vol"] = implied_vol
            updated.at[row_index, "momentum"] = momentum

    for column in ["earnings_surprise", "estimated_earnings", "actual_earnings", "sue", "implied_vol", "momentum"]:
        updated[column] = pd.to_numeric(updated[column], errors="coerce")
    return updated


def _recompute_sue(financials_df: pd.DataFrame) -> pd.DataFrame:
    """Recompute SUE from prior same-ticker earnings surprises only.

    This standardizes each event's earnings surprise by the sample standard
    deviation of that ticker's *prior* surprise history, avoiding look-ahead
    leakage from future earnings events.
    """

    if financials_df.empty:
        return financials_df

    updated = financials_df.copy()
    updated["ticker"] = updated["ticker"].astype(str).str.upper()
    updated["date"] = pd.to_datetime(updated["date"], errors="coerce")
    updated["earnings_surprise"] = pd.to_numeric(
        updated["earnings_surprise"],
        errors="coerce",
    )
    updated["sue"] = 0.0

    for _, ticker_rows in updated.groupby("ticker", sort=False):
        ordered_rows = ticker_rows.sort_values("date")
        prior_surprises: list[float] = []

        for row_index, row in ordered_rows.iterrows():
            current_surprise = _safe_float(row.get("earnings_surprise"))
            if math.isnan(current_surprise):
                updated.at[row_index, "sue"] = 0.0
                continue

            if len(prior_surprises) >= 2:
                std_dev = float(pd.Series(prior_surprises, dtype="float64").std(ddof=1))
                if not math.isnan(std_dev) and std_dev > 0.0:
                    updated.at[row_index, "sue"] = float(current_surprise / std_dev)
                else:
                    updated.at[row_index, "sue"] = 0.0
            else:
                updated.at[row_index, "sue"] = 0.0

            prior_surprises.append(current_surprise)

    updated["date"] = updated["date"].dt.strftime("%Y-%m-%d")
    updated["sue"] = pd.to_numeric(updated["sue"], errors="coerce").fillna(0.0)
    return updated


def _coverage_summary(financials_df: pd.DataFrame) -> str:
    """Build the end-of-run financial coverage summary string."""

    financial_tickers = financials_df["ticker"].nunique() if not financials_df.empty else 0
    financial_years = set(
        pd.to_datetime(financials_df.get("date"), errors="coerce")
        .dt.year.dropna()
        .astype(int)
        .tolist()
    )
    years_covered = sorted(financial_years)

    return (
        f"tickers with financials:  {financial_tickers}\n"
        f"total financial rows:     {len(financials_df)}\n"
        f"years covered:            {years_covered}"
    )


def collect_real_data(tickers: list[str], years: list[int]) -> None:
    """Collect real structured financial features into ``financials.csv``.

    Args:
        tickers: Tickers to fetch from yfinance and Yahoo Finance.
        years: Target years to keep.
    """

    financials_df = _load_existing_financials()

    total_tickers = len(tickers)
    starting_financial_rows = len(financials_df)
    financial_failures = 0
    for index, raw_ticker in enumerate(tickers, start=1):
        ticker = _canonicalize_ticker(raw_ticker)
        print(f"Fetching {ticker} ({index}/{total_tickers})...")

        if _ticker_has_financial_coverage(ticker, financials_df, years):
            continue

        try:
            new_financials = _fetch_surprises_for_ticker(ticker, years)
            financials_df = _merge_new_rows(
                financials_df,
                new_financials,
                subset=["ticker", "date"],
            )
            financials_df = _recompute_sue(financials_df)
            _save_financials(financials_df)
        except Exception as exc:
            _log_error(ticker, f"financials: {exc}")
            financial_failures += 1

        time.sleep(YFINANCE_SLEEP_SECONDS)

    financials_df = _update_financial_volatility(financials_df)
    financials_df = _recompute_sue(financials_df)
    _save_financials(financials_df)
    print(
        "rows added this run:"
        f" financials={len(financials_df) - starting_financial_rows}"
    )
    if financial_failures:
        print(
            "warning: financial collection failed "
            f"for {financial_failures} ticker requests."
        )
    print(_coverage_summary(financials_df))


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the collector."""

    parser = argparse.ArgumentParser(description="Collect real earnings event data.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Optional ticker subset, e.g. --tickers AAPL MSFT GOOGL",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        help="Optional target years, e.g. --years 2024 2025",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    selected_tickers = (
        [_canonicalize_ticker(ticker) for ticker in arguments.tickers]
        if arguments.tickers
        else _load_default_tickers()
    )
    selected_years = arguments.years if arguments.years else DEFAULT_YEARS
    collect_real_data(selected_tickers, selected_years)
