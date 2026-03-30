"""End-to-end dataset builder for transcripts, market data, SEC facts, and Reddit activity."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd

from src.data.join_data import JoinOutputConfig, build_company_level_dataset, build_event_level_dataset, save_joined_datasets
from src.data.load_reddit_comments import RedditCommentsLoadConfig, load_reddit_comments
from src.data.load_reddit_posts import RedditPostsLoadConfig, load_reddit_posts
from src.data.load_sec_companyfacts import SECLoaderConfig, load_sec_fundamentals
from src.data.load_transcripts import TranscriptLoadConfig, load_transcripts
from src.data.load_yfinance import YFinanceLoadConfig, load_yfinance_data
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class DatasetBuildConfig:
    """Top-level configuration for the dataset creation pipeline."""

    start_date: str | None = None
    end_date: str | None = None
    years_history: int = 3
    max_tickers: int | None = None
    refresh: bool = False


def _resolve_date_window(start_date: str | None, end_date: str | None, years_history: int) -> tuple[str, str]:
    """Resolve the default date window used across all loaders."""
    end_timestamp = pd.Timestamp.today().normalize() if end_date is None else pd.Timestamp(end_date).normalize()
    start_timestamp = (
        end_timestamp - pd.DateOffset(years=years_history)
        if start_date is None
        else pd.Timestamp(start_date).normalize()
    )
    return start_timestamp.strftime("%Y-%m-%d"), end_timestamp.strftime("%Y-%m-%d")


def build_datasets(config: DatasetBuildConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full dataset build and return company-level and event-level outputs."""
    start_date, end_date = _resolve_date_window(config.start_date, config.end_date, config.years_history)
    LOGGER.info("Building datasets for %s through %s", start_date, end_date)

    transcripts_df = load_transcripts(
        TranscriptLoadConfig(
            start_date=start_date,
            end_date=end_date,
            refresh=config.refresh,
        )
    )
    if transcripts_df.empty:
        raise ValueError("No transcript events were available after normalization and date filtering.")

    if config.max_tickers is not None:
        allowed_tickers = sorted(transcripts_df["ticker"].dropna().unique())[: config.max_tickers]
        transcripts_df = transcripts_df[transcripts_df["ticker"].isin(allowed_tickers)].reset_index(drop=True)
        LOGGER.info("Restricted transcript universe to %s tickers", len(allowed_tickers))

    social_start_date = (pd.Timestamp(start_date) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")

    price_history_df, market_features_df, yfinance_metadata_df = load_yfinance_data(
        transcripts_df=transcripts_df,
        config=YFinanceLoadConfig(
            start_date=(pd.Timestamp(start_date) - pd.Timedelta(days=40)).strftime("%Y-%m-%d"),
            end_date=end_date,
            refresh=config.refresh,
        ),
    )
    sec_fundamentals_df = load_sec_fundamentals(
        events_df=transcripts_df,
        config=SECLoaderConfig(refresh=config.refresh),
    )
    transcript_companies = transcripts_df[["ticker", "company_name"]].drop_duplicates()
    reddit_posts_df = load_reddit_posts(
        transcript_companies_df=transcript_companies,
        config=RedditPostsLoadConfig(
            start_date=social_start_date,
            end_date=end_date,
            refresh=config.refresh,
        ),
    )
    reddit_comments_df = load_reddit_comments(
        transcript_companies_df=transcript_companies,
        config=RedditCommentsLoadConfig(
            start_date=social_start_date,
            end_date=end_date,
            refresh=config.refresh,
        ),
    )

    event_level_df = build_event_level_dataset(
        transcripts_df=transcripts_df,
        market_features_df=market_features_df,
        sec_fundamentals_df=sec_fundamentals_df,
        reddit_posts_df=reddit_posts_df,
        reddit_comments_df=reddit_comments_df,
    )
    company_level_df = build_company_level_dataset(
        transcripts_df=transcripts_df,
        price_history_df=price_history_df,
        yfinance_metadata_df=yfinance_metadata_df,
        sec_fundamentals_df=sec_fundamentals_df,
        reddit_posts_df=reddit_posts_df,
        reddit_comments_df=reddit_comments_df,
    )
    save_joined_datasets(company_level_df, event_level_df, config=JoinOutputConfig())
    LOGGER.info(
        "Dataset build complete. Companies=%s Events=%s",
        len(company_level_df),
        len(event_level_df),
    )
    return company_level_df, event_level_df


def _parse_args() -> DatasetBuildConfig:
    """Parse CLI arguments into a build config."""
    parser = argparse.ArgumentParser(description="Build the S&P 500 earnings datasets.")
    parser.add_argument("--start-date", default=None, help="Inclusive event start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", default=None, help="Inclusive event end date in YYYY-MM-DD format.")
    parser.add_argument("--years-history", type=int, default=3, help="Default lookback window when --start-date is omitted.")
    parser.add_argument("--max-tickers", type=int, default=None, help="Optional cap on the ticker universe for smoke tests.")
    parser.add_argument("--refresh", action="store_true", help="Re-download and overwrite cached raw artifacts.")
    args = parser.parse_args()
    return DatasetBuildConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        years_history=args.years_history,
        max_tickers=args.max_tickers,
        refresh=args.refresh,
    )


def main() -> None:
    """CLI entrypoint."""
    build_datasets(_parse_args())


if __name__ == "__main__":
    main()
