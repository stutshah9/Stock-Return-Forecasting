"""Load and normalize Reddit posts for the transcript-backed company universe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.normalize import (
    build_company_name_map,
    clean_text,
    filter_date_range,
    load_hf_dataset_frame,
    normalize_ticker,
    resolve_company_to_ticker,
    series_from_candidates,
)
from src.utils.io_utils import save_dataframe
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

DEFAULT_POSTS_DATASET = "emilpartow/reddit_finance_posts_sp500"
DEFAULT_POSTS_CACHE = Path("data/raw/hf_reddit_finance_posts_sp500.parquet")
DEFAULT_POSTS_OUTPUT = Path("data/processed/reddit_posts.parquet")


@dataclass(slots=True)
class RedditPostsLoadConfig:
    """Configuration for Reddit post loading."""

    dataset_name: str = DEFAULT_POSTS_DATASET
    split: str = "train"
    raw_cache_path: Path = DEFAULT_POSTS_CACHE
    output_path: Path = DEFAULT_POSTS_OUTPUT
    start_date: str | None = None
    end_date: str | None = None
    refresh: bool = False


def normalize_reddit_posts_frame(raw_df: pd.DataFrame, transcript_companies_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Reddit post rows into a stable schema keyed by ticker."""
    if raw_df.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "company_name",
                "post_id",
                "subreddit",
                "title",
                "body",
                "text",
                "created_timestamp",
                "score",
                "engagement",
            ]
        )
    company_map = build_company_name_map(transcript_companies_df)
    normalized = pd.DataFrame(index=raw_df.index)
    normalized["ticker"] = series_from_candidates(raw_df, ["ticker", "symbol", "stock_symbol"]).map(normalize_ticker)
    if normalized["ticker"].isna().any():
        company_series = series_from_candidates(raw_df, ["company_name", "company", "issuer_name"])
        normalized.loc[normalized["ticker"].isna(), "ticker"] = company_series[normalized["ticker"].isna()].map(
            lambda name: resolve_company_to_ticker(name, company_map)
        )
    normalized["company_name"] = series_from_candidates(raw_df, ["company_name", "company", "issuer_name"]).astype("string")
    normalized["post_id"] = series_from_candidates(raw_df, ["post_id", "id", "name"]).astype("string")
    normalized["subreddit"] = series_from_candidates(raw_df, ["subreddit"]).astype("string")
    normalized["title"] = series_from_candidates(raw_df, ["title", "post_title"]).map(clean_text)
    normalized["body"] = series_from_candidates(raw_df, ["body", "selftext", "text", "post_text"]).map(clean_text)
    normalized["text"] = (normalized["title"].fillna("") + "\n\n" + normalized["body"].fillna("")).map(clean_text)
    datetime_series = series_from_candidates(raw_df, ["created_datetime", "created_at", "timestamp", "created_timestamp"])
    normalized["created_timestamp"] = pd.to_datetime(datetime_series, errors="coerce")
    if normalized["created_timestamp"].isna().all():
        normalized["created_timestamp"] = pd.to_datetime(
            series_from_candidates(raw_df, ["created_utc", "created"]),
            unit="s",
            errors="coerce",
        )
    normalized["score"] = pd.to_numeric(series_from_candidates(raw_df, ["score", "ups"]), errors="coerce")
    num_comments = pd.to_numeric(series_from_candidates(raw_df, ["num_comments"], default=0), errors="coerce").fillna(0)
    normalized["engagement"] = normalized["score"].fillna(0) + num_comments
    allowed_tickers = set(transcript_companies_df["ticker"].dropna().unique())
    normalized = normalized[normalized["ticker"].isin(allowed_tickers)]
    normalized = normalized.dropna(subset=["ticker", "created_timestamp"]).drop_duplicates(subset=["post_id"], keep="first")
    return normalized.sort_values(["ticker", "created_timestamp"]).reset_index(drop=True)


def load_reddit_posts(
    transcript_companies_df: pd.DataFrame,
    config: RedditPostsLoadConfig,
) -> pd.DataFrame:
    """Load, normalize, filter, and save Reddit posts."""
    raw_df = load_hf_dataset_frame(
        dataset_name=config.dataset_name,
        split=config.split,
        cache_path=config.raw_cache_path,
        refresh=config.refresh,
    )
    posts_df = normalize_reddit_posts_frame(raw_df, transcript_companies_df=transcript_companies_df)
    posts_df = filter_date_range(posts_df, "created_timestamp", config.start_date, config.end_date)
    save_dataframe(posts_df, config.output_path)
    LOGGER.info("Normalized %s Reddit posts to %s", len(posts_df), config.output_path)
    return posts_df
