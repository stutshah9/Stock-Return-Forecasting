"""Load and normalize Reddit comments for the transcript-backed company universe."""

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

DEFAULT_COMMENTS_DATASET = "emilpartow/reddit_comments_sp500"
DEFAULT_COMMENTS_CACHE = Path("data/raw/hf_reddit_comments_sp500.parquet")
DEFAULT_COMMENTS_OUTPUT = Path("data/processed/reddit_comments.parquet")


@dataclass(slots=True)
class RedditCommentsLoadConfig:
    """Configuration for Reddit comment loading."""

    dataset_name: str = DEFAULT_COMMENTS_DATASET
    split: str = "train"
    raw_cache_path: Path = DEFAULT_COMMENTS_CACHE
    output_path: Path = DEFAULT_COMMENTS_OUTPUT
    start_date: str | None = None
    end_date: str | None = None
    refresh: bool = False


def normalize_reddit_comments_frame(raw_df: pd.DataFrame, transcript_companies_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Reddit comment rows into a stable schema keyed by ticker."""
    if raw_df.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "company_name",
                "comment_id",
                "parent_post_id",
                "body",
                "created_timestamp",
                "score",
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
    normalized["comment_id"] = series_from_candidates(raw_df, ["comment_id", "id", "name"]).astype("string")
    normalized["parent_post_id"] = series_from_candidates(raw_df, ["post_id", "parent_post_id", "link_id"]).astype("string")
    normalized["body"] = series_from_candidates(raw_df, ["comment", "body", "text"]).map(clean_text)
    normalized["created_timestamp"] = pd.to_datetime(
        series_from_candidates(raw_df, ["comment_created_utc", "created_utc"]),
        unit="s",
        errors="coerce",
    )
    if normalized["created_timestamp"].isna().all():
        normalized["created_timestamp"] = pd.to_datetime(
            series_from_candidates(raw_df, ["created_datetime", "created_at", "timestamp"]),
            errors="coerce",
        )
    normalized["score"] = pd.to_numeric(series_from_candidates(raw_df, ["comment_score", "score"]), errors="coerce")
    allowed_tickers = set(transcript_companies_df["ticker"].dropna().unique())
    normalized = normalized[normalized["ticker"].isin(allowed_tickers)]
    normalized = normalized.dropna(subset=["ticker", "created_timestamp"]).drop_duplicates(subset=["comment_id"], keep="first")
    return normalized.sort_values(["ticker", "created_timestamp"]).reset_index(drop=True)


def load_reddit_comments(
    transcript_companies_df: pd.DataFrame,
    config: RedditCommentsLoadConfig,
) -> pd.DataFrame:
    """Load, normalize, filter, and save Reddit comments."""
    raw_df = load_hf_dataset_frame(
        dataset_name=config.dataset_name,
        split=config.split,
        cache_path=config.raw_cache_path,
        refresh=config.refresh,
    )
    comments_df = normalize_reddit_comments_frame(raw_df, transcript_companies_df=transcript_companies_df)
    comments_df = filter_date_range(comments_df, "created_timestamp", config.start_date, config.end_date)
    save_dataframe(comments_df, config.output_path)
    LOGGER.info("Normalized %s Reddit comments to %s", len(comments_df), config.output_path)
    return comments_df
