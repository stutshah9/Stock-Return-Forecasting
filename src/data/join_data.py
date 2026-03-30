"""Join transcript, market, SEC, and Reddit data into final datasets."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils.io_utils import save_dataframe
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class JoinOutputConfig:
    """Output locations for final joined datasets."""

    company_level_output_path: str = "data/processed/company_level_dataset.parquet"
    event_level_output_path: str = "data/processed/event_level_dataset.parquet"


def _concat_text(values: pd.Series) -> str:
    """Concatenate non-empty text values with stable ordering."""
    texts = [str(value).strip() for value in values.fillna("") if str(value).strip()]
    return "\n\n".join(texts)


def aggregate_reddit_windows(
    events_df: pd.DataFrame,
    reddit_posts_df: pd.DataFrame,
    reddit_comments_df: pd.DataFrame,
    windows: tuple[int, ...] = (1, 3, 7),
) -> pd.DataFrame:
    """Aggregate Reddit features over pre-event windows ending at the event date."""
    posts = reddit_posts_df.copy()
    comments = reddit_comments_df.copy()
    if not posts.empty:
        posts["created_timestamp"] = pd.to_datetime(posts["created_timestamp"], errors="coerce")
    if not comments.empty:
        comments["created_timestamp"] = pd.to_datetime(comments["created_timestamp"], errors="coerce")

    records: list[dict[str, object]] = []
    for _, event in events_df.iterrows():
        event_date = pd.Timestamp(event["event_date"]).normalize()
        record: dict[str, object] = {
            "event_id": event["event_id"],
            "ticker": event["ticker"],
            "event_date": event_date,
        }
        for window in windows:
            window_start = event_date - pd.Timedelta(days=window)
            post_window = posts[
                (posts["ticker"] == event["ticker"])
                & (posts["created_timestamp"] >= window_start)
                & (posts["created_timestamp"] < event_date)
            ].copy()
            comment_window = comments[
                (comments["ticker"] == event["ticker"])
                & (comments["created_timestamp"] >= window_start)
                & (comments["created_timestamp"] < event_date)
            ].copy()
            record[f"reddit_post_count_{window}d"] = int(len(post_window))
            record[f"reddit_comment_count_{window}d"] = int(len(comment_window))
            record[f"reddit_post_score_mean_{window}d"] = (
                float(post_window["score"].mean()) if not post_window.empty and "score" in post_window.columns else None
            )
            record[f"reddit_comment_score_mean_{window}d"] = (
                float(comment_window["score"].mean()) if not comment_window.empty and "score" in comment_window.columns else None
            )
            text_parts = []
            if not post_window.empty:
                text_parts.append(_concat_text(post_window["text"]))
            if not comment_window.empty:
                text_parts.append(_concat_text(comment_window["body"]))
            record[f"reddit_text_{window}d"] = "\n\n".join(part for part in text_parts if part).strip()
        records.append(record)
    return pd.DataFrame(records).sort_values(["event_date", "ticker"]).reset_index(drop=True)


def build_event_level_dataset(
    transcripts_df: pd.DataFrame,
    market_features_df: pd.DataFrame,
    sec_fundamentals_df: pd.DataFrame,
    reddit_posts_df: pd.DataFrame,
    reddit_comments_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the main event-level dataset with transcript events as the anchor table."""
    reddit_aggregates = aggregate_reddit_windows(
        events_df=transcripts_df,
        reddit_posts_df=reddit_posts_df,
        reddit_comments_df=reddit_comments_df,
    )
    event_level_df = transcripts_df.copy()
    for frame in [market_features_df, sec_fundamentals_df, reddit_aggregates]:
        event_level_df = event_level_df.merge(frame, on=["event_id", "ticker", "event_date"], how="left")
    event_level_df = event_level_df.sort_values(["event_date", "ticker"]).reset_index(drop=True)
    return event_level_df


def build_company_level_dataset(
    transcripts_df: pd.DataFrame,
    price_history_df: pd.DataFrame,
    yfinance_metadata_df: pd.DataFrame,
    sec_fundamentals_df: pd.DataFrame,
    reddit_posts_df: pd.DataFrame,
    reddit_comments_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a broadly useful company-level summary dataset."""
    transcript_summary = (
        transcripts_df.groupby("ticker", dropna=False)
        .agg(
            company_name=("company_name", "first"),
            first_event_date=("event_date", "min"),
            last_event_date=("event_date", "max"),
            transcript_event_count=("event_id", "nunique"),
            avg_transcript_word_count=("transcript_word_count", "mean"),
        )
        .reset_index()
    )
    if price_history_df.empty:
        price_summary = pd.DataFrame(columns=["ticker", "price_history_start", "price_history_end", "price_row_count"])
    else:
        price_summary = (
            price_history_df.groupby("ticker", dropna=False)
            .agg(
                price_history_start=("date", "min"),
                price_history_end=("date", "max"),
                price_row_count=("date", "count"),
            )
            .reset_index()
        )
    if sec_fundamentals_df.empty:
        latest_sec = pd.DataFrame(columns=["ticker"])
    else:
        latest_sec = (
            sec_fundamentals_df.sort_values(["ticker", "event_date"])
            .groupby("ticker", dropna=False)
            .tail(1)
            .drop(columns=["event_id", "event_date", "fiscal_year", "fiscal_quarter"], errors="ignore")
        )
    post_summary = (
        reddit_posts_df.groupby("ticker", dropna=False)
        .agg(
            reddit_posts_total=("post_id", "count"),
            first_reddit_post_timestamp=("created_timestamp", "min"),
            last_reddit_post_timestamp=("created_timestamp", "max"),
        )
        .reset_index()
        if not reddit_posts_df.empty
        else pd.DataFrame(columns=["ticker", "reddit_posts_total", "first_reddit_post_timestamp", "last_reddit_post_timestamp"])
    )
    comment_summary = (
        reddit_comments_df.groupby("ticker", dropna=False)
        .agg(
            reddit_comments_total=("comment_id", "count"),
            first_reddit_comment_timestamp=("created_timestamp", "min"),
            last_reddit_comment_timestamp=("created_timestamp", "max"),
        )
        .reset_index()
        if not reddit_comments_df.empty
        else pd.DataFrame(columns=["ticker", "reddit_comments_total", "first_reddit_comment_timestamp", "last_reddit_comment_timestamp"])
    )
    company_level_df = transcript_summary.copy()
    for frame in [yfinance_metadata_df, price_summary, latest_sec, post_summary, comment_summary]:
        company_level_df = company_level_df.merge(frame, on="ticker", how="left")
    company_level_df = company_level_df.sort_values("ticker").reset_index(drop=True)
    return company_level_df


def save_joined_datasets(
    company_level_df: pd.DataFrame,
    event_level_df: pd.DataFrame,
    config: JoinOutputConfig,
) -> None:
    """Save the final joined parquet datasets."""
    save_dataframe(company_level_df, config.company_level_output_path)
    save_dataframe(event_level_df, config.event_level_output_path)
    LOGGER.info("Saved company-level dataset to %s", config.company_level_output_path)
    LOGGER.info("Saved event-level dataset to %s", config.event_level_output_path)
