"""Load and normalize the Bose345 earnings transcript dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.normalize import (
    build_event_id,
    clean_text,
    filter_date_range,
    load_hf_dataset_frame,
    parse_fiscal_quarter,
    parse_fiscal_year,
    series_from_candidates,
    normalize_ticker,
)
from src.utils.io_utils import save_dataframe
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

DEFAULT_TRANSCRIPT_DATASET = "Bose345/sp500_earnings_transcripts"
DEFAULT_TRANSCRIPT_CACHE = Path("data/raw/hf_bose345_sp500_earnings_transcripts.parquet")
DEFAULT_TRANSCRIPT_OUTPUT = Path("data/processed/transcripts.parquet")


@dataclass(slots=True)
class TranscriptLoadConfig:
    """Configuration for transcript loading."""

    dataset_name: str = DEFAULT_TRANSCRIPT_DATASET
    split: str = "train"
    raw_cache_path: Path = DEFAULT_TRANSCRIPT_CACHE
    output_path: Path = DEFAULT_TRANSCRIPT_OUTPUT
    start_date: str | None = None
    end_date: str | None = None
    refresh: bool = False


def normalize_transcript_frame(raw_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Normalize a transcript dataframe into a stable event-level schema."""
    if raw_df.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "ticker",
                "company_name",
                "event_date",
                "event_datetime",
                "fiscal_year",
                "fiscal_quarter",
                "transcript_text",
                "transcript_word_count",
                "transcript_char_count",
                "transcript_source_dataset",
                "source_row_id",
            ]
        )

    normalized = pd.DataFrame(index=raw_df.index)
    normalized["ticker"] = series_from_candidates(
        raw_df,
        ["ticker", "symbol", "stock_symbol", "stock", "company_symbol"],
    ).map(normalize_ticker)
    normalized["company_name"] = series_from_candidates(
        raw_df,
        ["company_name", "company", "issuer_name", "name"],
    ).astype("string")
    normalized["event_date"] = pd.to_datetime(
        series_from_candidates(raw_df, ["earnings_date", "call_date", "date", "event_date"]),
        errors="coerce",
    ).dt.normalize()
    normalized["event_datetime"] = pd.to_datetime(
        series_from_candidates(raw_df, ["earnings_datetime", "call_datetime", "datetime", "timestamp"]),
        errors="coerce",
    )
    event_datetime_missing = normalized["event_datetime"].isna()
    normalized.loc[event_datetime_missing, "event_datetime"] = normalized.loc[event_datetime_missing, "event_date"]

    fiscal_year_series = series_from_candidates(raw_df, ["fiscal_year", "year", "fy", "calendar_year"])
    fiscal_quarter_series = series_from_candidates(raw_df, ["fiscal_quarter", "fiscal_period", "quarter", "fq"])
    combined_period_series = series_from_candidates(raw_df, ["fiscal_period", "datafqtr", "datacqtr"], default=None)

    normalized["fiscal_year"] = fiscal_year_series.map(parse_fiscal_year).astype("Int64")
    normalized["fiscal_quarter"] = fiscal_quarter_series.map(parse_fiscal_quarter)
    combined_years = combined_period_series.map(parse_fiscal_year)
    combined_quarters = combined_period_series.map(parse_fiscal_quarter)
    normalized["fiscal_year"] = normalized["fiscal_year"].fillna(combined_years).astype("Int64")
    normalized["fiscal_quarter"] = normalized["fiscal_quarter"].where(
        normalized["fiscal_quarter"].notna(),
        combined_quarters,
    )

    normalized["transcript_text"] = series_from_candidates(
        raw_df,
        ["transcript", "content", "raw_text", "text", "body"],
    ).map(clean_text)
    normalized["source_row_id"] = series_from_candidates(
        raw_df,
        ["source_event_id", "event_id", "id", "call_id", "transcript_id"],
    ).astype("string")
    normalized["transcript_source_dataset"] = dataset_name
    normalized["transcript_word_count"] = normalized["transcript_text"].map(lambda text: len(text.split()) if text else 0)
    normalized["transcript_char_count"] = normalized["transcript_text"].map(len)
    normalized["event_id"] = normalized.apply(
        lambda row: build_event_id(
            row["ticker"],
            row["event_date"],
            row["fiscal_year"],
            row["fiscal_quarter"],
        ),
        axis=1,
    )

    normalized["company_name"] = normalized["company_name"].fillna(normalized["ticker"])
    normalized = normalized.dropna(subset=["ticker", "event_date"])
    normalized = normalized[normalized["transcript_text"].str.len() > 0]
    normalized = normalized.drop_duplicates(subset=["event_id"], keep="first")
    normalized = normalized.sort_values(["event_date", "ticker"]).reset_index(drop=True)
    return normalized


def load_transcripts(config: TranscriptLoadConfig) -> pd.DataFrame:
    """Load, normalize, filter, and save transcript events."""
    raw_df = load_hf_dataset_frame(
        dataset_name=config.dataset_name,
        split=config.split,
        cache_path=config.raw_cache_path,
        refresh=config.refresh,
    )
    transcripts = normalize_transcript_frame(raw_df, dataset_name=config.dataset_name)
    transcripts = filter_date_range(transcripts, "event_date", config.start_date, config.end_date)
    save_dataframe(transcripts, config.output_path)
    LOGGER.info("Normalized %s transcript events to %s", len(transcripts), config.output_path)
    return transcripts
