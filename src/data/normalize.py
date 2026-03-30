"""Normalization helpers shared across dataset source loaders."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.utils.io_utils import ensure_dir
from src.utils.logging_utils import get_logger
from src.utils.time_utils import normalize_date

LOGGER = get_logger(__name__)

try:  # pragma: no cover - networked dependency
    from datasets import load_dataset
except Exception:  # pragma: no cover
    load_dataset = None


CORPORATE_SUFFIXES = {
    "co",
    "company",
    "corp",
    "corporation",
    "group",
    "holding",
    "holdings",
    "inc",
    "incorporated",
    "ltd",
    "llc",
    "plc",
}


def first_present_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Return the first matching column, using case-insensitive lookup as fallback."""
    lowered = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def series_from_candidates(
    df: pd.DataFrame,
    candidates: Iterable[str],
    default: Any = None,
) -> pd.Series:
    """Return the first available column as a Series, or a default-filled Series."""
    column = first_present_column(df, candidates)
    if column is None:
        return pd.Series([default] * len(df), index=df.index, dtype="object")
    return df[column]


def normalize_ticker(value: Any) -> str | None:
    """Normalize a ticker-like field to uppercase alphanumerics plus dots and dashes."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    ticker = re.sub(r"[^A-Za-z0-9.\-]", "", str(value).strip().upper())
    return ticker or None


def normalize_company_name(value: Any) -> str:
    """Normalize company names for fuzzy mapping across sources."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", str(value).lower())
    tokens = [token for token in cleaned.split() if token and token not in CORPORATE_SUFFIXES]
    return " ".join(tokens).strip()


def resolve_company_to_ticker(company_name: Any, mapping: dict[str, str]) -> str | None:
    """Resolve a company name to a ticker using normalized exact and containment matches."""
    normalized = normalize_company_name(company_name)
    if not normalized:
        return None
    if normalized in mapping:
        return mapping[normalized]
    matches = sorted(
        {
            ticker
            for name_key, ticker in mapping.items()
            if normalized in name_key or name_key in normalized
        }
    )
    return matches[0] if len(matches) == 1 else None


def build_company_name_map(frame: pd.DataFrame) -> dict[str, str]:
    """Build a normalized company-name to ticker map from a transcript anchor table."""
    mapping: dict[str, str] = {}
    if frame.empty or "ticker" not in frame.columns:
        return mapping
    for _, row in frame[["ticker", "company_name"]].dropna().drop_duplicates().iterrows():
        normalized = normalize_company_name(row["company_name"])
        ticker = normalize_ticker(row["ticker"])
        if normalized and ticker:
            mapping[normalized] = ticker
    return mapping


def clean_text(value: Any) -> str:
    """Collapse whitespace and strip empty placeholders."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_fiscal_quarter(value: Any) -> str | None:
    """Normalize a fiscal-quarter-like value into Q1..Q4 when possible."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    match = re.search(r"([1-4])", str(value).upper())
    if not match:
        return None
    return f"Q{match.group(1)}"


def parse_fiscal_year(value: Any) -> int | None:
    """Normalize a fiscal-year-like value into an integer when possible."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    match = re.search(r"(19|20)\d{2}", str(value))
    if not match:
        try:
            numeric = int(float(value))
        except Exception:
            return None
        return numeric if 1900 <= numeric <= 2100 else None
    return int(match.group(0))


def build_event_id(
    ticker: Any,
    event_date: Any,
    fiscal_year: Any = None,
    fiscal_quarter: Any = None,
) -> str:
    """Build a deterministic event identifier."""
    ticker_value = normalize_ticker(ticker) or "UNKNOWN"
    date_value = normalize_date(event_date)
    date_string = "NA" if pd.isna(date_value) else date_value.strftime("%Y-%m-%d")
    year_value = parse_fiscal_year(fiscal_year)
    quarter_value = parse_fiscal_quarter(fiscal_quarter) or "NA"
    year_string = "NA" if year_value is None else str(year_value)
    return f"{ticker_value}_{date_string}_{year_string}_{quarter_value}"


def filter_date_range(
    df: pd.DataFrame,
    date_col: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    """Filter a dataframe by an inclusive normalized date range."""
    if df.empty or date_col not in df.columns:
        return df.copy()
    filtered = df.copy()
    filtered[date_col] = pd.to_datetime(filtered[date_col], errors="coerce").dt.normalize()
    if start_date:
        filtered = filtered[filtered[date_col] >= pd.Timestamp(start_date)]
    if end_date:
        filtered = filtered[filtered[date_col] <= pd.Timestamp(end_date)]
    return filtered.reset_index(drop=True)


def load_hf_dataset_frame(
    dataset_name: str,
    split: str,
    cache_path: str | Path,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load a Hugging Face dataset into pandas and cache it as parquet."""
    cache_file = Path(cache_path)
    if cache_file.exists() and not refresh:
        return pd.read_parquet(cache_file)
    if load_dataset is None:
        raise RuntimeError("The `datasets` package is required to load Hugging Face datasets.")
    token = os.getenv("HF_TOKEN") or None
    dataset = load_dataset(dataset_name, split=split, token=token)
    frame = dataset.to_pandas()
    ensure_dir(cache_file.parent)
    frame.to_parquet(cache_file, index=False)
    LOGGER.info("Cached %s to %s", dataset_name, cache_file)
    return frame
