"""Download earnings call transcripts from Hugging Face into ``transcripts.csv``."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, login
import pyarrow.parquet as pq
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"
SP500_TICKERS_PATH = DATA_DIR / "sp500_tickers.csv"
TRANSCRIPTS_PATH = DATA_DIR / "transcripts.csv"
DATASET_NAME = "Bose345/sp500_earnings_transcripts"
TARGET_YEARS = {2021, 2022, 2023, 2024, 2025}
EXPECTED_FIELDS = {"symbol", "date", "content"}
TRANSCRIPT_COLUMNS = ["ticker", "date", "year", "quarter", "text"]


def _load_hf_token() -> str:
    """Load ``HF_TOKEN`` from a local ``.env`` file."""

    env_candidates = [
        WORKSPACE_ROOT / ".env",
        PROJECT_ROOT / ".env",
    ]
    for candidate in env_candidates:
        if candidate.is_file():
            load_dotenv(candidate, override=False)

    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is missing from the local .env file.")
    return token


def _canonicalize_ticker(ticker: Any) -> str:
    """Normalize ticker symbols for matching and storage."""

    return str(ticker).strip().upper().replace(".", "-")


def _load_sp500_tickers() -> set[str]:
    """Load the cached S&P 500 ticker universe."""

    tickers_df = pd.read_csv(SP500_TICKERS_PATH)
    return {_canonicalize_ticker(value) for value in tickers_df["ticker"].tolist()}


def _load_existing_transcripts() -> pd.DataFrame:
    """Load existing transcripts into the project schema."""

    if TRANSCRIPTS_PATH.is_file():
        transcripts_df = pd.read_csv(TRANSCRIPTS_PATH)
    else:
        transcripts_df = pd.DataFrame(columns=TRANSCRIPT_COLUMNS)

    for column in TRANSCRIPT_COLUMNS:
        if column not in transcripts_df.columns:
            transcripts_df[column] = pd.NA

    transcripts_df["ticker"] = transcripts_df["ticker"].astype(str).map(_canonicalize_ticker)
    transcripts_df["date"] = transcripts_df["date"].astype(str)
    transcripts_df["year"] = pd.to_numeric(transcripts_df["year"], errors="coerce").astype("Int64")
    transcripts_df["quarter"] = transcripts_df["quarter"].astype(str)
    transcripts_df["text"] = transcripts_df["text"].fillna("").astype(str)
    return transcripts_df[TRANSCRIPT_COLUMNS]


def _parse_date(date_value: Any) -> pd.Timestamp | None:
    """Parse a raw dataset date field into a normalized timestamp."""

    raw_value = str(date_value or "").strip()
    if not raw_value:
        return None

    timestamp = pd.to_datetime(raw_value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp).normalize()


def _quarter_from_month(month: int) -> str:
    """Map calendar month to quarter label."""

    return f"Q{((month - 1) // 3) + 1}"


def _print_schema_preview(first_row: dict[str, Any]) -> None:
    """Print the first row keys and full first row payload."""

    print(f"ds[0].keys(): {list(first_row.keys())}")
    print(f"ds[0]: {first_row}")


def _load_dataset_source(token: str) -> tuple[str, Any, dict[str, Any]]:
    """Load the dataset source and return its mode, handle, and first row."""

    try:
        dataset = load_dataset(DATASET_NAME, split="train")
        return "datasets", dataset, dict(dataset[0])
    except Exception as exc:
        print(f"load_dataset failed: {exc}")

    parquet_path = hf_hub_download(
        repo_id=DATASET_NAME,
        filename="parquet_files/part-0.parquet",
        repo_type="dataset",
        token=token,
    )
    parquet_file = pq.ParquetFile(parquet_path)
    first_row = next(parquet_file.iter_batches(batch_size=1)).to_pylist()[0]
    return "parquet", parquet_file, first_row


def _iter_rows(mode: str, source: Any) -> Iterable[dict[str, Any]]:
    """Iterate dataset rows from either ``datasets`` or parquet sources."""

    if mode == "datasets":
        for row in source:
            yield dict(row)
        return

    for batch in source.iter_batches(batch_size=128, columns=["symbol", "date", "content"]):
        for row in batch.to_pylist():
            yield dict(row)


def main() -> None:
    """Download transcripts, validate schema, and append project-formatted rows."""

    token = _load_hf_token()
    login(token=token)
    mode, source, first_row = _load_dataset_source(token)

    _print_schema_preview(first_row)

    actual_keys = set(first_row.keys())
    if not EXPECTED_FIELDS.issubset(actual_keys):
        print("Field names differ from the assumed mapping.")
        print(f"Actual keys: {sorted(actual_keys)}")
        print("Please confirm which dataset fields map to ticker, date, and transcript.")
        raise SystemExit(1)

    sp500_tickers = _load_sp500_tickers()
    existing_df = _load_existing_transcripts()
    existing_keys = {
        (row["ticker"], row["date"])
        for _, row in existing_df.iterrows()
    }
    new_rows: list[dict[str, Any]] = []

    for row in tqdm(_iter_rows(mode, source), desc="Processing transcripts"):
        ticker = _canonicalize_ticker(row["symbol"])
        if ticker not in sp500_tickers:
            continue

        event_date = _parse_date(row["date"])
        if event_date is None or int(event_date.year) not in TARGET_YEARS:
            continue

        transcript_text = str(row["content"] or "").strip()
        if len(transcript_text) < 100:
            continue

        new_rows.append(
            {
                "ticker": ticker,
                "date": event_date.strftime("%Y-%m-%d"),
                "year": int(event_date.year),
                "quarter": _quarter_from_month(int(event_date.month)),
                "text": transcript_text,
            }
        )

    new_df = pd.DataFrame(new_rows, columns=TRANSCRIPT_COLUMNS)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["ticker", "date"], keep="last")
    combined_df = combined_df.sort_values(by=["ticker", "date"])
    combined_df.to_csv(TRANSCRIPTS_PATH, index=False)

    combined_key_df = combined_df[["ticker", "date", "year", "text"]].copy()
    added_df = combined_key_df[
        ~combined_key_df.apply(
            lambda row: (row["ticker"], row["date"]) in existing_keys,
            axis=1,
        )
    ].drop_duplicates(subset=["ticker", "date"], keep="last")

    if added_df.empty:
        print("rows added:       0")
        print("tickers covered:  0")
        print("years covered:    []")
        print("avg text length:  0 chars")
        return

    tickers_covered = int(added_df["ticker"].nunique())
    years_covered = sorted(
        pd.to_numeric(added_df["year"], errors="coerce").dropna().astype(int).unique().tolist()
    )
    avg_text_length = int(round(added_df["text"].str.len().mean()))

    print(f"rows added:       {len(added_df)}")
    print(f"tickers covered:  {tickers_covered}")
    print(f"years covered:    {years_covered}")
    print(f"avg text length:  {avg_text_length} chars")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
