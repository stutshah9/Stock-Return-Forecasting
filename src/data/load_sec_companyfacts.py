"""Load SEC companyfacts and align normalized fundamentals to transcript events."""

from __future__ import annotations

import json
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

from src.data.normalize import parse_fiscal_quarter, parse_fiscal_year
from src.utils.io_utils import ensure_dir, save_dataframe
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

FLOW_CONCEPTS: dict[str, list[str]] = {
    "sec_revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "sec_net_income": ["NetIncomeLoss", "ProfitLoss"],
    "sec_eps": [
        "EarningsPerShareDiluted",
        "EarningsPerShareBasicAndDiluted",
        "EarningsPerShareBasic",
    ],
}

STOCK_CONCEPTS: dict[str, list[str]] = {
    "sec_assets": ["Assets"],
    "sec_liabilities": ["Liabilities"],
    "sec_cash_and_cash_equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "sec_shares_outstanding": [
        "EntityCommonStockSharesOutstanding",
        "CommonStockSharesOutstanding",
    ],
}


@dataclass(slots=True)
class SECLoaderConfig:
    """Configuration for SEC companyfacts ingestion."""

    output_path: Path = Path("data/processed/sec_fundamentals.parquet")
    companyfacts_cache_dir: Path = Path("data/raw/sec/companyfacts")
    ticker_map_cache_path: Path = Path("data/raw/sec/company_tickers.json")
    bulk_companyfacts_zip_path: Path = Path("data/external/sec/companyfacts.zip")
    refresh: bool = False
    pause_seconds: float = 0.2


def _normalize_observations(observations: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert raw companyfacts observations into a typed dataframe."""
    if not observations:
        return pd.DataFrame()
    frame = pd.DataFrame(observations)
    if frame.empty:
        return frame
    empty_series = pd.Series([None] * len(frame), index=frame.index)
    frame["val"] = pd.to_numeric(frame["val"] if "val" in frame.columns else empty_series, errors="coerce")
    frame["end"] = pd.to_datetime(frame["end"] if "end" in frame.columns else empty_series, errors="coerce")
    frame["filed"] = pd.to_datetime(frame["filed"] if "filed" in frame.columns else empty_series, errors="coerce")
    frame["fy"] = (frame["fy"] if "fy" in frame.columns else empty_series).map(parse_fiscal_year)
    frame["fp"] = (frame["fp"] if "fp" in frame.columns else empty_series).map(parse_fiscal_quarter)
    frame["form"] = (frame["form"] if "form" in frame.columns else empty_series).astype("string")
    frame["frame"] = (frame["frame"] if "frame" in frame.columns else empty_series).astype("string")
    frame["is_quarter_like"] = (
        frame["fp"].astype("string").isin(["Q1", "Q2", "Q3", "Q4"])
        | frame["frame"].astype("string").str.contains("Q", case=False, na=False)
        | frame["form"].astype("string").str.startswith("10-Q", na=False)
        | frame["form"].astype("string").str.startswith("10-K", na=False)
    )
    return frame.dropna(subset=["val"]).reset_index(drop=True)


def _match_unit(unit_name: str, preferred_prefixes: Iterable[str]) -> bool:
    """Return whether a SEC unit key matches a preferred prefix."""
    lowered = str(unit_name).lower()
    return any(lowered == prefix.lower() or lowered.startswith(prefix.lower()) for prefix in preferred_prefixes)


def _extract_concept_observations(
    companyfacts_json: dict[str, Any],
    concept_aliases: list[str],
    preferred_units: Iterable[str],
) -> pd.DataFrame:
    """Extract typed observations for a list of GAAP concept aliases."""
    facts = ((companyfacts_json.get("facts") or {}).get("us-gaap") or {})
    observations: list[dict[str, Any]] = []
    for alias in concept_aliases:
        payload = facts.get(alias) or {}
        units = payload.get("units") or {}
        unit_keys = [key for key in units if _match_unit(key, preferred_units)] or list(units.keys())
        for unit_key in unit_keys:
            for row in units.get(unit_key, []):
                normalized = dict(row)
                normalized["concept"] = alias
                normalized["unit"] = unit_key
                observations.append(normalized)
        if observations:
            break
    return _normalize_observations(observations)


def _select_observation_for_event(
    observations: pd.DataFrame,
    event_date: pd.Timestamp,
    fiscal_year: int | None,
    fiscal_quarter: str | None,
    prefer_quarter_like: bool,
) -> tuple[pd.Series | None, str]:
    """Select the best observation for an event using fiscal and date alignment."""
    if observations.empty:
        return None, "missing"
    eligible = observations.copy()
    eligible = eligible[
        (eligible["filed"].notna() & (eligible["filed"] <= event_date))
        | (eligible["end"].notna() & (eligible["end"] <= event_date))
    ]
    if eligible.empty:
        return None, "missing"
    if fiscal_year is not None and fiscal_quarter is not None:
        matched = eligible[(eligible["fy"] == fiscal_year) & (eligible["fp"] == fiscal_quarter)]
        if not matched.empty:
            matched = matched.sort_values(["filed", "end"], ascending=[False, False])
            return matched.iloc[0], "exact_fiscal_match"
    if prefer_quarter_like:
        quarter_like = eligible[eligible["is_quarter_like"]]
        if not quarter_like.empty:
            eligible = quarter_like
    eligible = eligible.sort_values(["filed", "end"], ascending=[False, False])
    return eligible.iloc[0], "latest_available_before_event"


class SECCompanyfactsClient:
    """Thin SEC client with local caching and optional bulk ZIP support."""

    def __init__(self, config: SECLoaderConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self._resolve_user_agent(), "Accept-Encoding": "gzip, deflate"})
        self._ticker_map: dict[str, int] | None = None
        self._zip_index: dict[str, str] | None = None

    def _resolve_user_agent(self) -> str:
        user_agent = os.getenv("SEC_USER_AGENT")
        if not user_agent:
            raise ValueError(
                "SEC_USER_AGENT is required. Set it in your environment or .env file before running the dataset builder."
            )
        return user_agent

    def load_ticker_map(self) -> dict[str, int]:
        """Load the SEC ticker-to-CIK mapping from cache or the live endpoint."""
        if self._ticker_map is not None and not self.config.refresh:
            return self._ticker_map
        cache_path = self.config.ticker_map_cache_path
        if cache_path.exists() and not self.config.refresh:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            response = self.session.get(SEC_TICKER_MAP_URL, timeout=30)
            response.raise_for_status()
            payload = response.json()
            ensure_dir(cache_path.parent)
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
            time.sleep(self.config.pause_seconds)

        if isinstance(payload, dict):
            records = payload.values()
        else:
            records = payload
        mapping: dict[str, int] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            ticker = str(record.get("ticker") or "").upper().strip()
            cik_value = record.get("cik_str") or record.get("cik") or record.get("cikStr")
            if not ticker or cik_value in {None, ""}:
                continue
            mapping[ticker] = int(cik_value)
        self._ticker_map = mapping
        return mapping

    def _cache_file(self, cik: int) -> Path:
        return ensure_dir(self.config.companyfacts_cache_dir) / f"CIK{cik:010d}.json"

    def _load_from_bulk_zip(self, cik: int) -> dict[str, Any] | None:
        zip_path = self.config.bulk_companyfacts_zip_path
        if not zip_path.exists():
            return None
        if self._zip_index is None:
            with zipfile.ZipFile(zip_path) as zip_file:
                self._zip_index = {
                    Path(member).name: member
                    for member in zip_file.namelist()
                    if member.lower().endswith(".json")
                }
        member_name = self._zip_index.get(f"CIK{cik:010d}.json")
        if member_name is None:
            return None
        with zipfile.ZipFile(zip_path) as zip_file:
            return json.loads(zip_file.read(member_name).decode("utf-8"))

    def load_companyfacts(self, cik: int) -> dict[str, Any] | None:
        """Load one companyfacts JSON payload from cache, bulk ZIP, or SEC API."""
        cache_path = self._cache_file(cik)
        if cache_path.exists() and not self.config.refresh:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        payload = self._load_from_bulk_zip(cik)
        if payload is None:
            url = SEC_COMPANYFACTS_URL.format(cik=f"{cik:010d}")
            response = self.session.get(url, timeout=30)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            payload = response.json()
            time.sleep(self.config.pause_seconds)
        ensure_dir(cache_path.parent)
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload


def align_sec_companyfacts_to_events(
    events_df: pd.DataFrame,
    client: SECCompanyfactsClient,
) -> pd.DataFrame:
    """Align SEC companyfacts values to transcript-backed earnings events."""
    if events_df.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "ticker",
                "cik",
                "event_date",
                "fiscal_year",
                "fiscal_quarter",
                "sec_fiscal_year",
                "sec_fiscal_quarter",
                "sec_filed_date",
                "sec_match_type",
                "sec_revenue",
                "sec_net_income",
                "sec_assets",
                "sec_liabilities",
                "sec_cash_and_cash_equivalents",
                "sec_shares_outstanding",
                "sec_eps",
                "fundamentals_available",
            ]
        )

    ticker_map = client.load_ticker_map()
    companyfacts_cache: dict[str, dict[str, Any] | None] = {}
    records: list[dict[str, Any]] = []
    for _, event in events_df.iterrows():
        ticker = str(event["ticker"]).upper()
        cik = ticker_map.get(ticker)
        record: dict[str, Any] = {
            "event_id": event["event_id"],
            "ticker": ticker,
            "cik": cik,
            "event_date": pd.Timestamp(event["event_date"]).normalize(),
            "fiscal_year": parse_fiscal_year(event.get("fiscal_year")),
            "fiscal_quarter": parse_fiscal_quarter(event.get("fiscal_quarter")),
            "sec_fiscal_year": None,
            "sec_fiscal_quarter": None,
            "sec_filed_date": pd.NaT,
            "sec_match_type": "missing",
            "sec_revenue": None,
            "sec_net_income": None,
            "sec_assets": None,
            "sec_liabilities": None,
            "sec_cash_and_cash_equivalents": None,
            "sec_shares_outstanding": None,
            "sec_eps": None,
            "fundamentals_available": False,
        }
        if cik is None:
            records.append(record)
            continue

        cache_key = str(cik)
        if cache_key not in companyfacts_cache:
            try:
                companyfacts_cache[cache_key] = client.load_companyfacts(cik)
            except Exception as exc:
                LOGGER.warning("SEC companyfacts fetch failed for %s (%s): %s", ticker, cik, exc)
                companyfacts_cache[cache_key] = None
        payload = companyfacts_cache[cache_key]
        if not payload:
            records.append(record)
            continue

        latest_filed_date = pd.NaT
        match_types: list[str] = []
        for column_name, aliases in FLOW_CONCEPTS.items():
            preferred_units = ["USD/shares", "USD", "usd/shares"] if column_name == "sec_eps" else ["USD"]
            observations = _extract_concept_observations(payload, aliases, preferred_units=preferred_units)
            selected, match_type = _select_observation_for_event(
                observations,
                event_date=record["event_date"],
                fiscal_year=record["fiscal_year"],
                fiscal_quarter=record["fiscal_quarter"],
                prefer_quarter_like=True,
            )
            match_types.append(match_type)
            if selected is not None:
                record[column_name] = float(selected["val"])
                record["sec_fiscal_year"] = selected.get("fy") or record["sec_fiscal_year"]
                record["sec_fiscal_quarter"] = selected.get("fp") or record["sec_fiscal_quarter"]
                filed = selected.get("filed")
                if pd.notna(filed):
                    latest_filed_date = max(pd.Timestamp(filed), latest_filed_date) if pd.notna(latest_filed_date) else pd.Timestamp(filed)

        for column_name, aliases in STOCK_CONCEPTS.items():
            preferred_units = ["shares"] if "shares" in column_name else ["USD"]
            observations = _extract_concept_observations(payload, aliases, preferred_units=preferred_units)
            selected, match_type = _select_observation_for_event(
                observations,
                event_date=record["event_date"],
                fiscal_year=record["fiscal_year"],
                fiscal_quarter=record["fiscal_quarter"],
                prefer_quarter_like=False,
            )
            match_types.append(match_type)
            if selected is not None:
                record[column_name] = float(selected["val"])
                record["sec_fiscal_year"] = selected.get("fy") or record["sec_fiscal_year"]
                record["sec_fiscal_quarter"] = selected.get("fp") or record["sec_fiscal_quarter"]
                filed = selected.get("filed")
                if pd.notna(filed):
                    latest_filed_date = max(pd.Timestamp(filed), latest_filed_date) if pd.notna(latest_filed_date) else pd.Timestamp(filed)

        record["sec_filed_date"] = latest_filed_date
        record["sec_match_type"] = "exact_fiscal_match" if "exact_fiscal_match" in match_types else (
            "latest_available_before_event" if "latest_available_before_event" in match_types else "missing"
        )
        core_columns = [
            "sec_revenue",
            "sec_net_income",
            "sec_assets",
            "sec_liabilities",
            "sec_cash_and_cash_equivalents",
            "sec_shares_outstanding",
            "sec_eps",
        ]
        record["fundamentals_available"] = any(
            value is not None and not pd.isna(value)
            for value in (record.get(column_name) for column_name in core_columns)
        )
        records.append(record)

    frame = pd.DataFrame(records).sort_values(["event_date", "ticker"]).reset_index(drop=True)
    return frame


def load_sec_fundamentals(
    events_df: pd.DataFrame,
    config: SECLoaderConfig,
) -> pd.DataFrame:
    """Load and align SEC companyfacts, then save a normalized parquet table."""
    client = SECCompanyfactsClient(config)
    fundamentals_df = align_sec_companyfacts_to_events(events_df, client)
    save_dataframe(fundamentals_df, config.output_path)
    LOGGER.info("Aligned SEC fundamentals for %s transcript events", len(fundamentals_df))
    return fundamentals_df
