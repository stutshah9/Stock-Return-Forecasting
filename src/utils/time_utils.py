"""Time and date helpers used across data ingestion and evaluation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


def to_timestamp(value: Any) -> pd.Timestamp | pd.NaT:
    """Coerce a value to pandas Timestamp without raising for bad inputs."""
    if value is None or value == "":
        return pd.NaT
    return pd.to_datetime(value, errors="coerce")


def normalize_date(value: Any) -> pd.Timestamp | pd.NaT:
    """Coerce a value to a normalized date-like Timestamp."""
    ts = to_timestamp(value)
    if pd.isna(ts):
        return pd.NaT
    return ts.normalize()


def today_timestamp() -> pd.Timestamp:
    """Return the current local date as pandas Timestamp."""
    return pd.Timestamp(datetime.now()).normalize()
