"""I/O utilities for configuration, data frames, and lightweight serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file {path} did not contain a mapping.")
    return data


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Write JSON with stable formatting."""
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, default=_json_default, sort_keys=True)


def _json_default(value: Any) -> Any:
    """JSON fallback for pandas and numpy values."""
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Persist a dataframe to CSV or parquet based on the file suffix."""
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    suffix = path_obj.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path_obj, index=False)
    elif suffix == ".parquet":
        df.to_parquet(path_obj, index=False)
    else:
        raise ValueError(f"Unsupported dataframe output format: {path_obj}")


def read_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a dataframe from CSV or parquet."""
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path_obj)
    if suffix == ".parquet":
        return pd.read_parquet(path_obj)
    raise ValueError(f"Unsupported dataframe input format: {path_obj}")
