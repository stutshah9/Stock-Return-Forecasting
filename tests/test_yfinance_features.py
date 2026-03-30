from __future__ import annotations

import math
from types import SimpleNamespace

import pandas as pd

from src.data import load_yfinance
from src.data.load_yfinance import build_market_features, fetch_company_metadata, fetch_price_history


def test_build_market_features_uses_previous_close_for_pre_event_inputs() -> None:
    events_df = pd.DataFrame(
        [
            {
                "event_id": "AAA_2024-01-10_2024_Q1",
                "ticker": "AAA",
                "event_date": "2024-01-10",
            }
        ]
    )
    price_history_df = pd.DataFrame(
        [
            {"ticker": "AAA", "date": "2024-01-03", "open": 9.8, "high": 10.1, "low": 9.7, "close": 10.0, "volume": 100},
            {"ticker": "AAA", "date": "2024-01-04", "open": 10.0, "high": 10.3, "low": 9.9, "close": 10.2, "volume": 110},
            {"ticker": "AAA", "date": "2024-01-05", "open": 10.2, "high": 10.5, "low": 10.1, "close": 10.4, "volume": 120},
            {"ticker": "AAA", "date": "2024-01-08", "open": 10.4, "high": 10.6, "low": 10.3, "close": 10.5, "volume": 130},
            {"ticker": "AAA", "date": "2024-01-09", "open": 10.5, "high": 10.7, "low": 10.4, "close": 10.6, "volume": 140},
            {"ticker": "AAA", "date": "2024-01-10", "open": 10.7, "high": 10.9, "low": 10.6, "close": 10.8, "volume": 150},
            {"ticker": "AAA", "date": "2024-01-11", "open": 10.8, "high": 11.1, "low": 10.7, "close": 11.0, "volume": 160},
        ]
    )
    price_history_df["date"] = pd.to_datetime(price_history_df["date"])

    features = build_market_features(events_df, price_history_df)

    assert len(features) == 1
    row = features.iloc[0]
    assert row["previous_close"] == 10.6
    assert row["event_close"] == 10.8
    assert row["next_trading_close"] == 11.0
    assert math.isclose(row["log_return_target"], math.log(11.0 / 10.8), rel_tol=1e-9)
    assert math.isclose(row["pre_return_5d"], math.log(10.6 / 10.0), rel_tol=1e-9)


def test_fetch_price_history_works_when_yfinance_download_has_no_proxy_arg(tmp_path, monkeypatch) -> None:
    calls: list[str] = []

    def fake_download(tickers, start=None, end=None, progress=False, auto_adjust=False, group_by="column", threads=False):
        calls.append(tickers)
        return pd.DataFrame(
            {
                "Open": [10.0, 10.5],
                "High": [10.1, 10.8],
                "Low": [9.9, 10.4],
                "Close": [10.0, 10.7],
                "Volume": [100, 120],
            },
            index=pd.to_datetime(["2024-01-09", "2024-01-10"]),
        )

    monkeypatch.setenv("YFINANCE_PROXY", "http://proxy.example.com:8080")
    monkeypatch.setattr(load_yfinance, "yf", SimpleNamespace(download=fake_download))

    history = fetch_price_history(
        ticker="AAA",
        start_date="2024-01-01",
        end_date="2024-01-31",
        raw_price_dir=tmp_path,
        refresh=True,
    )

    assert len(history) == 2
    assert calls == ["AAA"]
    assert history["close"].tolist() == [10.0, 10.7]


def test_fetch_company_metadata_retries_with_dash_symbol(monkeypatch) -> None:
    requested: list[str] = []

    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def get_info(self) -> dict[str, object]:
            requested.append(self.symbol)
            if self.symbol == "BF.B":
                raise RuntimeError("Quote not found")
            return {"longName": "Brown-Forman", "exchange": "NYSE"}

    monkeypatch.setattr(load_yfinance, "yf", SimpleNamespace(Ticker=FakeTicker))

    metadata = fetch_company_metadata("BF.B")

    assert requested == ["BF.B", "BF-B"]
    assert metadata["yf_symbol_used"] == "BF-B"
    assert metadata["yf_long_name"] == "Brown-Forman"
