from __future__ import annotations

import pandas as pd

from src.data.load_transcripts import normalize_transcript_frame


def test_normalize_transcript_frame_handles_candidate_columns() -> None:
    raw_df = pd.DataFrame(
        [
            {
                "symbol": "aapl",
                "company": "Apple Inc.",
                "date": "2024-01-25",
                "quarter": "Q1",
                "year": 2024,
                "content": " Strong quarter with services growth. ",
                "id": "row-1",
            }
        ]
    )

    normalized = normalize_transcript_frame(raw_df, dataset_name="demo/transcripts")

    assert len(normalized) == 1
    row = normalized.iloc[0]
    assert row["ticker"] == "AAPL"
    assert row["company_name"] == "Apple Inc."
    assert str(row["event_date"].date()) == "2024-01-25"
    assert row["fiscal_year"] == 2024
    assert row["fiscal_quarter"] == "Q1"
    assert row["transcript_text"] == "Strong quarter with services growth."
    assert row["event_id"].startswith("AAPL_2024-01-25_2024_Q1")
