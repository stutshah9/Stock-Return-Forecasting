from __future__ import annotations

import pandas as pd

from src.data.join_data import aggregate_reddit_windows, build_company_level_dataset, build_event_level_dataset


def test_join_data_builds_reddit_windows_and_final_tables() -> None:
    transcripts_df = pd.DataFrame(
        [
            {
                "event_id": "AAA_2024-01-10_2024_Q1",
                "ticker": "AAA",
                "company_name": "AAA Corp",
                "event_date": pd.Timestamp("2024-01-10"),
                "event_datetime": pd.Timestamp("2024-01-10 16:30:00"),
                "fiscal_year": 2024,
                "fiscal_quarter": "Q1",
                "transcript_text": "Confident commentary.",
                "transcript_word_count": 2,
                "transcript_char_count": 21,
                "transcript_source_dataset": "demo",
            }
        ]
    )
    market_features_df = pd.DataFrame(
        [
            {
                "event_id": "AAA_2024-01-10_2024_Q1",
                "ticker": "AAA",
                "event_date": pd.Timestamp("2024-01-10"),
                "event_close": 10.0,
                "next_trading_close": 10.5,
                "log_return_target": 0.04879,
            }
        ]
    )
    sec_df = pd.DataFrame(
        [
            {
                "event_id": "AAA_2024-01-10_2024_Q1",
                "ticker": "AAA",
                "event_date": pd.Timestamp("2024-01-10"),
                "cik": 1234,
                "sec_revenue": 100.0,
                "fundamentals_available": True,
            }
        ]
    )
    posts_df = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "post_id": "p1",
                "text": "Bullish post",
                "score": 5,
                "created_timestamp": pd.Timestamp("2024-01-09 10:00:00"),
            },
            {
                "ticker": "AAA",
                "post_id": "p2",
                "text": "Older post",
                "score": 2,
                "created_timestamp": pd.Timestamp("2024-01-05 10:00:00"),
            },
        ]
    )
    comments_df = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "comment_id": "c1",
                "body": "Supportive comment",
                "score": 3,
                "created_timestamp": pd.Timestamp("2024-01-08 11:00:00"),
            }
        ]
    )
    metadata_df = pd.DataFrame([{"ticker": "AAA", "yf_exchange": "NMS"}])
    price_history_df = pd.DataFrame(
        [
            {"ticker": "AAA", "date": pd.Timestamp("2024-01-01"), "close": 9.0},
            {"ticker": "AAA", "date": pd.Timestamp("2024-01-11"), "close": 10.5},
        ]
    )

    reddit_agg = aggregate_reddit_windows(transcripts_df, posts_df, comments_df)
    assert reddit_agg.loc[0, "reddit_post_count_1d"] == 1
    assert reddit_agg.loc[0, "reddit_comment_count_3d"] == 1
    assert "Bullish post" in reddit_agg.loc[0, "reddit_text_7d"]

    event_level = build_event_level_dataset(transcripts_df, market_features_df, sec_df, posts_df, comments_df)
    company_level = build_company_level_dataset(transcripts_df, price_history_df, metadata_df, sec_df, posts_df, comments_df)

    assert len(event_level) == 1
    assert len(company_level) == 1
    assert event_level.loc[0, "sec_revenue"] == 100.0
    assert company_level.loc[0, "reddit_posts_total"] == 2
