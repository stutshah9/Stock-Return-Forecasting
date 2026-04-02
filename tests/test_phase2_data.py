import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.dataset import EarningsDataset
from data.loader import _reddit_json, fetch_reddit_posts, load_earnings_event


def _write_fixtures(root: str) -> None:
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)

    transcripts = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "date": "2023-02-02",
                "text": "We delivered strong quarter results and robust guidance.",
            },
            {
                "ticker": "AAPL",
                "date": "2023-05-04",
                "text": "Services and wearables remained resilient this quarter.",
            },
        ]
    )
    financials = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "date": "2023-02-02",
                "earnings_surprise": 0.08,
                "std_dev_surprise": 0.04,
                "implied_vol": 0.32,
            },
            {
                "ticker": "AAPL",
                "date": "2023-05-04",
                "earnings_surprise": 0.03,
                "std_dev_surprise": 0.05,
                "implied_vol": 0.29,
            },
        ]
    )

    transcripts.to_csv(os.path.join(data_dir, "transcripts.csv"), index=False)
    financials.to_csv(os.path.join(data_dir, "financials.csv"), index=False)
    print("Fixture files created.")


def main() -> None:
    root = os.path.dirname(os.path.dirname(__file__))

    # Step A
    _write_fixtures(root)

    # Step B
    url = "https://www.reddit.com/search.json?q=AAPL&sort=new&limit=5&type=link"
    data = _reddit_json(url)
    assert isinstance(data, dict), "Reddit response must be a dictionary"
    assert "data" in data, "Response missing top-level data key"
    assert "children" in data["data"], "Response missing data.children"
    assert isinstance(data["data"]["children"], list), "children must be a list"

    children = data["data"]["children"]
    for item in children:
        payload = item.get("data", {})
        assert "title" in payload, "Post missing title"
        assert "selftext" in payload, "Post missing selftext"
        assert "created_utc" in payload, "Post missing created_utc"

    print("Live Reddit post count:", len(children))
    if children:
        sample_title = str(children[0].get("data", {}).get("title", ""))
        print("Sample title:", sample_title[:80])

    # Step C
    event = load_earnings_event("AAPL", "2023-02-02")
    assert isinstance(event, dict), "Event must be a dictionary"
    assert "ticker" in event, "Event missing ticker"
    assert "date" in event, "Event missing date"
    assert "transcript" in event, "Event missing transcript"
    assert "sentiment_posts" in event, "Event missing sentiment_posts"
    assert "features" in event, "Event missing features"
    assert "label" in event, "Event missing label"
    assert isinstance(event["ticker"], str), "ticker must be a string"
    assert isinstance(event["date"], str), "date must be a string"
    assert isinstance(event["transcript"], str), "transcript must be a string"
    assert isinstance(event["sentiment_posts"], list), "sentiment_posts must be a list"
    assert isinstance(event["features"], dict), "features must be a dictionary"
    assert isinstance(event["label"], float), "label must be a float"

    print("Loaded event ticker:", event["ticker"])
    print("Loaded event date:", event["date"])
    print("Loaded event feature keys:", sorted(list(event["features"].keys())))
    print("Loaded event label:", event["label"])

    # Step D
    event_2 = load_earnings_event("AAPL", "2023-05-04")
    ds = EarningsDataset([event, event_2])
    assert len(ds) == 2, "Dataset length must be 2"

    item = ds[0]
    assert isinstance(item, dict), "Dataset item must be a dictionary"
    assert "transcript" in item, "Dataset item missing transcript"
    assert "posts" in item, "Dataset item missing posts"
    assert "features" in item, "Dataset item missing features"
    assert "label" in item, "Dataset item missing label"
    assert isinstance(item["features"], torch.Tensor), "features must be a Tensor"
    assert item["features"].shape == (3,), "features shape must be (3,)"
    assert item["features"].dtype == torch.float32, "features dtype must be float32"
    assert isinstance(item["label"], torch.Tensor), "label must be a Tensor"
    assert item["label"].shape == (), "label must be a scalar tensor"
    assert item["label"].dtype == torch.float32, "label dtype must be float32"

    _ = fetch_reddit_posts("AAPL", "2023-02-02")

    print("All data pipeline tests passed.")


if __name__ == "__main__":
    main()
