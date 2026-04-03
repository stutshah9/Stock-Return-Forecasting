"""PyTorch dataset wrappers for multimodal earnings events."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset


class EarningsDataset(Dataset):
    """Dataset wrapper for multimodal earnings event dictionaries.

        The dataset expects pre-loaded cached event dictionaries and converts the
        stored features, sentiment aggregates, and target into PyTorch tensors
        during item access.
    """

    @classmethod
    def compute_feature_stats(
        cls,
        events: list[dict[str, Any]],
    ) -> dict[str, list[float]]:
        """Compute z-score normalization statistics from cached event features."""

        import numpy as np

        arr = np.array([e["features"] for e in events], dtype=float)
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std": np.where(arr.std(axis=0) < 1e-8, 1.0, arr.std(axis=0)).tolist(),
        }

    def __init__(
        self,
        events: list[dict[str, Any]],
        feature_stats: dict[str, list[float]] | None = None,
    ) -> None:
        """Initialize the dataset with a list of event dictionaries.

        Args:
            events: Cached event payloads created by ``data/build_cache.py``.
            feature_stats: Optional mean/std statistics for z-score normalization.
        """

        self.events = events
        self.stats = feature_stats

    def __len__(self) -> int:
        """Return the number of events available in the dataset."""

        return len(self.events)

    def __getitem__(self, index: int) -> dict[str, str | list[str] | Tensor]:
        """Return a single event formatted for model consumption.

        Args:
            index: Position of the requested event in the dataset.

        Returns:
            A dictionary containing transcript text, raw social posts, normalized
            financial features, raw financial features for metadata uses,
            sentiment features, and a scalar label tensor.
        """

        try:
            event = self.events[index]
        except Exception:
            event = {}

        transcript = str(event.get("transcript", ""))

        raw_posts = event.get("sentiment_raw", [])
        if isinstance(raw_posts, list):
            posts = [str(post) for post in raw_posts]
        else:
            posts = []

        feature_values: list[float] = []
        for feature_value in list(event.get("features", []))[:3]:
            try:
                feature_values.append(float(feature_value))
            except Exception:
                feature_values.append(0.0)
        while len(feature_values) < 3:
            feature_values.append(0.0)

        sentiment_values: list[float] = []
        for sentiment_value in list(event.get("sentiment_features", []))[:2]:
            try:
                sentiment_values.append(float(sentiment_value))
            except Exception:
                sentiment_values.append(0.0)
        while len(sentiment_values) < 2:
            sentiment_values.append(0.0)

        try:
            label_value = float(event.get("label", 0.0))
        except Exception:
            label_value = 0.0

        raw = torch.tensor(feature_values, dtype=torch.float32)
        if self.stats:
            mean = torch.tensor(self.stats["mean"], dtype=torch.float32)
            std = torch.tensor(self.stats["std"], dtype=torch.float32)
            features = (raw - mean) / std
        else:
            features = raw
        sentiment = torch.tensor(sentiment_values, dtype=torch.float32)
        label = torch.tensor(label_value, dtype=torch.float32)

        return {
            "ticker": str(event.get("ticker", "")),
            "date": str(event.get("date", "")),
            "transcript": transcript,
            "posts": posts,
            "features": features,
            "raw_features": raw,
            "sentiment": sentiment,
            "label": label,
        }
