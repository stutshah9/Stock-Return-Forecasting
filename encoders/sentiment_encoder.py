"""Sentiment feature aggregation and encoding utilities."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from textblob import TextBlob
import yaml


def _load_encoder_config() -> dict[str, Any]:
    """Load encoder defaults from the project configuration file."""

    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
    except OSError:
        config = {}
    return config


_ENCODER_CONFIG = _load_encoder_config()
_DEFAULT_EMBED_DIM = int(_ENCODER_CONFIG.get("model", {}).get("embed_dim", 64))
_DEFAULT_DROPOUT = float(_ENCODER_CONFIG.get("model", {}).get("dropout", 0.1))


def aggregate_posts(posts: list[str]) -> Tensor:
    """Aggregate post texts into sentiment and volume features.

    Args:
        posts: Raw post strings.

    Returns:
        A tensor of shape ``[2]`` containing average sentiment polarity and the
        log-transformed message volume.
    """

    if not posts:
        return torch.tensor([0.0, 0.0], dtype=torch.float32)

    sentiment_scores: list[float] = []
    for post in posts:
        try:
            sentiment_scores.append(float(TextBlob(str(post)).sentiment.polarity))
        except Exception:
            continue

    average_sentiment = (
        float(sum(sentiment_scores) / len(sentiment_scores))
        if sentiment_scores
        else 0.0
    )
    log_message_volume = float(math.log1p(len(posts)))
    return torch.tensor(
        [average_sentiment, log_message_volume],
        dtype=torch.float32,
    )


class SentimentEncoder(nn.Module):
    """Encode aggregated sentiment statistics into the shared embedding space."""

    def __init__(
        self,
        input_dim: int = 2,
        embed_dim: int = _DEFAULT_EMBED_DIM,
        dropout: float = _DEFAULT_DROPOUT,
    ) -> None:
        """Initialize the sentiment encoder.

        Args:
            input_dim: Number of aggregated sentiment features.
            embed_dim: Output embedding dimension.
            dropout: Unused placeholder kept for signature compatibility.
        """

        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.GELU(),
            nn.Linear(16, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode a batch of sentiment features into shape ``[batch, embed_dim]``.

        Args:
            x: Input tensor with shape ``[batch, 2]``.

        Returns:
            Encoded sentiment embeddings with shape ``[batch, embed_dim]``.
        """

        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.network(x.float())
