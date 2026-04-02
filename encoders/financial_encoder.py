"""Structured financial feature encoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
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


class FinancialEncoder(nn.Module):
    """Encode structured financial indicators into the shared embedding space."""

    def __init__(
        self,
        input_dim: int = 3,
        embed_dim: int = _DEFAULT_EMBED_DIM,
        dropout: float = _DEFAULT_DROPOUT,
    ) -> None:
        """Initialize the financial feature encoder.

        Args:
            input_dim: Number of structured input features.
            embed_dim: Output embedding dimension.
            dropout: Dropout probability applied after the hidden layer.
        """

        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def feature_names(self) -> list[str]:
        """Return the ordered structured features expected by the encoder."""

        return ["sue", "momentum", "implied_vol"]

    def forward(self, x: Tensor) -> Tensor:
        """Encode a batch of financial features into shape ``[batch, embed_dim]``.

        Args:
            x: Input tensor with shape ``[batch, 3]``.

        Returns:
            Encoded financial embeddings with shape ``[batch, embed_dim]``.
        """

        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.network(x.float())
