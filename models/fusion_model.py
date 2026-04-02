"""Multimodal fusion model for earnings-return forecasting."""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor, nn
import yaml

try:
    from encoders.financial_encoder import FinancialEncoder
    from encoders.sentiment_encoder import SentimentEncoder
    from encoders.text_encoder import TranscriptEncoder
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from encoders.financial_encoder import FinancialEncoder
    from encoders.sentiment_encoder import SentimentEncoder
    from encoders.text_encoder import TranscriptEncoder


class MultimodalForecastModel(nn.Module):
    """Fuse transcript, financial, and sentiment inputs for Gaussian forecasting."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the multimodal fusion model from a configuration dictionary.

        Args:
            config: Project configuration values loaded from ``config.yaml``.
        """

        super().__init__()
        self.config = config

        data_config = config.get("data", {})
        model_config = config.get("model", {})

        self.embed_dim = int(model_config.get("embed_dim", 64))
        self.dropout = float(model_config.get("dropout", 0.1))
        self.text_frozen = bool(model_config.get("text_frozen", True))
        self.cache_dir = data_config.get("cache_dir")

        if self.embed_dim % 4 != 0:
            raise ValueError("embed_dim must be divisible by 4 for 4-head attention.")

        self.transcript_encoder = TranscriptEncoder(
            embed_dim=self.embed_dim,
            frozen=self.text_frozen,
            cache_dir=self.cache_dir,
            chunk_size=int(data_config.get("chunk_size", 256)),
            max_chunks=int(data_config.get("max_chunks", 16)),
        )
        self.financial_encoder = FinancialEncoder(
            input_dim=3,
            embed_dim=self.embed_dim,
            dropout=self.dropout,
        )
        self.sentiment_encoder = SentimentEncoder(
            input_dim=2,
            embed_dim=self.embed_dim,
            dropout=self.dropout,
        )

        self.cross_attention = nn.MultiheadAttention(
            self.embed_dim,
            num_heads=4,
            batch_first=True,
        )
        self.gaussian_head = nn.Linear(self.embed_dim, 2)
        self.explanation_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 32),
        )
        self.introspective_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    @classmethod
    def load_from_config(cls, config_path: str) -> MultimodalForecastModel:
        """Instantiate the model from a YAML configuration file.

        Args:
            config_path: Absolute or relative path to ``config.yaml``.

        Returns:
            A configured ``MultimodalForecastModel`` instance.
        """

        with open(config_path, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
        return cls(config=config)

    def forward(
        self,
        transcript: list[str] | None = None,
        financial: Tensor | None = None,
        sentiment: Tensor | None = None,
        transcripts: list[str] | None = None,
    ) -> dict[str, Tensor]:
        """Run a forward pass through the multimodal fusion model.

        Args:
            transcript: Batch of transcript strings.
            financial: Structured financial inputs with shape ``[B, 3]``.
            sentiment: Aggregated sentiment inputs with shape ``[B, 2]``.
            transcripts: Alias for ``transcript`` for compatibility with callers.

        Returns:
            A dictionary with Gaussian parameters, explanation features, and an
            introspective explanation-confidence score.
        """

        transcript_inputs = transcripts if transcripts is not None else transcript
        if transcript_inputs is None:
            raise ValueError("Transcript inputs are required.")
        if financial is None:
            raise ValueError("Financial inputs are required.")
        if sentiment is None:
            raise ValueError("Sentiment inputs are required.")

        device = self.gaussian_head.weight.device
        financial = financial.to(device=device, dtype=torch.float32)
        sentiment = sentiment.to(device=device, dtype=torch.float32)

        transcript_emb = self.transcript_encoder(transcript_inputs)
        financial_emb = self.financial_encoder(financial)
        sentiment_emb = self.sentiment_encoder(sentiment)

        query = transcript_emb.unsqueeze(1)
        key_value = torch.stack((financial_emb, sentiment_emb), dim=1)
        attended_output, _ = self.cross_attention(query, key_value, key_value)
        fused_embedding = attended_output.squeeze(1)

        gaussian_params = self.gaussian_head(fused_embedding)
        mu = gaussian_params[:, 0]
        log_sigma = torch.clamp(gaussian_params[:, 1], min=-3.0, max=3.0)

        explanation_vec = self.explanation_head(fused_embedding)
        introspective_score = self.introspective_head(explanation_vec).squeeze(-1)

        return {
            "mu": mu,
            "log_sigma": log_sigma,
            "explanation_vec": explanation_vec,
            "introspective_score": introspective_score,
        }

    def loss(self, output: dict[str, Tensor], y: Tensor) -> Tensor:
        """Compute the Gaussian negative log-likelihood loss.

        Args:
            output: Model outputs from ``forward``.
            y: Ground-truth returns with shape ``[B]``.

        Returns:
            A scalar Gaussian negative log-likelihood loss.
        """

        mu = output["mu"]
        log_sigma = output["log_sigma"]
        targets = y.to(device=mu.device, dtype=torch.float32).view(-1)

        variance = torch.exp(2.0 * log_sigma)
        squared_error = (targets - mu) ** 2
        nll = 0.5 * (
            squared_error / variance + 2.0 * log_sigma + math.log(2.0 * math.pi)
        )

        # TODO: Add an explanation-alignment or calibration-aware regularizer.
        return nll.mean()


if __name__ == "__main__":
    config_file = Path(__file__).resolve().parents[1] / "config.yaml"
    model = MultimodalForecastModel.load_from_config(str(config_file))
    model.eval()

    batch_size = 2
    sample_transcripts = [
        "Revenue grew and margins expanded after the earnings release.",
        "Management cited macro uncertainty but maintained long-term targets.",
    ]
    sample_financial = torch.randn(batch_size, 3, dtype=torch.float32)
    sample_sentiment = torch.randn(batch_size, 2, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(
            transcripts=sample_transcripts,
            financial=sample_financial,
            sentiment=sample_sentiment,
        )

    for key, value in outputs.items():
        print(key, tuple(value.shape))
