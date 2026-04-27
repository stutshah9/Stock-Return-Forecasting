"""Multimodal fusion model for earnings-return forecasting."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F
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


class CrossModalFusionLayer(nn.Module):
    """Symmetric three-way cross-attention over text, financial, and sentiment tokens."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim,
                    num_heads=num_heads,
                    batch_first=True,
                    dropout=dropout,
                )
                for _ in range(3)
            ]
        )
        self.pre_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(3)])
        self.post_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(3)])
        self.feedforwards = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 2, embed_dim),
                )
                for _ in range(3)
            ]
        )

    def forward(self, tokens: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Cross-attend each modality token to the other two modalities."""

        updated_tokens: list[Tensor] = []
        attention_maps: list[Tensor] = []

        for modality_index in range(tokens.size(1)):
            query = tokens[:, modality_index : modality_index + 1, :]
            other_indices = [
                index for index in range(tokens.size(1)) if index != modality_index
            ]
            context = tokens[:, other_indices, :]
            attn_output, attn_weights = self.attentions[modality_index](
                query=query,
                key=context,
                value=context,
                need_weights=True,
                average_attn_weights=False,
            )
            residual = self.pre_norms[modality_index](query + attn_output)
            ffn_output = self.feedforwards[modality_index](residual)
            updated = self.post_norms[modality_index](residual + ffn_output)
            updated_tokens.append(updated)
            attention_maps.append(attn_weights.mean(dim=1).squeeze(1))

        return torch.cat(updated_tokens, dim=1), attention_maps


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
        cache_dir = data_config.get("cache_dir")
        if cache_dir is None and self.text_frozen:
            cache_dir = str(Path(__file__).resolve().parents[1] / "data" / "transcript_cache")
        self.cache_dir = cache_dir
        self.num_attention_heads = 4
        self.num_fusion_layers = int(model_config.get("fusion_layers", 2))

        if self.embed_dim % self.num_attention_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by {self.num_attention_heads} "
                "for multimodal attention."
            )

        self.transcript_encoder = TranscriptEncoder(
            embed_dim=self.embed_dim,
            frozen=self.text_frozen,
            cache_dir=self.cache_dir,
            min_chunk_size=int(data_config.get("min_chunk_size", 256)),
            chunk_size=int(
                data_config.get(
                    "max_chunk_size",
                    max(int(data_config.get("chunk_size", 256)), 512),
                )
            ),
            unfrozen_backbone_layers=int(model_config.get("text_unfrozen_layers", 0)),
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

        self.fusion_layers = nn.ModuleList(
            [
                CrossModalFusionLayer(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_attention_heads,
                    dropout=self.dropout,
                )
                for _ in range(self.num_fusion_layers)
            ]
        )
        self.fusion_norm = nn.LayerNorm(self.embed_dim)
        self.gaussian_head = nn.Linear(self.embed_dim, 2)
        self.explanation_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 32),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(34, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, 1),
        )
        self._loss_debug_printed = False

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
        return_explanations: bool = False,
    ) -> dict[str, Any]:
        """Run a forward pass through the multimodal fusion model.

        Args:
            transcript: Batch of transcript strings.
            financial: Structured financial inputs with shape ``[B, 3]``.
            sentiment: Aggregated sentiment inputs with shape ``[B, 2]``.
            transcripts: Alias for ``transcript`` for compatibility with callers.
            return_explanations: Whether to include lightweight natural-language
                explanation strings in the returned dictionary.

        Returns:
            A dictionary with Gaussian parameters, attention-derived confidence,
            explanation features, and optional natural-language explanations.
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

        modality_tokens = torch.stack(
            (transcript_emb, financial_emb, sentiment_emb),
            dim=1,
        )
        attention_history: list[list[Tensor]] = []
        for fusion_layer in self.fusion_layers:
            modality_tokens, layer_attention = fusion_layer(modality_tokens)
            attention_history.append(layer_attention)
        fused_embedding = self.fusion_norm(modality_tokens.mean(dim=1))

        gaussian_params = self.gaussian_head(fused_embedding)
        mu = gaussian_params[:, 0]
        log_variance = torch.clamp(gaussian_params[:, 1], min=-6.0, max=6.0)
        variance = torch.exp(log_variance)
        log_sigma = 0.5 * log_variance

        explanation_vec = self.explanation_head(fused_embedding)
        attention_stability = self._attention_stability_score(attention_history)
        modality_consistency = self._modality_consistency_score(modality_tokens)
        modality_strength = modality_tokens.norm(dim=-1)
        variance_confidence = self._variance_confidence(log_sigma)
        confidence_inputs = torch.cat(
            (
                explanation_vec,
                attention_stability.unsqueeze(-1),
                modality_consistency.unsqueeze(-1),
            ),
            dim=-1,
        )
        # Keep the introspective head separate from the Gaussian head, then
        # explicitly regularize agreement in the loss.
        introspective_score = torch.sigmoid(self.confidence_head(confidence_inputs)).squeeze(-1)

        outputs: dict[str, Any] = {
            "mu": mu,
            "log_sigma": log_sigma,
            "log_variance": log_variance,
            "variance": variance,
            "explanation_vec": explanation_vec,
            "attention_stability": attention_stability,
            "modality_consistency": modality_consistency,
            "modality_strengths": modality_strength,
            "variance_confidence": variance_confidence,
            "introspective_score": introspective_score,
            "attention_weights": tuple(
                torch.stack(layer_attention, dim=1)
                for layer_attention in attention_history
            ),
        }
        if return_explanations:
            outputs["explanations"] = self._build_explanations(
                mu=mu,
                variance=variance,
                modality_tokens=modality_tokens,
            )
        return outputs

    def _attention_stability_score(self, attention_history: list[list[Tensor]]) -> Tensor:
        """Derive an introspective confidence score from layer-to-layer stability."""

        if not attention_history:
            raise ValueError("attention_history must contain at least one fusion layer.")

        if len(attention_history) == 1:
            batch_size = attention_history[0][0].shape[0]
            device = attention_history[0][0].device
            return torch.ones(batch_size, device=device)

        stacked_history = [
            torch.stack(layer_attention, dim=1)
            for layer_attention in attention_history
        ]
        layer_differences = [
            (current_layer - previous_layer).abs().mean(dim=(1, 2))
            for previous_layer, current_layer in zip(
                stacked_history,
                stacked_history[1:],
            )
        ]
        mean_difference = torch.stack(layer_differences, dim=0).mean(dim=0)
        return torch.clamp(1.0 - mean_difference, min=0.0, max=1.0)

    def _variance_confidence(self, log_sigma: Tensor) -> Tensor:
        """Map lower predictive dispersion to higher confidence on a 0-1 scale."""

        sigma_center = log_sigma.detach().mean()
        sigma_scale = log_sigma.detach().std(unbiased=False).clamp(min=1e-3)
        return torch.sigmoid(-(log_sigma - sigma_center) / sigma_scale)

    def _modality_consistency_score(self, modality_tokens: Tensor) -> Tensor:
        """Estimate cross-modality agreement from the fused token geometry."""

        centered = modality_tokens - modality_tokens.mean(dim=1, keepdim=True)
        disagreement = centered.norm(dim=-1).mean(dim=1)
        return 1.0 / (1.0 + disagreement)

    def _confidence_target(self, residual_magnitude: Tensor) -> Tensor:
        """Convert detached residual size into a bounded confidence target."""

        center = residual_magnitude.mean()
        scale = residual_magnitude.std(unbiased=False).clamp(min=1e-3)
        return torch.sigmoid(-(residual_magnitude - center) / scale)

    def _build_explanations(
        self,
        mu: Tensor,
        variance: Tensor,
        modality_tokens: Tensor,
    ) -> list[str]:
        """Generate lightweight natural-language forecast explanations."""

        modality_names = ("transcript", "financial", "sentiment")
        modality_strength = modality_tokens.norm(dim=-1)
        explanations: list[str] = []

        for index in range(mu.shape[0]):
            dominant_index = int(modality_strength[index].argmax().item())
            direction = "positive" if float(mu[index].item()) >= 0.0 else "negative"
            uncertainty = "elevated" if float(variance[index].item()) > 1.0 else "moderate"
            explanations.append(
                "The model forecasts a "
                f"{direction} next-day return with {uncertainty} uncertainty, "
                f"driven most strongly by the {modality_names[dominant_index]} modality "
                "after cross-modal interaction."
            )

        return explanations

    def loss(self, output: dict[str, Any], y: Tensor) -> Tensor:
        """Compute the Gaussian negative log-likelihood plus directional loss.

        Args:
            output: Model outputs from ``forward``.
            y: Ground-truth returns with shape ``[B]``.

        Returns:
            A scalar combined training loss.
        """

        mu = output["mu"]
        log_variance = output.get("log_variance")
        if log_variance is None:
            log_variance = 2.0 * output["log_sigma"]
        variance = output.get("variance")
        if variance is None:
            variance = torch.exp(log_variance)
        targets = y.to(device=mu.device, dtype=torch.float32).view(-1)

        nll_loss = 0.5 * (
            ((targets - mu) ** 2) / variance
            + log_variance
            + math.log(2.0 * math.pi)
        ).mean()

        mu_scale = mu.detach().std(unbiased=False).clamp(min=1e-6)
        mu_scaled = mu / mu_scale
        dir_logits = mu_scaled * 3.0
        dir_target = (targets > 0).float()
        dir_loss = F.binary_cross_entropy_with_logits(dir_logits, dir_target)

        residual_scale_target = (targets - mu).detach().abs().clamp(min=1e-4, max=1.0)
        confidence_target = self._confidence_target(residual_scale_target)
        uncertainty_alignment_loss = F.smooth_l1_loss(
            output["log_sigma"],
            torch.log(residual_scale_target),
        )
        uncertainty_alignment_weight = float(
            self.config.get("training", {}).get("uncertainty_alignment_weight", 0.05)
        )
        confidence_calibration_loss = F.smooth_l1_loss(
            output["introspective_score"],
            confidence_target,
        )
        confidence_calibration_weight = float(
            self.config.get("training", {}).get("confidence_calibration_weight", 0.10)
        )
        attention_alignment_loss = F.smooth_l1_loss(
            output.get("attention_stability", output["introspective_score"]),
            output.get("variance_confidence", self._variance_confidence(output["log_sigma"])),
        )
        attention_alignment_weight = float(
            self.config.get("training", {}).get("attention_alignment_weight", 0.05)
        )
        explanation_alignment_loss = F.smooth_l1_loss(
            output["introspective_score"],
            output.get("variance_confidence", self._variance_confidence(output["log_sigma"])),
        )
        explanation_alignment_weight = float(
            self.config.get("training", {}).get("explanation_alignment_weight", 0.05)
        )

        total_loss = (
            nll_loss
            + 0.3 * dir_loss
            + uncertainty_alignment_weight * uncertainty_alignment_loss
            + confidence_calibration_weight * confidence_calibration_loss
            + attention_alignment_weight * attention_alignment_loss
            + explanation_alignment_weight * explanation_alignment_loss
        )

        if os.environ.get("DEBUG_DIRECTIONAL_LOSS") == "1" and not self._loss_debug_printed:
            print(f"dir_loss value: {dir_loss.item():.6f}")
            print(f"nll_loss value: {nll_loss.item():.6f}")
            print(
                "uncertainty_alignment_loss value: "
                f"{uncertainty_alignment_loss.item():.6f}"
            )
            print(
                "confidence_calibration_loss value: "
                f"{confidence_calibration_loss.item():.6f}"
            )
            print(
                "attention_alignment_loss value: "
                f"{attention_alignment_loss.item():.6f}"
            )
            print(
                "explanation_alignment_loss value: "
                f"{explanation_alignment_loss.item():.6f}"
            )
            print(f"total loss: {total_loss.item():.6f}")
            print(f"fraction of mu > 0: {(mu > 0).float().mean().item():.6f}")
            print(f"fraction of y > 0: {(targets > 0).float().mean().item():.6f}")
            self._loss_debug_printed = True

        return total_loss


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
