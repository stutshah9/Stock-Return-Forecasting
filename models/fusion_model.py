"""Multimodal fusion model for earnings-return forecasting."""

from __future__ import annotations

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
    """Fuse transcript, financial, and sentiment inputs for quantile forecasting."""

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
        self.quantile_levels = self._configured_quantile_levels(config)
        self.coverage_quantile_pairs = self._coverage_quantile_pairs(config, self.quantile_levels)
        self.default_interval_coverage = self._default_interval_coverage(self.coverage_quantile_pairs)
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
        self.quantile_head = nn.Linear(self.embed_dim, len(self.quantile_levels))
        self.point_head = nn.Linear(self.embed_dim, 1)
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

    @staticmethod
    def _configured_quantile_levels(config: dict[str, Any]) -> tuple[float, ...]:
        """Return the sorted quantile levels used by the prediction head."""

        training_config = config.get("training", {})
        quantiles = sorted(
            {
                float(value)
                for value in training_config.get(
                    "pinball_quantiles",
                    [0.025, 0.05, 0.10, 0.50, 0.90, 0.95, 0.975],
                )
                if 0.0 < float(value) < 1.0
            }
        )
        if 0.50 not in quantiles:
            quantiles.append(0.50)
            quantiles.sort()
        return tuple(float(value) for value in quantiles)

    @staticmethod
    def _coverage_quantile_pairs(
        config: dict[str, Any],
        quantile_levels: tuple[float, ...],
    ) -> dict[float, tuple[float, float]]:
        """Match each requested coverage level to its lower/upper quantile pair."""

        calibration_config = config.get("calibration", {})
        coverages = [
            float(value)
            for value in calibration_config.get("coverage_levels", [0.80, 0.90, 0.95])
            if 0.0 < float(value) < 1.0
        ]
        if not coverages:
            coverages = [0.80, 0.90, 0.95]

        level_set = set(quantile_levels)
        pairs: dict[float, tuple[float, float]] = {}
        for coverage in coverages:
            alpha = 1.0 - coverage
            lower = round(alpha / 2.0, 6)
            upper = round(1.0 - alpha / 2.0, 6)
            if lower not in level_set or upper not in level_set:
                raise ValueError(
                    "pinball_quantiles must include matching lower/upper levels "
                    f"for calibration coverage {coverage:.2f}. "
                    f"Missing pair ({lower}, {upper})."
                )
            pairs[float(coverage)] = (float(lower), float(upper))
        return pairs

    @staticmethod
    def _default_interval_coverage(
        coverage_quantile_pairs: dict[float, tuple[float, float]],
    ) -> float:
        """Choose the primary interval coverage used for confidence features."""

        if 0.90 in coverage_quantile_pairs:
            return 0.90
        return sorted(coverage_quantile_pairs.keys())[len(coverage_quantile_pairs) // 2]

    def _default_interval_quantiles(self) -> tuple[float, float]:
        """Return the default interval pair used for diagnostics and confidence heads."""

        return self.coverage_quantile_pairs[self.default_interval_coverage]

    def _base_interval_from_predictions(
        self,
        quantile_predictions: Tensor,
        lower_quantile: float,
        upper_quantile: float,
    ) -> tuple[Tensor, Tensor]:
        """Select the predicted lower and upper bounds for a target coverage level."""

        lower_index = self.quantile_levels.index(lower_quantile)
        upper_index = self.quantile_levels.index(upper_quantile)
        return quantile_predictions[:, lower_index], quantile_predictions[:, upper_index]

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
            A dictionary with interval quantiles, attention-derived confidence,
            explanation features, and optional natural-language explanations.
        """

        transcript_inputs = transcripts if transcripts is not None else transcript
        if transcript_inputs is None:
            raise ValueError("Transcript inputs are required.")
        if financial is None:
            raise ValueError("Financial inputs are required.")
        if sentiment is None:
            raise ValueError("Sentiment inputs are required.")

        device = self.quantile_head.weight.device
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

        quantile_params = self.quantile_head(fused_embedding)
        quantile_predictions, _ = torch.sort(quantile_params, dim=-1)
        median_index = self.quantile_levels.index(0.50)
        quantile_median = quantile_predictions[:, median_index]
        point_mu = self.point_head(fused_embedding).squeeze(-1)
        mu = point_mu
        default_lower_quantile, default_upper_quantile = self._default_interval_quantiles()
        q_low, q_high = self._base_interval_from_predictions(
            quantile_predictions,
            default_lower_quantile,
            default_upper_quantile,
        )
        interval_width = q_high - q_low
        base_intervals = {
            float(coverage): self._base_interval_from_predictions(
                quantile_predictions,
                lower_quantile,
                upper_quantile,
            )
            for coverage, (lower_quantile, upper_quantile) in self.coverage_quantile_pairs.items()
        }

        explanation_vec = self.explanation_head(fused_embedding)
        attention_stability = self._attention_stability_score(attention_history)
        modality_consistency = self._modality_consistency_score(modality_tokens)
        interval_confidence = self._interval_confidence(interval_width)
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
            "quantile_levels": torch.tensor(
                self.quantile_levels,
                device=quantile_predictions.device,
                dtype=quantile_predictions.dtype,
            ),
            "quantile_predictions": quantile_predictions,
            "base_intervals": base_intervals,
            "q_low": q_low,
            "q_high": q_high,
            "mu": mu,
            "point_mu": point_mu,
            "quantile_median": quantile_median,
            "interval_width": interval_width,
            "explanation_vec": explanation_vec,
            "attention_stability": attention_stability,
            "modality_consistency": modality_consistency,
            "interval_confidence": interval_confidence,
            "variance_confidence": interval_confidence,
            "introspective_score": introspective_score,
            "attention_weights": tuple(
                torch.stack(layer_attention, dim=1)
                for layer_attention in attention_history
            ),
        }
        if return_explanations:
            outputs["explanations"] = self._build_explanations(
                mu=mu,
                interval_width=interval_width,
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

    def _interval_confidence(self, interval_width: Tensor) -> Tensor:
        """Map narrower predictive intervals to higher confidence on a 0-1 scale."""

        width_center = interval_width.detach().mean()
        width_scale = interval_width.detach().std(unbiased=False).clamp(min=1e-3)
        return torch.sigmoid(-(interval_width - width_center) / width_scale)

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
        interval_width: Tensor,
        modality_tokens: Tensor,
    ) -> list[str]:
        """Generate lightweight natural-language forecast explanations."""

        modality_names = ("transcript", "financial", "sentiment")
        modality_strength = modality_tokens.norm(dim=-1)
        explanations: list[str] = []

        for index in range(mu.shape[0]):
            dominant_index = int(modality_strength[index].argmax().item())
            direction = "positive" if float(mu[index].item()) >= 0.0 else "negative"
            uncertainty = "elevated" if float(interval_width[index].item()) > 0.10 else "moderate"
            explanations.append(
                "The model forecasts a "
                f"{direction} next-day return with {uncertainty} uncertainty, "
                f"driven most strongly by the {modality_names[dominant_index]} modality "
                "after cross-modal interaction."
            )

        return explanations

    def loss(self, output: dict[str, Any], y: Tensor) -> Tensor:
        """Compute pinball quantile loss with optional auxiliary regularizers.

        Args:
            output: Model outputs from ``forward``.
            y: Ground-truth returns with shape ``[B]``.

        Returns:
            A scalar combined training loss.
        """

        q_low = output["q_low"]
        q_high = output["q_high"]
        mu = output.get("mu")
        if mu is None:
            mu = 0.5 * (q_low + q_high)
        quantile_median = output.get("quantile_median")
        if quantile_median is None:
            quantile_median = mu
        interval_width = output.get("interval_width")
        if interval_width is None:
            interval_width = (q_high - q_low).clamp(min=1e-6)
        targets = y.to(device=mu.device, dtype=torch.float32).view(-1)

        quantile_levels = output.get("quantile_levels")
        quantile_predictions = output.get("quantile_predictions")
        if quantile_levels is None or quantile_predictions is None:
            default_lower_quantile, default_upper_quantile = self._default_interval_quantiles()
            quantile_levels = torch.tensor(
                [default_lower_quantile, 0.50, default_upper_quantile],
                device=targets.device,
                dtype=targets.dtype,
            )
            quantile_predictions = torch.stack([q_low, mu, q_high], dim=-1)
        else:
            quantile_levels = quantile_levels.to(device=targets.device, dtype=targets.dtype).view(-1)
            quantile_predictions = quantile_predictions.to(device=targets.device, dtype=targets.dtype)

        quantile_errors = targets.unsqueeze(-1) - quantile_predictions
        pinball_components = torch.maximum(
            quantile_levels.unsqueeze(0) * quantile_errors,
            (quantile_levels.unsqueeze(0) - 1.0) * quantile_errors,
        )
        pinball_loss = pinball_components.mean()
        point_loss_weight = float(
            self.config.get("training", {}).get("point_loss_weight", 1.0)
        )
        point_loss_beta = float(
            self.config.get("training", {}).get("point_loss_beta", 0.02)
        )
        direction_loss_weight = float(
            self.config.get("training", {}).get("direction_loss_weight", 0.05)
        )
        point_loss = F.smooth_l1_loss(mu, targets, beta=max(point_loss_beta, 1e-6))
        median_anchor_loss = F.smooth_l1_loss(mu, quantile_median.detach(), beta=max(point_loss_beta, 1e-6))
        target_scale = targets.detach().abs().mean().clamp(min=1e-3)
        dir_logits = mu / target_scale
        dir_target = (targets >= 0.0).float()
        dir_loss = F.binary_cross_entropy_with_logits(dir_logits, dir_target)

        uncertainty_alignment_weight = float(
            self.config.get("training", {}).get("uncertainty_alignment_weight", 0.05)
        )
        confidence_calibration_weight = float(
            self.config.get("training", {}).get("confidence_calibration_weight", 0.10)
        )
        attention_alignment_weight = float(
            self.config.get("training", {}).get("attention_alignment_weight", 0.05)
        )
        explanation_alignment_weight = float(
            self.config.get("training", {}).get("explanation_alignment_weight", 0.05)
        )

        uncertainty_alignment_loss = torch.zeros((), device=targets.device, dtype=targets.dtype)
        confidence_calibration_loss = torch.zeros((), device=targets.device, dtype=targets.dtype)
        attention_alignment_loss = torch.zeros((), device=targets.device, dtype=targets.dtype)
        explanation_alignment_loss = torch.zeros((), device=targets.device, dtype=targets.dtype)

        if (
            uncertainty_alignment_weight > 0.0
            or confidence_calibration_weight > 0.0
            or attention_alignment_weight > 0.0
            or explanation_alignment_weight > 0.0
        ):
            residual_scale_target = (targets - mu).detach().abs().clamp(min=1e-4, max=1.0)
            interval_miss = torch.maximum(q_low - targets, targets - q_high).clamp(min=0.0)
            confidence_target = self._confidence_target(
                (interval_miss.detach() + 0.25 * interval_width.detach()).clamp(min=1e-4, max=1.0)
            )
            interval_confidence = output.get("interval_confidence", self._interval_confidence(interval_width))

            if uncertainty_alignment_weight > 0.0:
                uncertainty_alignment_loss = F.smooth_l1_loss(
                    interval_width,
                    (2.0 * residual_scale_target).clamp(min=1e-4, max=2.0),
                )
            if confidence_calibration_weight > 0.0:
                confidence_calibration_loss = F.smooth_l1_loss(
                    output["introspective_score"],
                    confidence_target,
                )
            if attention_alignment_weight > 0.0:
                attention_alignment_loss = F.smooth_l1_loss(
                    output.get("attention_stability", output["introspective_score"]),
                    interval_confidence,
                )
            if explanation_alignment_weight > 0.0:
                explanation_alignment_loss = F.smooth_l1_loss(
                    output["introspective_score"],
                    interval_confidence,
                )

        total_loss = (
            pinball_loss
            + point_loss_weight * (point_loss + 0.1 * median_anchor_loss)
            + direction_loss_weight * dir_loss
            + uncertainty_alignment_weight * uncertainty_alignment_loss
            + confidence_calibration_weight * confidence_calibration_loss
            + attention_alignment_weight * attention_alignment_loss
            + explanation_alignment_weight * explanation_alignment_loss
        )

        if os.environ.get("DEBUG_DIRECTIONAL_LOSS") == "1" and not self._loss_debug_printed:
            print(f"dir_loss value: {dir_loss.item():.6f}")
            print(f"pinball_loss value: {pinball_loss.item():.6f}")
            print(f"point_loss value: {point_loss.item():.6f}")
            print(f"median_anchor_loss value: {median_anchor_loss.item():.6f}")
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
            print(f"mean interval width: {interval_width.mean().item():.6f}")
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
