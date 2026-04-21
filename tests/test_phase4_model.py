import os
import sys
from copy import deepcopy

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.fusion_model import MultimodalForecastModel


def main() -> None:
    root = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(root, "config.yaml")

    model = MultimodalForecastModel.load_from_config(config_path)
    model.eval()

    batch_size = 4
    financial = torch.randn(batch_size, 3, dtype=torch.float32)
    sentiment = torch.randn(batch_size, 2, dtype=torch.float32)
    transcripts = [
        "AAPL delivered solid revenue growth with strong services momentum.",
        "Management highlighted macro uncertainty and mixed regional demand.",
        "Margins improved due to supply chain normalization and cost discipline.",
        "Guidance suggests cautious optimism for the upcoming quarter.",
    ]

    outputs = model(
        financial=financial,
        sentiment=sentiment,
        transcripts=transcripts,
        return_explanations=True,
    )
    assert isinstance(outputs, dict), "Model output must be a dictionary"
    assert "q_low" in outputs, "Model output missing q_low"
    assert "q_high" in outputs, "Model output missing q_high"
    assert "mu" in outputs, "Model output missing mu"
    assert "point_mu" in outputs, "Model output missing point_mu"
    assert "quantile_median" in outputs, "Model output missing quantile_median"
    assert "interval_width" in outputs, "Model output missing interval_width"
    assert "attention_stability" in outputs, "Model output missing attention_stability"
    assert "modality_consistency" in outputs, "Model output missing modality_consistency"
    assert "interval_confidence" in outputs, "Model output missing interval_confidence"
    assert "introspective_score" in outputs, "Model output missing introspective_score"
    assert "explanation_vec" in outputs, "Model output missing explanation_vec"
    assert "attention_weights" in outputs, "Model output missing attention_weights"
    assert "explanations" in outputs, "Model output missing explanations"

    q_low = outputs["q_low"]
    q_high = outputs["q_high"]
    mu = outputs["mu"]
    point_mu = outputs["point_mu"]
    quantile_median = outputs["quantile_median"]
    interval_width = outputs["interval_width"]
    attention_stability = outputs["attention_stability"]
    modality_consistency = outputs["modality_consistency"]
    interval_confidence = outputs["interval_confidence"]
    introspective_score = outputs["introspective_score"]
    explanation_vec = outputs["explanation_vec"]
    attention_weights = outputs["attention_weights"]
    explanations = outputs["explanations"]

    assert q_low.shape == (batch_size,), "q_low shape mismatch"
    assert q_high.shape == (batch_size,), "q_high shape mismatch"
    assert mu.shape == (batch_size,), "mu shape mismatch"
    assert point_mu.shape == (batch_size,), "point_mu shape mismatch"
    assert quantile_median.shape == (batch_size,), "quantile_median shape mismatch"
    assert interval_width.shape == (batch_size,), "interval_width shape mismatch"
    assert attention_stability.shape == (batch_size,), "attention_stability shape mismatch"
    assert modality_consistency.shape == (batch_size,), "modality_consistency shape mismatch"
    assert interval_confidence.shape == (batch_size,), "interval_confidence shape mismatch"
    assert introspective_score.shape == (batch_size,), "introspective_score shape mismatch"
    assert explanation_vec.shape == (batch_size, 32), "explanation_vec shape mismatch"
    assert len(attention_weights) == len(model.fusion_layers), "attention history length mismatch"
    assert len(explanations) == batch_size, "There must be one explanation per sample"

    assert torch.all(q_high >= q_low).item(), "q_high must be >= q_low"
    assert torch.all(interval_width > 0.0).item(), "interval_width must be strictly positive"
    assert torch.all(attention_stability >= 0.0).item(), "attention_stability must be >= 0"
    assert torch.all(attention_stability <= 1.0).item(), "attention_stability must be <= 1"
    assert torch.all(modality_consistency >= 0.0).item(), "modality_consistency must be >= 0"
    assert torch.all(modality_consistency <= 1.0).item(), "modality_consistency must be <= 1"
    assert torch.all(interval_confidence >= 0.0).item(), "interval_confidence must be >= 0"
    assert torch.all(interval_confidence <= 1.0).item(), "interval_confidence must be <= 1"
    assert torch.all(introspective_score >= 0.0).item(), "introspective_score must be >= 0"
    assert torch.all(introspective_score <= 1.0).item(), "introspective_score must be <= 1"
    assert torch.allclose(interval_width, q_high - q_low), "interval_width must equal q_high - q_low"
    assert not torch.allclose(introspective_score, attention_stability), "introspective_score should be a learned confidence signal rather than a direct copy of attention_stability"

    assert isinstance(model.transcript_encoder, nn.Module), "Transcript encoder missing"
    assert isinstance(model.financial_encoder, nn.Module), "Financial encoder missing"
    assert isinstance(model.sentiment_encoder, nn.Module), "Sentiment encoder missing"
    assert len(model.fusion_layers) >= 1, "Model must include at least one fusion layer"
    for fusion_layer in model.fusion_layers:
        assert len(fusion_layer.attentions) == 3, "Each fusion layer must cross-attend all three modalities"
        for attention in fusion_layer.attentions:
            assert isinstance(attention, nn.MultiheadAttention), "Fusion must use multi-head cross-attention"

    for layer_attention in attention_weights:
        assert layer_attention.shape == (batch_size, 3, 2), "Each fusion layer should store three cross-attention maps over the other two modalities"

    for key in (
        "q_low",
        "q_high",
        "mu",
        "point_mu",
        "quantile_median",
        "interval_width",
        "attention_stability",
        "modality_consistency",
        "interval_confidence",
        "introspective_score",
        "explanation_vec",
    ):
        tensor = outputs[key]
        assert not torch.isnan(tensor).any().item(), "Model output contains NaN"
    assert all(isinstance(explanation, str) and explanation for explanation in explanations), "Explanations must be non-empty strings"

    y_true = torch.randn(batch_size, dtype=torch.float32)
    loss_value = model.loss(outputs, y_true)
    assert isinstance(loss_value, torch.Tensor), "loss must return a Tensor"
    assert loss_value.shape == (), "loss must be scalar"
    assert not torch.isnan(loss_value).item(), "loss must not be NaN"
    assert loss_value.item() > 0.0, "loss must be positive"

    aligned_output = {
        "q_low": torch.tensor([-0.05, -0.35], dtype=torch.float32),
        "q_high": torch.tensor([0.05, -0.25], dtype=torch.float32),
        "mu": torch.zeros(2, dtype=torch.float32),
        "interval_width": torch.tensor([0.10, 0.10], dtype=torch.float32),
        "introspective_score": torch.tensor([0.9, 0.1], dtype=torch.float32),
        "interval_confidence": torch.tensor([0.9, 0.1], dtype=torch.float32),
    }
    misaligned_output = {
        "q_low": aligned_output["q_low"],
        "q_high": aligned_output["q_high"],
        "mu": aligned_output["mu"],
        "interval_width": aligned_output["interval_width"],
        "introspective_score": torch.tensor([0.1, 0.9], dtype=torch.float32),
        "interval_confidence": aligned_output["interval_confidence"],
    }
    zero_targets = torch.zeros(2, dtype=torch.float32)
    aligned_loss = model.loss(aligned_output, zero_targets)
    misaligned_loss = model.loss(misaligned_output, zero_targets)
    assert aligned_loss.item() < misaligned_loss.item(), "alignment regularizer should reward agreement between introspective confidence and interval confidence"

    preserved_training_config = deepcopy(model.config.get("training", {}))
    good_interval_output = {
        "q_low": torch.tensor([0.10, -0.35], dtype=torch.float32),
        "q_high": torch.tensor([0.30, -0.15], dtype=torch.float32),
        "mu": torch.tensor([0.20, -0.25], dtype=torch.float32),
        "interval_width": torch.tensor([0.20, 0.20], dtype=torch.float32),
        "introspective_score": torch.full((2,), 0.5, dtype=torch.float32),
        "interval_confidence": torch.full((2,), 0.5, dtype=torch.float32),
    }
    poor_interval_output = {
        "q_low": torch.tensor([-0.20, -0.05], dtype=torch.float32),
        "q_high": torch.tensor([-0.10, 0.05], dtype=torch.float32),
        "mu": torch.tensor([-0.15, 0.00], dtype=torch.float32),
        "interval_width": torch.tensor([0.10, 0.10], dtype=torch.float32),
        "introspective_score": torch.full((2,), 0.5, dtype=torch.float32),
        "interval_confidence": torch.full((2,), 0.5, dtype=torch.float32),
    }
    pinball_test_targets = torch.tensor([0.25, -0.30], dtype=torch.float32)
    zero_alignment_weights = {
        "uncertainty_alignment_weight": 0.0,
        "confidence_calibration_weight": 0.0,
        "attention_alignment_weight": 0.0,
        "explanation_alignment_weight": 0.0,
    }

    model.config["training"] = {
        **preserved_training_config,
        **zero_alignment_weights,
        "pinball_quantiles": [0.10, 0.90],
    }
    good_interval_loss = model.loss(good_interval_output, pinball_test_targets)
    poor_interval_loss = model.loss(poor_interval_output, pinball_test_targets)
    model.config["training"] = preserved_training_config

    assert good_interval_loss.item() >= 0.0, "quantile training loss should be non-negative"
    assert good_interval_loss.item() < poor_interval_loss.item(), "intervals closer to the target quantiles should receive lower pinball loss"

    print("All fusion model tests passed.")


if __name__ == "__main__":
    main()
