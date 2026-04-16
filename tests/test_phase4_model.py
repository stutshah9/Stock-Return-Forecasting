import os
import sys

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
    assert "mu" in outputs, "Model output missing mu"
    assert "log_sigma" in outputs, "Model output missing log_sigma"
    assert "log_variance" in outputs, "Model output missing log_variance"
    assert "variance" in outputs, "Model output missing variance"
    assert "introspective_score" in outputs, "Model output missing introspective_score"
    assert "explanation_vec" in outputs, "Model output missing explanation_vec"
    assert "attention_weights" in outputs, "Model output missing attention_weights"
    assert "explanations" in outputs, "Model output missing explanations"

    mu = outputs["mu"]
    log_sigma = outputs["log_sigma"]
    log_variance = outputs["log_variance"]
    variance = outputs["variance"]
    introspective_score = outputs["introspective_score"]
    explanation_vec = outputs["explanation_vec"]
    attention_weights = outputs["attention_weights"]
    explanations = outputs["explanations"]

    assert mu.shape == (batch_size,), "mu shape mismatch"
    assert log_sigma.shape == (batch_size,), "log_sigma shape mismatch"
    assert log_variance.shape == (batch_size,), "log_variance shape mismatch"
    assert variance.shape == (batch_size,), "variance shape mismatch"
    assert introspective_score.shape == (batch_size,), "introspective_score shape mismatch"
    assert explanation_vec.shape == (batch_size, 32), "explanation_vec shape mismatch"
    assert len(attention_weights) == len(model.fusion_layers), "attention history length mismatch"
    assert len(explanations) == batch_size, "There must be one explanation per sample"

    assert torch.all(log_sigma <= 3.0).item(), "log_sigma must be clamped to upper bound"
    assert torch.all(log_sigma >= -3.0).item(), "log_sigma must be clamped to lower bound"
    assert torch.all(log_variance <= 6.0).item(), "log_variance must be clamped to upper bound"
    assert torch.all(log_variance >= -6.0).item(), "log_variance must be clamped to lower bound"
    assert torch.all(variance > 0.0).item(), "variance must be strictly positive"
    assert torch.all(introspective_score >= 0.0).item(), "introspective_score must be >= 0"
    assert torch.all(introspective_score <= 1.0).item(), "introspective_score must be <= 1"
    assert torch.allclose(log_variance, 2.0 * log_sigma), "log_variance must equal 2 * log_sigma"
    assert torch.allclose(variance, torch.exp(log_variance)), "variance must equal exp(log_variance)"

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

    for key in ("mu", "log_sigma", "log_variance", "variance", "introspective_score", "explanation_vec"):
        tensor = outputs[key]
        assert not torch.isnan(tensor).any().item(), "Model output contains NaN"
    assert all(isinstance(explanation, str) and explanation for explanation in explanations), "Explanations must be non-empty strings"

    y_true = torch.randn(batch_size, dtype=torch.float32)
    loss_value = model.loss(outputs, y_true)
    assert isinstance(loss_value, torch.Tensor), "loss must return a Tensor"
    assert loss_value.shape == (), "loss must be scalar"
    assert not torch.isnan(loss_value).item(), "loss must not be NaN"
    assert loss_value.item() > 0.0, "loss must be positive"

    print("All fusion model tests passed.")


if __name__ == "__main__":
    main()
