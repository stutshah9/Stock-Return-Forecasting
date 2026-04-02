import os
import sys

import torch

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

    outputs = model(financial=financial, sentiment=sentiment, transcripts=transcripts)
    assert isinstance(outputs, dict), "Model output must be a dictionary"
    assert "mu" in outputs, "Model output missing mu"
    assert "log_sigma" in outputs, "Model output missing log_sigma"
    assert "introspective_score" in outputs, "Model output missing introspective_score"
    assert "explanation_vec" in outputs, "Model output missing explanation_vec"

    mu = outputs["mu"]
    log_sigma = outputs["log_sigma"]
    introspective_score = outputs["introspective_score"]
    explanation_vec = outputs["explanation_vec"]

    assert mu.shape == (batch_size,), "mu shape mismatch"
    assert log_sigma.shape == (batch_size,), "log_sigma shape mismatch"
    assert introspective_score.shape == (batch_size,), "introspective_score shape mismatch"
    assert explanation_vec.shape == (batch_size, 32), "explanation_vec shape mismatch"

    assert torch.all(log_sigma <= 3.0).item(), "log_sigma must be clamped to upper bound"
    assert torch.all(log_sigma >= -3.0).item(), "log_sigma must be clamped to lower bound"
    assert torch.all(introspective_score >= 0.0).item(), "introspective_score must be >= 0"
    assert torch.all(introspective_score <= 1.0).item(), "introspective_score must be <= 1"

    for key in ("mu", "log_sigma", "introspective_score", "explanation_vec"):
        tensor = outputs[key]
        assert not torch.isnan(tensor).any().item(), "Model output contains NaN"

    y_true = torch.randn(batch_size, dtype=torch.float32)
    loss_value = model.loss(outputs, y_true)
    assert isinstance(loss_value, torch.Tensor), "loss must return a Tensor"
    assert loss_value.shape == (), "loss must be scalar"
    assert not torch.isnan(loss_value).item(), "loss must not be NaN"
    assert loss_value.item() > 0.0, "loss must be positive"

    print("All fusion model tests passed.")


if __name__ == "__main__":
    main()
