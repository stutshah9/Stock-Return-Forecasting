import os
import subprocess
import sys
import tempfile

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def main() -> None:
    root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    config_path = os.path.join(root, "config.yaml")
    train_config_path = config_path

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if (
        str(config.get("training", {}).get("device", "auto")).lower() == "cuda"
        and not torch.cuda.is_available()
    ):
        config.setdefault("training", {})["device"] = "cpu"
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
            encoding="utf-8",
        ) as temp_config:
            yaml.safe_dump(config, temp_config)
            train_config_path = temp_config.name

    subprocess.run(
        ["python3", "experiments/train.py", "--dry-run", "--config", train_config_path],
        cwd=root,
        check=True,
    )

    model_path = os.path.join(root, "model.pt")
    cal_outputs_path = os.path.join(root, "cal_outputs.pt")

    assert os.path.isfile(model_path), "model.pt must exist after training dry-run"
    assert os.path.getsize(model_path) > 0, "model.pt must be non-empty"
    state_dict = torch.load(model_path, map_location="cpu")
    assert isinstance(state_dict, dict), "model.pt must contain a state dict dictionary"
    assert len(state_dict) > 0, "model.pt state dict must not be empty"

    assert os.path.isfile(cal_outputs_path), "cal_outputs.pt must exist after training dry-run"
    saved = torch.load(cal_outputs_path, map_location="cpu")
    assert isinstance(saved, dict), "cal_outputs.pt must contain a dictionary"
    assert "outputs" in saved, "cal_outputs missing outputs key"
    assert "labels" in saved, "cal_outputs missing labels key"
    assert "regimes" in saved, "cal_outputs missing regimes key"

    outputs = saved["outputs"]
    labels = saved["labels"]
    regimes = saved["regimes"]
    assert isinstance(outputs, list), "outputs must be a list"
    assert isinstance(labels, list), "labels must be a list"
    assert isinstance(regimes, list), "regimes must be a list"
    assert len(outputs) == len(labels), "outputs and labels lengths must match"
    assert len(labels) == len(regimes), "labels and regimes lengths must match"

    for output in outputs:
        assert isinstance(output, dict), "each output must be a dictionary"
        assert "mu" in output, "output missing mu"
        assert "log_sigma" in output, "output missing log_sigma"
        assert "introspective_score" in output, "output missing introspective_score"

    print("All training tests passed.")


if __name__ == "__main__":
    main()
