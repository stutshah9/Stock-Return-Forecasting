import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def main() -> None:
    root = os.path.dirname(os.path.dirname(__file__))

    required_dirs = [
        "calibration",
        "data",
        "encoders",
        "experiments",
        "models",
        "tests",
    ]
    required_files = [
        "config.yaml",
        "requirements.txt",
        os.path.join("calibration", "__init__.py"),
        os.path.join("calibration", "conformal.py"),
        os.path.join("data", "__init__.py"),
        os.path.join("data", "dataset.py"),
        os.path.join("data", "financials.csv"),
        os.path.join("data", "loader.py"),
        os.path.join("data", "transcripts.csv"),
        os.path.join("encoders", "__init__.py"),
        os.path.join("encoders", "financial_encoder.py"),
        os.path.join("encoders", "sentiment_encoder.py"),
        os.path.join("encoders", "text_encoder.py"),
        os.path.join("experiments", "evaluate.py"),
        os.path.join("experiments", "train.py"),
        os.path.join("models", "__init__.py"),
        os.path.join("models", "fusion_model.py"),
    ]

    for directory in required_dirs:
        path = os.path.join(root, directory)
        assert os.path.isdir(path), "Required directory is missing"

    for file_path in required_files:
        path = os.path.join(root, file_path)
        assert os.path.isfile(path), "Required file is missing"

    config_path = os.path.join(root, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    assert isinstance(cfg, dict), "config.yaml must parse to a dictionary"
    assert "data" in cfg, "config missing data section"
    assert "chunk_size" in cfg["data"], "config missing data.chunk_size"
    assert "model" in cfg, "config missing model section"
    assert "embed_dim" in cfg["model"], "config missing model.embed_dim"
    assert "training" in cfg, "config missing training section"
    assert "lr" in cfg["training"], "config missing training.lr"
    assert "calibration" in cfg, "config missing calibration section"
    assert (
        "coverage_levels" in cfg["calibration"]
    ), "config missing calibration.coverage_levels"

    print("All scaffold tests passed.")


if __name__ == "__main__":
    main()
