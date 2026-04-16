import math
import os
import sys
import warnings

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from encoders.financial_encoder import FinancialEncoder
from encoders.sentiment_encoder import SentimentEncoder, aggregate_posts
from encoders.text_encoder import TranscriptEncoder


def main() -> None:
    root = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(root, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    embed_dim = int(cfg["model"]["embed_dim"])

    # TranscriptEncoder tests
    transcript_encoder = TranscriptEncoder(embed_dim=embed_dim)
    transcript_batch = [
        "Revenue grew and margins expanded this quarter.",
        "Guidance was revised due to macro headwinds.",
    ]
    transcript_out = transcript_encoder(transcript_batch)
    assert isinstance(transcript_out, torch.Tensor), "Transcript output must be a Tensor"
    assert transcript_out.shape == (2, embed_dim), "Transcript output shape mismatch"
    assert not torch.isnan(transcript_out).any().item(), "Transcript output contains NaN"

    frozen_encoder = TranscriptEncoder(embed_dim=embed_dim, frozen=True)
    for param in frozen_encoder.backbone.parameters():
        assert not param.requires_grad, "Frozen transcript backbone params must be non-trainable"
    assert any(
        param.requires_grad for param in frozen_encoder.projection.parameters()
    ), "Transcript projection should remain trainable for the fixed-embedding baseline"
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        token_ids = frozen_encoder._tokenize_transcript("hello " * 2000)
    assert len(token_ids) > 512, "Long transcript tokenization regression test is malformed"
    assert not any(
        "Token indices sequence length is longer than the specified maximum sequence length"
        in str(warning.message)
        for warning in captured_warnings
    ), "Long transcript preprocessing should not emit a misleading max-length warning"

    # FinancialEncoder tests
    financial_encoder = FinancialEncoder(embed_dim=embed_dim)
    financial_in = torch.tensor([[0.2, -0.1, 0.3], [0.4, 0.0, -0.2]], dtype=torch.float32)
    financial_out = financial_encoder(financial_in)
    assert isinstance(financial_out, torch.Tensor), "Financial output must be a Tensor"
    assert financial_out.shape == (2, embed_dim), "Financial output shape mismatch"
    assert financial_encoder.feature_names() == [
        "sue",
        "momentum",
        "implied_vol",
    ], "Financial feature_names mismatch"

    # SentimentEncoder + aggregate_posts tests
    posts_batch = [
        ["AAPL beat estimates", "Strong iPhone demand"],
        ["Concerns about macro demand", "Valuation still high"],
    ]
    agg = aggregate_posts(posts_batch)
    assert isinstance(agg, torch.Tensor), "aggregate_posts must return a Tensor"
    assert agg.shape == (2,), "aggregate_posts output shape must be (2,)"
    assert not any(math.isnan(float(x)) for x in agg.tolist()), "aggregate_posts contains NaN"

    sentiment_encoder = SentimentEncoder(embed_dim=embed_dim)
    sentiment_in = torch.tensor([[0.1, -0.2], [0.0, 0.3]], dtype=torch.float32)
    sentiment_out = sentiment_encoder(sentiment_in)
    assert isinstance(sentiment_out, torch.Tensor), "Sentiment output must be a Tensor"
    assert sentiment_out.shape == (2, embed_dim), "Sentiment output shape mismatch"

    assert transcript_out.shape == financial_out.shape, "Transcript and Financial shapes must match"
    assert financial_out.shape == sentiment_out.shape, "Financial and Sentiment shapes must match"

    print("All encoder tests passed.")


if __name__ == "__main__":
    main()
