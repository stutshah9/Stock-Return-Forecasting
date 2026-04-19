from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from calibration.conformal import EventConditionedConformalPredictor, assign_regime


def _sample_outputs(n: int) -> list[dict[str, float]]:
    outputs: list[dict[str, float]] = []
    for _ in range(n):
        outputs.append(
            {
                "mu": random.uniform(-0.08, 0.08),
                "log_sigma": random.uniform(-1.5, 1.0),
                "introspective_score": random.uniform(0.05, 0.95),
            }
        )
    return outputs


def main() -> None:
    random.seed(42)

    predictor = EventConditionedConformalPredictor(coverage_levels=[0.80, 0.90, 0.95])
    explanation_predictor = EventConditionedConformalPredictor(
        coverage_levels=[0.80, 0.90, 0.95],
        use_explanation_adjustment=True,
    )

    outputs = _sample_outputs(90)
    labels = [random.uniform(-0.10, 0.10) for _ in range(90)]
    regimes = (
        (["low_surprise_low_vol"] * 15)
        + (["low_surprise_high_vol"] * 15)
        + (["medium_surprise_low_vol"] * 15)
        + (["medium_surprise_high_vol"] * 15)
        + (["high_surprise_low_vol"] * 15)
        + (["high_surprise_high_vol"] * 15)
    )
    metadata = []
    for index in range(90):
        metadata.append(
            {
                "message_volume": float(index % 30),
                "explanation_confidence": 0.15 if index % 2 == 0 else 0.85,
            }
        )

    predictor.calibrate(
        cal_outputs=outputs,
        cal_labels=labels,
        cal_regimes=regimes,
        cal_metadata=metadata,
    )
    explanation_predictor.calibrate(
        cal_outputs=outputs,
        cal_labels=labels,
        cal_regimes=regimes,
        cal_metadata=metadata,
    )

    required_keys = [
        ("low_surprise_low_vol", 0.80),
        ("low_surprise_low_vol", 0.90),
        ("low_surprise_low_vol", 0.95),
        ("low_surprise_high_vol", 0.80),
        ("low_surprise_high_vol", 0.90),
        ("low_surprise_high_vol", 0.95),
        ("medium_surprise_low_vol", 0.80),
        ("medium_surprise_low_vol", 0.90),
        ("medium_surprise_low_vol", 0.95),
        ("medium_surprise_high_vol", 0.80),
        ("medium_surprise_high_vol", 0.90),
        ("medium_surprise_high_vol", 0.95),
        ("high_surprise_low_vol", 0.80),
        ("high_surprise_low_vol", 0.90),
        ("high_surprise_low_vol", 0.95),
        ("high_surprise_high_vol", 0.80),
        ("high_surprise_high_vol", 0.90),
        ("high_surprise_high_vol", 0.95),
    ]
    for key in required_keys:
        assert key in predictor.thresholds, "Missing threshold key"
        assert predictor.thresholds[key] > 0.0, "Threshold must be positive"

    low_score_out = {"mu": 0.01, "log_sigma": 0.4, "introspective_score": 0.1}
    high_score_out = {"mu": 0.01, "log_sigma": 0.4, "introspective_score": 0.9}

    lo90, hi90 = predictor.predict_interval(
        low_score_out,
        regime="medium_surprise_low_vol",
        coverage=0.90,
    )
    assert lo90 < hi90, "Interval bounds must satisfy lo < hi"

    lo_low, hi_low = predictor.predict_interval(
        low_score_out,
        regime="medium_surprise_low_vol",
        coverage=0.90,
        metadata={"message_volume": 1.0, "explanation_confidence": 0.10},
    )
    lo_high, hi_high = predictor.predict_interval(
        high_score_out,
        regime="medium_surprise_low_vol",
        coverage=0.90,
        metadata={"message_volume": 1.0, "explanation_confidence": 0.90},
    )
    width_low = hi_low - lo_low
    width_high = hi_high - lo_high
    assert width_low > 0.0 and width_high > 0.0, "Interval widths must be positive"
    diagnostics_low = predictor.interval_diagnostics(
        low_score_out,
        regime="medium_surprise_low_vol",
        coverage=0.90,
        metadata={"message_volume": 1.0, "explanation_confidence": 0.10},
    )
    diagnostics_high = predictor.interval_diagnostics(
        high_score_out,
        regime="medium_surprise_low_vol",
        coverage=0.90,
        metadata={"message_volume": 1.0, "explanation_confidence": 0.90},
    )
    assert "combined_threshold" in diagnostics_low, "Missing combined conformal threshold"
    assert "score_band" in diagnostics_low, "Missing score band diagnostics"
    assert "attention_band" in diagnostics_low, "Missing attention band diagnostics"
    assert diagnostics_low["attention_source"] in {
        "regime_only",
        "regime_attention",
        "global_attention_fallback",
    }, "Missing observable attention conditioning source"
    assert (
        diagnostics_low["combined_threshold"] == diagnostics_high["combined_threshold"]
    ), "Default event-conditioned calibration should not change interval width solely from explanation score"

    exp_lo_low, exp_hi_low = explanation_predictor.predict_interval(
        low_score_out,
        regime="medium_surprise_low_vol",
        coverage=0.90,
        metadata={"message_volume": 1.0, "explanation_confidence": 0.10},
    )
    exp_lo_high, exp_hi_high = explanation_predictor.predict_interval(
        high_score_out,
        regime="medium_surprise_low_vol",
        coverage=0.90,
        metadata={"message_volume": 1.0, "explanation_confidence": 0.90},
    )
    assert (
        (exp_hi_low - exp_lo_low) >= (exp_hi_high - exp_lo_high)
    ), "Explanation-augmented ablation should widen low-confidence intervals when enabled"

    lo95, hi95 = predictor.predict_interval(
        high_score_out,
        regime="medium_surprise_low_vol",
        coverage=0.95,
    )
    width95 = hi95 - lo95
    width90 = hi_high - lo_high
    assert width95 >= width90, "95% interval must be at least as wide as 90%"

    reject = predictor.selective_predict(
        {"mu": 1e-8, "log_sigma": 2.5, "introspective_score": 0.05},
        regime="medium_surprise_low_vol",
        coverage=0.90,
    )
    assert reject is None, "selective_predict should reject uncertain near-zero case"

    assert assign_regime(0.2, 0.20) == "low_surprise_low_vol", "assign_regime low/low failed"
    assert (
        assign_regime(1.0, 0.35) == "medium_surprise_high_vol"
    ), "assign_regime medium/high failed"
    assert assign_regime(-2.0, 0.10) == "high_surprise_low_vol", "assign_regime high/low failed"

    print("All conformal tests passed.")


if __name__ == "__main__":
    main()
