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

    outputs = _sample_outputs(90)
    labels = [random.uniform(-0.10, 0.10) for _ in range(90)]
    regimes = (
        (["low_surprise"] * 30)
        + (["medium_surprise"] * 30)
        + (["high_surprise"] * 30)
    )

    predictor.calibrate(
        cal_outputs=outputs,
        cal_labels=labels,
        cal_regimes=regimes,
    )

    required_keys = [
        ("low_surprise", 0.80),
        ("low_surprise", 0.90),
        ("low_surprise", 0.95),
        ("medium_surprise", 0.80),
        ("medium_surprise", 0.90),
        ("medium_surprise", 0.95),
        ("high_surprise", 0.80),
        ("high_surprise", 0.90),
        ("high_surprise", 0.95),
    ]
    for key in required_keys:
        assert key in predictor.thresholds, "Missing threshold key"
        assert predictor.thresholds[key] > 0.0, "Threshold must be positive"

    low_score_out = {"mu": 0.01, "log_sigma": 0.4, "introspective_score": 0.1}
    high_score_out = {"mu": 0.01, "log_sigma": 0.4, "introspective_score": 0.9}

    lo90, hi90 = predictor.predict_interval(
        low_score_out,
        regime="medium_surprise",
        coverage=0.90,
    )
    assert lo90 < hi90, "Interval bounds must satisfy lo < hi"

    lo_low, hi_low = predictor.predict_interval(
        low_score_out,
        regime="medium_surprise",
        coverage=0.90,
    )
    lo_high, hi_high = predictor.predict_interval(
        high_score_out,
        regime="medium_surprise",
        coverage=0.90,
    )
    width_low = hi_low - lo_low
    width_high = hi_high - lo_high
    assert width_low > width_high, "Low introspective score must widen interval"

    lo95, hi95 = predictor.predict_interval(
        high_score_out,
        regime="medium_surprise",
        coverage=0.95,
    )
    width95 = hi95 - lo95
    width90 = hi_high - lo_high
    assert width95 >= width90, "95% interval must be at least as wide as 90%"

    reject = predictor.selective_predict(
        {"mu": 1e-8, "log_sigma": 2.5, "introspective_score": 0.05},
        regime="medium_surprise",
        coverage=0.90,
    )
    assert reject is None, "selective_predict should reject uncertain near-zero case"

    assert assign_regime(0.2) == "low_surprise", "assign_regime low boundary failed"
    assert (
        assign_regime(1.0) == "medium_surprise"
    ), "assign_regime medium boundary failed"
    assert assign_regime(-2.0) == "high_surprise", "assign_regime high boundary failed"

    print("All conformal tests passed.")


if __name__ == "__main__":
    main()
