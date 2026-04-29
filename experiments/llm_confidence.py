"""LLM-style explanation confidence adapter for calibration experiments.

The project can run this adapter without network access. When a real LLM
provider is wired in later, keep the returned fields stable so evaluation can
compare stated confidence against rating-token confidence on the same split.
"""

from __future__ import annotations

import math
from typing import Any


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(result):
        return float(default)
    return result


def _prediction_direction(mu: float) -> str:
    return "positive" if mu >= 0.0 else "negative"


def build_confidence_prompt(
    event: dict[str, Any],
    output: dict[str, Any],
    metadata: dict[str, float],
) -> str:
    """Build the prompt shape used for a future provider-backed LLM call."""

    ticker = str(event.get("ticker", "")).upper()
    date = str(event.get("date", ""))
    mu = _safe_float(output.get("point_mu", output.get("mu", 0.0)))
    interval_width = _safe_float(output.get("q_high", mu)) - _safe_float(output.get("q_low", mu))
    return (
        "Explain this earnings-return forecast and state one confidence score from 0 to 100. "
        f"Ticker: {ticker}. Date: {date}. Forecast direction: {_prediction_direction(mu)}. "
        f"Predicted return: {mu:.4f}. Interval width: {max(interval_width, 0.0):.4f}. "
        f"Model confidence: {_safe_float(output.get('introspective_score', 0.5)):.3f}. "
        f"Modality disagreement: {_safe_float(metadata.get('modality_disagreement', 0.0)):.4f}."
    )


def offline_llm_confidence(
    event: dict[str, Any],
    output: dict[str, Any],
    metadata: dict[str, float],
) -> dict[str, float | str]:
    """Return deterministic LLM-style explanation and confidence diagnostics.

    The stated score is intentionally derived from observable uncertainty cues,
    not the ground-truth label. The logprob confidence is a calibrated proxy for
    how likely the rating token would be under a concise explanation prompt.
    """

    mu = _safe_float(output.get("point_mu", output.get("mu", 0.0)))
    lower = _safe_float(output.get("q_low", mu))
    upper = _safe_float(output.get("q_high", mu))
    interval_width = max(upper - lower, 1e-6)
    model_confidence = _clamp01(_safe_float(output.get("introspective_score", 0.5), 0.5))
    variance_confidence = _clamp01(_safe_float(output.get("variance_confidence", 0.5), 0.5))
    disagreement_confidence = _clamp01(_safe_float(metadata.get("disagreement_confidence", 0.5), 0.5))
    modality_consistency = _clamp01(_safe_float(output.get("modality_consistency", 0.5), 0.5))

    stated_confidence = _clamp01(
        0.35 * variance_confidence
        + 0.25 * disagreement_confidence
        + 0.25 * model_confidence
        + 0.15 * modality_consistency
    )
    width_penalty = 1.0 / (1.0 + 12.0 * interval_width)
    logprob_confidence = _clamp01(0.65 * stated_confidence + 0.35 * width_penalty)
    token_logprob = math.log(max(logprob_confidence, 1e-6))

    direction = _prediction_direction(mu)
    ticker = str(event.get("ticker", "")).upper()
    explanation = (
        f"The forecast is {direction} for {ticker}, with confidence shaped by interval width, "
        "cross-modal agreement, and the model's own uncertainty diagnostics."
    )
    return {
        "llm_explanation": explanation,
        "llm_stated_confidence": float(stated_confidence),
        "llm_rating_token_logprob": float(token_logprob),
        "llm_logprob_confidence": float(logprob_confidence),
    }


def attach_llm_confidence(
    events: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    metadata_rows: list[dict[str, float]],
) -> list[dict[str, float]]:
    """Attach LLM-style confidence fields to metadata rows."""

    if not (len(events) == len(outputs) == len(metadata_rows)):
        raise ValueError("LLM confidence inputs must have matching lengths.")

    enriched_rows: list[dict[str, float]] = []
    for event, output, metadata in zip(events, outputs, metadata_rows):
        llm_fields = offline_llm_confidence(event, output, metadata)
        enriched = dict(metadata)
        enriched["llm_stated_confidence"] = float(llm_fields["llm_stated_confidence"])
        enriched["llm_rating_token_logprob"] = float(llm_fields["llm_rating_token_logprob"])
        enriched["llm_logprob_confidence"] = float(llm_fields["llm_logprob_confidence"])
        enriched["llm_explanation"] = str(llm_fields["llm_explanation"])
        enriched_rows.append(enriched)
    return enriched_rows
