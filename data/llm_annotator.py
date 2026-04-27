"""LLM-based range predictor for earnings events.

Runs a local HuggingFace instruction-tuned model (e.g. Llama-3-8B-Instruct)
on the ARC cluster GPU to produce a calibrated return-range prediction,
confidence score, and explanation for each earnings event.
Results are cached to disk so each event is annotated only once.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch


_SYSTEM_PROMPT = (
    "You are a quantitative financial analyst specializing in earnings surprises "
    "and short-term stock price reactions. Given an earnings call transcript excerpt "
    "and key financial metrics, predict the likely range for the stock's next-day "
    "return after the earnings announcement.\n\n"
    "Respond with ONLY a valid JSON object — no markdown fences, no extra text:\n"
    '{"range_low": <float, lower bound in percent, e.g. -8.5>, '
    '"range_high": <float, upper bound in percent, e.g. 3.2>, '
    '"confidence": <float in [0,1], confidence that true return falls in this range>, '
    '"explanation": "<1-2 sentence explanation>"}\n\n'
    "The range should represent approximately an 80% credible interval. "
    "A narrower range signals higher conviction."
)


def _build_user_prompt(transcript: str, financial_features: list[float]) -> str:
    eps = financial_features[0] if len(financial_features) > 0 else 0.0
    rev = financial_features[1] if len(financial_features) > 1 else 0.0
    guid = financial_features[2] if len(financial_features) > 2 else 0.0
    excerpt = transcript[:4000]
    return (
        f"EPS Surprise (z-score): {eps:.4f}\n"
        f"Revenue Surprise (z-score): {rev:.4f}\n"
        f"Guidance Change (z-score): {guid:.4f}\n\n"
        f"Earnings Call Transcript (excerpt):\n{excerpt}"
    )


def _format_chat_prompt(model_name: str, system: str, user: str) -> str:
    """Format the system+user turn into the model's expected chat template string.

    We use the transformers tokenizer's apply_chat_template where possible, but
    provide a plain fallback for models that don't ship a template.
    """
    name = model_name.lower()
    if "llama" in name:
        return (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    if "mistral" in name or "mixtral" in name:
        return f"[INST] {system}\n\n{user} [/INST]"
    if "falcon" in name:
        return f"System: {system}\nUser: {user}\nAssistant:"
    # Generic instruct fallback
    return f"### System:\n{system}\n\n### User:\n{user}\n\n### Assistant:\n"


def _parse_llm_response(text: str) -> dict[str, Any]:
    """Extract and validate the JSON payload from a raw LLM response."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    # Some models prefix with explanation before the JSON — find the first `{`
    brace = text.find("{")
    if brace > 0:
        text = text[brace:]
    # Truncate after the closing `}`
    end = text.rfind("}")
    if end != -1:
        text = text[: end + 1]

    parsed = json.loads(text.strip())
    range_low = float(parsed.get("range_low", 0.0))
    range_high = float(parsed.get("range_high", 0.0))
    confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
    explanation = str(parsed.get("explanation", ""))
    if range_low > range_high:
        range_low, range_high = range_high, range_low
    return {
        "range_low": range_low,
        "range_high": range_high,
        "confidence": confidence,
        "explanation": explanation,
    }


def _derive_confidence(range_low: float, range_high: float, llm_confidence: float) -> float:
    """Blend LLM-stated confidence with a range-width signal.

    A wide range with claimed high confidence is penalised; the two signals
    are averaged so neither dominates.
    Width of ~10 pp → width_conf ≈ 0.5; 0 pp → 1.0; 20+ pp → ~0.2.
    """
    width_pct = abs(range_high - range_low)
    width_conf = 1.0 / (1.0 + width_pct / 10.0)
    return 0.5 * llm_confidence + 0.5 * width_conf


class LocalLLMClient:
    """Thin wrapper around a HuggingFace pipeline for single-event annotation.

    Loads the model once and reuses it across calls. Pass an instance of this
    to ``annotate_event`` / ``annotate_events`` in place of an API client.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device: str | None = None,
        max_new_tokens: int = 256,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if load_in_4bit:
            kwargs["load_in_4bit"] = True
        elif load_in_8bit:
            kwargs["load_in_8bit"] = True

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._pipeline = pipeline(
            "text-generation",
            model=hf_model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )

    def generate(self, prompt: str) -> str:
        """Run the model and return the generated text (excluding the prompt)."""
        result = self._pipeline(prompt)
        generated: str = result[0]["generated_text"]
        # Strip the input prompt from the output
        if generated.startswith(prompt):
            generated = generated[len(prompt):]
        return generated.strip()


def annotate_event(
    event: dict[str, Any],
    client: LocalLLMClient,
) -> dict[str, Any]:
    """Annotate a single earnings event with a locally-run LLM.

    Returns a dict with keys:
        llm_range_low    — predicted lower return bound (fraction, not percent)
        llm_range_high   — predicted upper return bound (fraction, not percent)
        llm_confidence   — blended confidence in [0, 1]
        llm_explanation  — natural-language explanation string
    """
    transcript = str(event.get("transcript", ""))
    raw_features = list(event.get("features", []))
    features: list[float] = []
    for value in raw_features[:3]:
        try:
            features.append(float(value))
        except Exception:
            features.append(0.0)
    while len(features) < 3:
        features.append(0.0)

    user_prompt = _build_user_prompt(transcript, features)
    full_prompt = _format_chat_prompt(client.model_name, _SYSTEM_PROMPT, user_prompt)
    raw_text = client.generate(full_prompt)
    parsed = _parse_llm_response(raw_text)
    confidence = _derive_confidence(parsed["range_low"], parsed["range_high"], parsed["confidence"])

    return {
        "llm_range_low": parsed["range_low"] / 100.0,
        "llm_range_high": parsed["range_high"] / 100.0,
        "llm_confidence": confidence,
        "llm_explanation": parsed["explanation"],
    }


def _event_cache_key(event: dict[str, Any]) -> str:
    ticker = str(event.get("ticker", "UNKNOWN")).upper().replace(".", "-")
    date = str(event.get("date", "0000-00-00"))
    return f"{ticker}_{date}"


def annotate_events(
    events: list[dict[str, Any]],
    cache_path: str | Path,
    client: LocalLLMClient,
) -> list[dict[str, Any]]:
    """Annotate a list of events, using a disk cache to skip already-done events.

    Newly annotated events are persisted after each call so that partial runs
    on ARC survive job preemptions.

    Args:
        events:     List of event dicts (must have 'ticker', 'date',
                    'transcript', 'features' keys).
        cache_path: Path to a ``.pt`` file used to store/restore annotations.
        client:     A ``LocalLLMClient`` instance already loaded on GPU.

    Returns:
        The input ``events`` list with each dict updated in-place to include
        ``llm_range_low``, ``llm_range_high``, ``llm_confidence``, and
        ``llm_explanation`` fields.
    """
    cache_path = Path(cache_path)
    cache: dict[str, dict[str, Any]] = {}

    if cache_path.is_file():
        try:
            loaded = torch.load(cache_path, map_location="cpu")
            if isinstance(loaded, dict):
                cache = loaded
        except Exception:
            cache = {}

    for i, event in enumerate(events):
        key = _event_cache_key(event)

        if key in cache:
            event.update(cache[key])
            continue

        try:
            annotation = annotate_event(event, client=client)
        except Exception as exc:
            print(f"[llm_annotator] Warning: failed to annotate {key}: {exc}")
            annotation = {
                "llm_range_low": 0.0,
                "llm_range_high": 0.0,
                "llm_confidence": 0.5,
                "llm_explanation": "",
            }

        cache[key] = annotation
        event.update(annotation)

        # Save after every event — survives Slurm preemption
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)

        if (i + 1) % 50 == 0:
            print(f"[llm_annotator] Annotated {i + 1}/{len(events)} events")

    return events


if __name__ == "__main__":
    import argparse
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    parser = argparse.ArgumentParser(
        description="Annotate events cache with local LLM range predictions."
    )
    parser.add_argument(
        "--events-cache",
        default=str(PROJECT_ROOT / "data" / "events_cache.pt"),
    )
    parser.add_argument(
        "--llm-cache",
        default=str(PROJECT_ROOT / "data" / "llm_annotations.pt"),
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model name or local path on ARC",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda / cpu (defaults to cuda if available)",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    args = parser.parse_args()

    raw = torch.load(args.events_cache, map_location="cpu")
    if not isinstance(raw, list):
        print("events_cache.pt must be a list of event dicts.")
        sys.exit(1)

    print(f"Loading model {args.model} ...")
    llm_client = LocalLLMClient(
        model_name=args.model,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    print(f"Annotating {len(raw)} events ...")
    annotate_events(raw, cache_path=args.llm_cache, client=llm_client)

    out_path = Path(args.events_cache).with_name("events_cache_llm.pt")
    torch.save(raw, out_path)
    print(f"Saved enriched cache to {out_path}")
