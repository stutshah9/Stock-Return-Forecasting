"""vLLM-backed forecast explainer that emits a numeric confidence and |return| estimate."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


_PROMPT_VERSION = "v2-numeric"
_CONFIDENCE_REGEX = re.compile(r"confidence[^:\n]*:\s*([0-9]{1,3})", re.IGNORECASE)
_ABS_RETURN_REGEX = re.compile(
    r"(?:absolute|expected|predicted)?\s*return[^:\n]*:\s*([+-]?[0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)
_DIGIT_TOKENS = {str(d) for d in range(10)}


class LLMExplainer:
    """Generate forecast explanations, numeric confidence, and predicted |return|."""

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        cache_path: str | Path | None = None,
        max_new_tokens: int = 128,
        top_logprobs: int = 10,
        gpu_memory_utilization: float = 0.85,
        quantization: str | None = "awq",
        tensor_parallel_size: int = 1,
        transcript_chars: int = 1500,
    ) -> None:
        from vllm import LLM, SamplingParams

        self.model_name = model_name
        self.transcript_chars = int(transcript_chars)
        self.params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            logprobs=int(top_logprobs),
            max_tokens=int(max_new_tokens),
        )
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=int(tensor_parallel_size),
            quantization=quantization,
            gpu_memory_utilization=float(gpu_memory_utilization),
        )
        self.tokenizer = self.llm.get_tokenizer()

        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.cache: dict[str, dict[str, Any]] = {}
        if self.cache_path is not None and self.cache_path.is_file():
            try:
                with self.cache_path.open("r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    self.cache = loaded
            except (OSError, json.JSONDecodeError):
                self.cache = {}

    def _build_prompt(self, item: dict[str, Any]) -> str:
        transcript = str(item.get("transcript") or "").strip()
        excerpt = transcript[: self.transcript_chars] if transcript else "(no transcript available)"
        mu = float(item["mu"])
        q_low = float(item["q_low"])
        q_high = float(item["q_high"])
        ticker = str(item.get("ticker", "")).upper() or "UNKNOWN"
        instruction = (
            "You are a financial analyst. Given an earnings call excerpt and a "
            "quantitative point forecast for the next-day stock return, answer "
            "three things: (1) explain the forecast in one sentence, (2) rate "
            "your confidence in the forecast on an integer scale from 0 to 100, "
            "and (3) predict the absolute (unsigned) magnitude of the next-day "
            "return as a percent of the current price.\n\n"
            f"Ticker: {ticker}\n"
            f"Earnings call excerpt:\n{excerpt}\n\n"
            f"Quantitative forecast: mu={mu:+.4f}, 90% interval=[{q_low:+.4f}, {q_high:+.4f}].\n\n"
            "Respond in EXACTLY this format and nothing else:\n"
            "Explanation: <one sentence>\n"
            "Confidence (0-100): <integer>\n"
            "Expected absolute return (percent): <number>"
        )
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return f"[INST] {instruction} [/INST]"

    @staticmethod
    def _cache_key(item: dict[str, Any]) -> str:
        transcript = str(item.get("transcript") or "")
        material = "|".join(
            [
                _PROMPT_VERSION,
                str(item.get("ticker", "")).upper(),
                str(item.get("date", "")),
                str(item.get("method", "")),
                f"{float(item['mu']):+.6f}",
                f"{float(item['q_low']):+.6f}",
                f"{float(item['q_high']):+.6f}",
                hashlib.sha1(transcript.encode("utf-8")).hexdigest()[:16],
            ]
        )
        return hashlib.sha1(material.encode("utf-8")).hexdigest()

    def _decode_token(self, token_id: int) -> str:
        return str(self.tokenizer.decode([int(token_id)]))

    def _soft_confidence_from_logprobs(self, generated_output: Any) -> float | None:
        """Compute soft confidence by reading digit logprobs of the Confidence number.

        Walks output tokens, finds where the model emitted "Confidence" and the
        digits that followed, and returns sum_d (digit_value * P(digit)) over
        the top-k logprob alternatives at each digit position. Returns None if
        the structure can't be matched (caller falls back to greedy parse).
        """

        token_ids = getattr(generated_output, "token_ids", None)
        token_logprobs = getattr(generated_output, "logprobs", None)
        if token_ids is None or token_logprobs is None:
            return None

        text_so_far = ""
        past_label = False
        per_digit_softs: list[float] = []
        for token_id, lp_dict in zip(token_ids, token_logprobs):
            decoded = self._decode_token(int(token_id))
            text_so_far += decoded

            if not past_label:
                if "confidence" in text_so_far.lower():
                    if ":" in text_so_far:
                        text_so_far = text_so_far.split(":", 1)[1]
                        past_label = True
                continue

            stripped = decoded.strip()
            if stripped == "" or not any(ch.isdigit() for ch in stripped):
                if per_digit_softs:
                    break
                continue

            if not (stripped.isdigit() and len(stripped) == 1) or lp_dict is None:
                continue

            digit_weights: dict[int, float] = {}
            for alt_id, alt_lp in lp_dict.items():
                alt_decoded = self._decode_token(int(alt_id)).strip()
                if len(alt_decoded) == 1 and alt_decoded.isdigit():
                    digit_weights[int(alt_decoded)] = math.exp(float(alt_lp.logprob))
            if not digit_weights:
                per_digit_softs.append(float(int(stripped)))
            else:
                total = sum(digit_weights.values())
                per_digit_softs.append(
                    sum(digit * (weight / total) for digit, weight in digit_weights.items())
                )
            if len(per_digit_softs) >= 3:
                break

        if not per_digit_softs:
            return None
        digits = per_digit_softs
        soft_value = 0.0
        place = 10 ** (len(digits) - 1)
        for digit_soft in digits:
            soft_value += digit_soft * place
            place //= 10
        return float(min(max(soft_value / 100.0, 0.0), 1.0))

    def _parse_response(self, generated_output: Any) -> dict[str, Any]:
        """Parse explanation, confidence (0-1), predicted abs return (decimal)."""

        text = str(generated_output.text)
        confidence_match = _CONFIDENCE_REGEX.search(text)
        if confidence_match is not None:
            greedy_confidence = float(confidence_match.group(1)) / 100.0
            greedy_confidence = min(max(greedy_confidence, 0.0), 1.0)
        else:
            greedy_confidence = 0.5

        soft = self._soft_confidence_from_logprobs(generated_output)
        confidence = soft if soft is not None else greedy_confidence

        return_match = _ABS_RETURN_REGEX.search(text)
        if return_match is not None:
            try:
                pct_value = float(return_match.group(1))
            except ValueError:
                pct_value = 2.0
            abs_return = pct_value / 100.0
        else:
            abs_return = 0.02
        abs_return = min(max(abs_return, 1e-3), 1.0)

        return {
            "explanation": text,
            "confidence": float(confidence),
            "greedy_confidence": float(greedy_confidence),
            "abs_return": float(abs_return),
        }

    def _save_cache(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(self.cache, fh)
        tmp_path.replace(self.cache_path)

    def explain_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any] | None] = [None] * len(items)
        prompts_to_generate: list[str] = []
        pending: list[tuple[int, str]] = []

        for index, item in enumerate(items):
            key = self._cache_key(item)
            cached = self.cache.get(key)
            if (
                cached is not None
                and "explanation" in cached
                and "confidence" in cached
                and "abs_return" in cached
            ):
                results[index] = {
                    "explanation": str(cached["explanation"]),
                    "confidence": float(cached["confidence"]),
                    "abs_return": float(cached["abs_return"]),
                }
                continue
            prompts_to_generate.append(self._build_prompt(item))
            pending.append((index, key))

        if prompts_to_generate:
            generations = self.llm.generate(prompts_to_generate, sampling_params=self.params)
            for (result_index, cache_key), generation in zip(pending, generations):
                gen = generation.outputs[0]
                parsed = self._parse_response(gen)
                results[result_index] = {
                    "explanation": parsed["explanation"],
                    "confidence": parsed["confidence"],
                    "abs_return": parsed["abs_return"],
                }
                self.cache[cache_key] = {
                    "explanation": parsed["explanation"],
                    "confidence": float(parsed["confidence"]),
                    "abs_return": float(parsed["abs_return"]),
                    "greedy_confidence": float(parsed.get("greedy_confidence", parsed["confidence"])),
                }
            self._save_cache()

        return [item if item is not None else {"explanation": "", "confidence": 0.5, "abs_return": 0.02} for item in results]
