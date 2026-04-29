"""vLLM-backed forecaster that emits a range, explanation, and confidence."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


_PROMPT_VERSION = "v3-range"
_LOWER_REGEX = re.compile(
    r"lower(?:\s+bound)?[^:\n]*:\s*([+-]?[0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)
_UPPER_REGEX = re.compile(
    r"upper(?:\s+bound)?[^:\n]*:\s*([+-]?[0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)
_CONFIDENCE_REGEX = re.compile(r"confidence[^:\n]*:\s*([0-9]{1,3})", re.IGNORECASE)


class LLMExplainer:
    """Generate forecast range, explanation, and an explicit confidence score."""

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
        ticker = str(item.get("ticker", "")).upper() or "UNKNOWN"
        instruction = (
            "You are a financial analyst. Given an earnings call excerpt, "
            "predict a 90% confidence interval for the next-day stock return.\n\n"
            f"Ticker: {ticker}\n"
            f"Earnings call excerpt:\n{excerpt}\n\n"
            "Respond in EXACTLY this format and nothing else:\n"
            "Lower bound (percent): <number>\n"
            "Upper bound (percent): <number>\n"
            "Confidence (0-100): <integer>\n"
            "Explanation: <one sentence>"
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
                hashlib.sha1(transcript.encode("utf-8")).hexdigest()[:16],
            ]
        )
        return hashlib.sha1(material.encode("utf-8")).hexdigest()

    def _parse_response(self, generated_output: Any) -> dict[str, Any]:
        """Extract LLM range, explanation, and confidence."""

        text = str(generated_output.text)
        lower_match = _LOWER_REGEX.search(text)
        upper_match = _UPPER_REGEX.search(text)
        if lower_match is not None:
            try:
                q_low_llm = float(lower_match.group(1)) / 100.0
            except ValueError:
                q_low_llm = -0.05
        else:
            q_low_llm = -0.05
        if upper_match is not None:
            try:
                q_high_llm = float(upper_match.group(1)) / 100.0
            except ValueError:
                q_high_llm = 0.05
        else:
            q_high_llm = 0.05
        if q_high_llm < q_low_llm:
            q_low_llm, q_high_llm = q_high_llm, q_low_llm

        confidence_match = _CONFIDENCE_REGEX.search(text)
        if confidence_match is not None:
            confidence = float(confidence_match.group(1)) / 100.0
        else:
            confidence = 0.5
        confidence = min(max(confidence, 0.0), 1.0)

        return {
            "explanation": text,
            "confidence": float(confidence),
            "llm_q_low": float(q_low_llm),
            "llm_q_high": float(q_high_llm),
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

        required_fields = ("explanation", "confidence", "llm_q_low", "llm_q_high")
        for index, item in enumerate(items):
            key = self._cache_key(item)
            cached = self.cache.get(key)
            if cached is not None and all(field in cached for field in required_fields):
                results[index] = {
                    "explanation": str(cached["explanation"]),
                    "confidence": float(cached["confidence"]),
                    "llm_q_low": float(cached["llm_q_low"]),
                    "llm_q_high": float(cached["llm_q_high"]),
                }
                continue
            prompts_to_generate.append(self._build_prompt(item))
            pending.append((index, key))

        if prompts_to_generate:
            generations = self.llm.generate(prompts_to_generate, sampling_params=self.params)
            for (result_index, cache_key), generation in zip(pending, generations):
                gen = generation.outputs[0]
                parsed = self._parse_response(gen)
                results[result_index] = parsed
                self.cache[cache_key] = {
                    "explanation": parsed["explanation"],
                    "confidence": float(parsed["confidence"]),
                    "llm_q_low": float(parsed["llm_q_low"]),
                    "llm_q_high": float(parsed["llm_q_high"]),
                }
            self._save_cache()

        return [
            item if item is not None else {
                "explanation": "",
                "confidence": 0.5,
                "llm_q_low": -0.05,
                "llm_q_high": 0.05,
            }
            for item in results
        ]
