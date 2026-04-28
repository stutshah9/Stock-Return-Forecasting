"""vLLM-backed forecast explainer that emits a calibrated confidence score."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


_CONFIDENCE_LABELS = ("HIGH", "MEDIUM", "LOW")
_LABEL_SCORE = {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.0}
_CONFIDENCE_REGEX = re.compile(r"Confidence:\s*(HIGH|MEDIUM|LOW)", re.IGNORECASE)


class LLMExplainer:
    """Generate per-event forecast explanations and confidence via vLLM."""

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        cache_path: str | Path | None = None,
        max_new_tokens: int = 96,
        top_logprobs: int = 5,
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
            "quantitative forecast for the next-day stock return, write one sentence "
            "explaining the forecast, then on a new line state your confidence as "
            "exactly one of HIGH, MEDIUM, or LOW.\n\n"
            f"Ticker: {ticker}\n"
            f"Earnings call excerpt:\n{excerpt}\n\n"
            f"Forecast: mu={mu:+.4f}, 90% interval=[{q_low:+.4f}, {q_high:+.4f}].\n\n"
            "Respond exactly in this format and nothing else:\n"
            "Explanation: <one sentence>\n"
            "Confidence: <HIGH|MEDIUM|LOW>"
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

    def _decode_token(self, token_id: int, logprob_obj: Any) -> str:
        decoded = getattr(logprob_obj, "decoded_token", None)
        if decoded is None:
            decoded = self.tokenizer.decode([int(token_id)])
        return str(decoded).strip().upper()

    def _extract_confidence(self, generated_output: Any) -> tuple[str, float]:
        text = str(generated_output.text)
        match = _CONFIDENCE_REGEX.search(text)
        if match is None:
            return text, 0.5
        hard_label = match.group(1).upper()
        hard_score = _LABEL_SCORE[hard_label]

        token_logprobs = getattr(generated_output, "logprobs", None)
        token_ids = getattr(generated_output, "token_ids", None)
        if token_logprobs is None or token_ids is None:
            return text, hard_score

        for token_id, lp_dict in zip(token_ids, token_logprobs):
            if lp_dict is None:
                continue
            chosen = self._decode_token(token_id, lp_dict.get(int(token_id)) if hasattr(lp_dict, "get") else None)
            if chosen not in _CONFIDENCE_LABELS:
                continue

            weights: dict[str, float] = {}
            for alt_id, alt_lp in lp_dict.items():
                alt_decoded = self._decode_token(int(alt_id), alt_lp)
                if alt_decoded in _CONFIDENCE_LABELS:
                    weights[alt_decoded] = math.exp(float(alt_lp.logprob))

            if not weights:
                return text, hard_score

            total = sum(weights.values())
            soft = sum(
                weights.get(label, 0.0) * _LABEL_SCORE[label] for label in _CONFIDENCE_LABELS
            ) / max(total, 1e-9)
            return text, float(min(max(soft, 0.0), 1.0))

        return text, hard_score

    def _save_cache(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(self.cache, fh)
        tmp_path.replace(self.cache_path)

    def explain_batch(self, items: list[dict[str, Any]]) -> list[tuple[str, float]]:
        results: list[tuple[str, float] | None] = [None] * len(items)
        prompts_to_generate: list[str] = []
        pending: list[tuple[int, str]] = []

        for index, item in enumerate(items):
            key = self._cache_key(item)
            cached = self.cache.get(key)
            if cached is not None and "explanation" in cached and "confidence" in cached:
                results[index] = (str(cached["explanation"]), float(cached["confidence"]))
                continue
            prompts_to_generate.append(self._build_prompt(item))
            pending.append((index, key))

        if prompts_to_generate:
            generations = self.llm.generate(prompts_to_generate, sampling_params=self.params)
            for (result_index, cache_key), generation in zip(pending, generations):
                gen = generation.outputs[0]
                explanation, confidence = self._extract_confidence(gen)
                results[result_index] = (explanation, confidence)
                self.cache[cache_key] = {
                    "explanation": explanation,
                    "confidence": float(confidence),
                }
            self._save_cache()

        return [item if item is not None else ("", 0.5) for item in results]
