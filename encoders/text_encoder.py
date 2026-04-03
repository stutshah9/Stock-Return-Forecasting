"""Transcript encoder built on top of FinBERT."""

from __future__ import annotations

import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer
import yaml


def _load_encoder_config() -> dict[str, Any]:
    """Load encoder-related defaults from the project configuration file."""

    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
    except OSError:
        config = {}
    return config


_ENCODER_CONFIG = _load_encoder_config()
_DEFAULT_EMBED_DIM = int(_ENCODER_CONFIG.get("model", {}).get("embed_dim", 64))
_DEFAULT_CHUNK_SIZE = int(_ENCODER_CONFIG.get("data", {}).get("chunk_size", 256))
_DEFAULT_MAX_CHUNKS = int(_ENCODER_CONFIG.get("data", {}).get("max_chunks", 16))


class TranscriptEncoder(nn.Module):
    """Encode transcript text into a fixed-size embedding with FinBERT."""

    def __init__(
        self,
        embed_dim: int = _DEFAULT_EMBED_DIM,
        frozen: bool = True,
        cache_dir: str | Path | None = None,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        max_chunks: int = _DEFAULT_MAX_CHUNKS,
    ) -> None:
        """Initialize the transcript encoder.

        Args:
            embed_dim: Output embedding size for the projected transcript vector.
            frozen: Whether to freeze the FinBERT backbone parameters.
            cache_dir: Optional directory for cached transcript chunk embeddings.
            chunk_size: Number of tokenizer tokens per transcript chunk.
            max_chunks: Maximum number of chunks to encode per transcript.
        """

        super().__init__()
        self.embed_dim = embed_dim
        self.frozen = frozen
        self.chunk_size = chunk_size
        self.chunk_overlap = 32
        self.max_chunks = max_chunks
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.cache_keys: list[str] | None = None

        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.backbone = AutoModel.from_pretrained("ProsusAI/finbert")
        self.projection = nn.Linear(self.backbone.config.hidden_size, embed_dim)

        if self.frozen:
            for parameter in self.parameters():
                parameter.requires_grad = False

            # Unfreeze last 2 encoder layers for domain adaptation
            for name, param in self.bert.named_parameters():
                if "encoder.layer.11" in name or "encoder.layer.10" in name:
                    param.requires_grad = True

            if not any(param.requires_grad for param in self.bert.parameters()):
                self.backbone.eval()

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def bert(self) -> nn.Module:
        """Compatibility alias for the FinBERT backbone module."""

        return self.backbone

    def train(self, mode: bool = True) -> TranscriptEncoder:
        """Set training mode while keeping a frozen backbone in eval mode."""

        super().train(mode)
        if self.frozen and not any(param.requires_grad for param in self.bert.parameters()):
            self.backbone.eval()
        return self

    def set_cache_keys(self, cache_keys: list[str] | None) -> None:
        """Set cache keys such as ``TICKER_YYYY-MM-DD`` for the next forward pass."""

        self.cache_keys = cache_keys

    def _sanitize_cache_key(self, cache_key: str) -> str:
        """Sanitize a cache key so it is safe to use in a file name."""

        return re.sub(r"[^A-Za-z0-9_.-]+", "_", cache_key).strip("_")

    def _cache_path(self, index: int) -> Path | None:
        """Return the cache path for a transcript index, if caching is enabled."""

        if self.cache_dir is None or any(param.requires_grad for param in self.bert.parameters()):
            return None
        if self.cache_keys is None or index >= len(self.cache_keys):
            return None
        safe_key = self._sanitize_cache_key(self.cache_keys[index])
        if not safe_key:
            return None
        return self.cache_dir / f"{safe_key}.pt"

    def _chunk_token_ids(self, token_ids: list[int]) -> list[list[int]]:
        """Split token ids into overlapping sliding-window chunks."""

        if not token_ids:
            return [[]]

        step = max(self.chunk_size - self.chunk_overlap, 1)
        chunks: list[list[int]] = []
        for start_index in range(0, len(token_ids), step):
            chunk = token_ids[start_index : start_index + self.chunk_size]
            if not chunk:
                break
            chunks.append(chunk)
            if len(chunks) >= self.max_chunks:
                break
            if start_index + self.chunk_size >= len(token_ids):
                break
        return chunks

    def _prepare_chunk_batch(self, chunk_token_ids: list[list[int]]) -> dict[str, Tensor]:
        """Convert transcript chunks into a padded batch for FinBERT."""

        encoded_chunks = [
            self.tokenizer.prepare_for_model(
                token_ids,
                add_special_tokens=True,
                truncation=True,
                return_attention_mask=True,
            )
            for token_ids in chunk_token_ids
        ]
        batch = self.tokenizer.pad(encoded_chunks, return_tensors="pt")
        device = self.projection.weight.device
        return {name: tensor.to(device) for name, tensor in batch.items()}

    def _mean_pool(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Mean-pool token embeddings with an attention mask."""

        expanded_mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        masked_hidden_states = hidden_states * expanded_mask
        token_counts = expanded_mask.sum(dim=1).clamp(min=1.0)
        return masked_hidden_states.sum(dim=1) / token_counts

    def _load_cached_chunk_embeddings(self, cache_path: Path | None) -> Tensor | None:
        """Load cached chunk embeddings from disk when available."""

        if cache_path is None or not cache_path.is_file():
            return None
        try:
            cached_embeddings = torch.load(cache_path, map_location="cpu")
        except Exception:
            return None
        if not isinstance(cached_embeddings, Tensor) or cached_embeddings.dim() != 2:
            return None
        return cached_embeddings

    def _save_cached_chunk_embeddings(
        self,
        cache_path: Path | None,
        chunk_embeddings: Tensor,
    ) -> None:
        """Persist chunk embeddings to disk for future frozen encoder passes."""

        if cache_path is None:
            return
        try:
            torch.save(chunk_embeddings.detach().cpu(), cache_path)
        except OSError:
            return

    def _compute_chunk_embeddings(self, text: str) -> Tensor:
        """Encode one transcript into a tensor of chunk embeddings."""

        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunk_token_ids = self._chunk_token_ids(token_ids)
        model_inputs = self._prepare_chunk_batch(chunk_token_ids)

        context_manager = (
            torch.no_grad()
            if not any(param.requires_grad for param in self.bert.parameters())
            else nullcontext()
        )
        with context_manager:
            outputs = self.backbone(**model_inputs)

        return self._mean_pool(outputs.last_hidden_state, model_inputs["attention_mask"])

    def forward(self, texts: list[str]) -> Tensor:
        """Encode a batch of transcripts into shape ``[batch, embed_dim]``.

        Args:
            texts: Raw transcript strings.

        Returns:
            A tensor of projected transcript embeddings with shape
            ``[batch, embed_dim]``.
        """

        device = self.projection.weight.device
        hidden_size = self.backbone.config.hidden_size

        if not texts:
            return torch.empty((0, self.embed_dim), device=device)

        transcript_embeddings: list[Tensor] = []
        for index, text in enumerate(texts):
            cache_path = self._cache_path(index)
            chunk_embeddings = self._load_cached_chunk_embeddings(cache_path)
            if chunk_embeddings is None:
                chunk_embeddings = self._compute_chunk_embeddings(text)
                self._save_cached_chunk_embeddings(cache_path, chunk_embeddings)

            chunk_embeddings = chunk_embeddings.to(device)
            if chunk_embeddings.numel() == 0:
                transcript_embedding = torch.zeros(hidden_size, device=device)
            else:
                transcript_embedding = chunk_embeddings.mean(dim=0)
            transcript_embeddings.append(transcript_embedding)

        stacked_embeddings = torch.stack(transcript_embeddings, dim=0)
        return self.projection(stacked_embeddings)
