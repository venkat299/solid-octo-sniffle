"""Embedding provider implementations for the job role analyzer."""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:  # pragma: no cover - executed when the package is missing
    SentenceTransformer = None  # type: ignore[assignment]

from .config import load_config
from .similarity import EmbeddingProvider


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by SentenceTransformer models."""

    def __init__(self, model_name: str | None = None, *, device: str | None = None) -> None:
        config = load_config()
        self._model_name = model_name or config.embedding_model
        self._device = device
        if SentenceTransformer is None:
            raise ModuleNotFoundError(
                "sentence-transformers is required for SentenceTransformerEmbeddingProvider."
            )
        self._model = _load_model(self._model_name, self._device)

    def embed(self, text: str) -> Sequence[float]:
        if not text:
            return []
        embedding = self._model.encode(  # type: ignore[operator]
            [text],
            normalize_embeddings=False,
            convert_to_numpy=False,
        )[0]
        return list(map(float, embedding))


@lru_cache(maxsize=4)
def _load_model(model_name: str, device: str | None) -> "SentenceTransformer":
    if SentenceTransformer is None:  # pragma: no cover - guarded above
        raise ModuleNotFoundError(
            "sentence-transformers is required to load embedding models."
        )
    return SentenceTransformer(model_name, device=device)
