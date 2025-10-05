"""Fallback utilities used when optional dependencies are unavailable."""
from __future__ import annotations

import hashlib
from typing import Sequence

from job_role_analyzer.similarity import EmbeddingProvider


class HashingEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider used when real models are unavailable."""

    def embed(self, text: str) -> Sequence[float]:  # pragma: no cover - simple mapping
        if not text:
            return []
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [value / 255.0 for value in digest]
