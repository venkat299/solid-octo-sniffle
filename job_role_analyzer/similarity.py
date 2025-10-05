from __future__ import annotations

import math
from typing import List, Protocol, Sequence, Tuple

try:  # pragma: no cover - exercised indirectly when faiss is installed
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - fallback for environments without faiss
    faiss = None  # type: ignore[assignment]
    _FAISS_AVAILABLE = False

try:  # pragma: no cover - numpy is optional when faiss isn't available
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - fallback path when numpy is missing
    np = None  # type: ignore[assignment]

from .config import load_config
from .data_models import JobRoleSummary
from .db import Database


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> Sequence[float]:
        ...


def _normalize_vector(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [component / norm for component in vector]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(left, right))


class _FallbackFaissIndex:
    """Lightweight FAISS substitute used when the library isn't available."""

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._vectors: List[List[float]] = []

    def add(self, matrix: Sequence[Sequence[float]]) -> None:
        for vector in matrix:
            values = list(vector)
            if len(values) != self._dimension:
                raise ValueError("All vectors must match the index dimensionality.")
            self._vectors.append(_normalize_vector(values))

    def search(self, matrix: Sequence[Sequence[float]], k: int) -> Tuple[List[List[float]], List[List[int]]]:
        requested_k = max(k, 0)
        if not self._vectors:
            zero_scores = [[0.0] * requested_k for _ in matrix]
            zero_indices = [[-1] * requested_k for _ in matrix]
            return zero_scores, zero_indices

        results_scores: List[List[float]] = []
        results_indices: List[List[int]] = []
        for query in matrix:
            values = list(query)
            if len(values) != self._dimension:
                raise ValueError("Query vectors must match the index dimensionality.")
            normalized_query = _normalize_vector(values)
            scores = [_dot(normalized_query, candidate) for candidate in self._vectors]
            ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
            top = ranked[:requested_k]
            padded_scores = [score for _, score in top]
            padded_indices = [idx for idx, _ in top]
            while len(padded_scores) < requested_k:
                padded_scores.append(0.0)
                padded_indices.append(-1)
            results_scores.append(padded_scores)
            results_indices.append(padded_indices)
        return results_scores, results_indices


class _FaissWrapper:
    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        if _FAISS_AVAILABLE and np is not None:
            self._index = faiss.IndexFlatIP(dimension)  # type: ignore[call-arg]
        else:
            self._index = _FallbackFaissIndex(dimension)
        self._use_numpy = _FAISS_AVAILABLE and np is not None

    def add(self, matrix: Sequence[Sequence[float]]) -> None:
        if self._use_numpy:
            array = np.array(matrix, dtype="float32")
            faiss.normalize_L2(array)  # type: ignore[operator]
            self._index.add(array)
        else:
            self._index.add(matrix)

    def search(self, matrix: Sequence[Sequence[float]], k: int) -> Tuple[List[List[float]], List[List[int]]]:
        if self._use_numpy:
            array = np.array(matrix, dtype="float32")
            faiss.normalize_L2(array)  # type: ignore[operator]
            distances, indices = self._index.search(array, k)
            return distances.tolist(), indices.tolist()
        return self._index.search(matrix, k)


class SimilarityChecker:
    def __init__(self, db: Database, embedding_provider: EmbeddingProvider) -> None:
        self.db = db
        self.embedding_provider = embedding_provider
        self.config = load_config()
        self._index: _FaissWrapper | None = None
        self._job_roles: List[JobRoleSummary] = []
        self._dimension: int | None = None
        self._ensure_index_initialized()

    def _ensure_index_initialized(self) -> None:
        if self.config.similarity_backend.lower() != "faiss":
            raise ValueError("Only the 'faiss' similarity backend is currently supported.")
        if self._index is not None:
            return
        embeddings: List[List[float]] = []
        job_roles: List[JobRoleSummary] = []
        for job_role, stored_embedding in self.db.iter_job_role_embeddings():
            if stored_embedding:
                job_roles.append(job_role)
                embeddings.append(list(stored_embedding))
        if not embeddings:
            return
        self._dimension = len(embeddings[0])
        self._index = _FaissWrapper(self._dimension)
        self._index.add(embeddings)
        self._job_roles = job_roles

    def _prepare_query(self, job_description: str) -> List[List[float]]:
        candidate_embedding = list(self.embedding_provider.embed(job_description))
        if not candidate_embedding:
            return []
        if self._dimension is not None and len(candidate_embedding) != self._dimension:
            raise ValueError("Embedding provider returned a vector with unexpected dimensionality.")
        return [candidate_embedding]

    def find_similar_role(self, job_description: str) -> Tuple[JobRoleSummary, float] | None:
        self._ensure_index_initialized()
        if self._index is None:
            return None
        query_matrix = self._prepare_query(job_description)
        if not query_matrix:
            return None
        distances, indices = self._index.search(query_matrix, k=1)
        best_index = indices[0][0]
        if best_index < 0:
            return None
        similarity = float(distances[0][0])
        if similarity < self.config.job_role_similarity_threshold:
            return None
        return self._job_roles[best_index], similarity

    def compute_embedding(self, job_description: str) -> List[float]:
        return list(self.embedding_provider.embed(job_description))

    def add_to_index(self, job_role: JobRoleSummary, embedding: Sequence[float]) -> None:
        vector = list(embedding)
        if not vector:
            return
        if self._index is None:
            self._dimension = len(vector)
            self._index = _FaissWrapper(self._dimension)
            self._job_roles = []
        elif len(vector) != self._dimension:
            raise ValueError("Embedding dimensionality must remain consistent for FAISS index.")
        self._index.add([vector])
        self._job_roles.append(job_role)
