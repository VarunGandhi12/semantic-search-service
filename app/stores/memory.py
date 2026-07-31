"""In-memory vector store backed by a single NumPy matrix.

Because vectors arrive L2-normalized, cosine similarity is just a matrix-vector
dot product — one BLAS call for the whole corpus. Fine to tens of thousands of
documents; beyond that, move to pgvector/FAISS behind the same `VectorStore`
protocol.
"""

from __future__ import annotations

import numpy as np

from app.domain.models import Document
from app.embeddings.base import Vector


class InMemoryVectorStore:
    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._ids: list[str] = []
        self._matrix: np.ndarray = np.empty((0, dim), dtype=np.float32)

    def add(self, documents: list[Document], vectors: list[Vector]) -> None:
        if len(documents) != len(vectors):
            raise ValueError("documents and vectors must be the same length")
        if not documents:
            return
        block = np.vstack(vectors).astype(np.float32)
        if block.shape[1] != self._dim:
            raise ValueError(f"expected dim {self._dim}, got {block.shape[1]}")
        self._ids.extend(doc.id for doc in documents)
        self._matrix = np.vstack([self._matrix, block])

    def search(self, query: Vector, k: int) -> list[tuple[str, float]]:
        if len(self._ids) == 0 or k <= 0:
            return []
        sims = self._matrix @ query  # cosine, since everything is normalized
        top = np.argsort(-sims)[:k]
        return [(self._ids[i], float(sims[i])) for i in top]

    def __len__(self) -> int:
        return len(self._ids)
