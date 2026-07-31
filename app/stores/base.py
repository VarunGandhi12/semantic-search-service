"""Vector store port.

Abstracts "hold vectors, return the nearest ones to a query". The default is an
in-memory NumPy store; a pgvector adapter (documented in the README) is a
drop-in for persistence and scale.
"""

from __future__ import annotations

from typing import Protocol

from app.domain.models import Document
from app.embeddings.base import Vector


class VectorStore(Protocol):
    def add(self, documents: list[Document], vectors: list[Vector]) -> None: ...

    def search(self, query: Vector, k: int) -> list[tuple[str, float]]:
        """Return up to `k` (document_id, similarity) pairs, best first."""
        ...

    def __len__(self) -> int: ...
