"""Embedding provider port.

The search engine depends only on this protocol. The default adapter is a
deterministic hashing embedder (offline, no model download); a
sentence-transformers adapter is a drop-in replacement for true semantic
vectors. See `app/embeddings/`.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float32]


class EmbeddingProvider(Protocol):
    #: Dimensionality of the vectors this provider emits.
    dim: int

    def embed(self, text: str) -> Vector:
        """Embed one string into an L2-normalized vector."""
        ...

    def embed_batch(self, texts: list[str]) -> list[Vector]:
        """Embed many strings. Defaults may just map `embed`, but batching lets
        real backends amortize model calls."""
        ...
