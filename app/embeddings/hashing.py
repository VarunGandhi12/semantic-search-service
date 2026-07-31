"""Deterministic hashing embedder (the offline default).

Uses the "hashing trick": token uni/bi-grams are hashed into a fixed-dimension
vector with signed accumulation, then L2-normalized. No model, no network, fully
deterministic — which makes it perfect for tests, CI, and reproducible demos.

It captures lexical overlap well. For true semantic similarity, swap in
`SentenceTransformerEmbedder` (see sibling module) — nothing else changes,
because both satisfy the `EmbeddingProvider` protocol.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from app.embeddings.base import Vector

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    bigrams = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
    return tokens + bigrams


def _hash(token: str) -> tuple[int, float]:
    digest = hashlib.md5(token.encode("utf-8")).digest()  # noqa: S324 — not security
    index = int.from_bytes(digest[:4], "little")
    sign = 1.0 if digest[4] & 1 else -1.0
    return index, sign


class HashingEmbedder:
    def __init__(self, dim: int = 512) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def embed(self, text: str) -> Vector:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in _tokenize(text):
            index, sign = _hash(token)
            vec[index % self.dim] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    def embed_batch(self, texts: list[str]) -> list[Vector]:
        return [self.embed(t) for t in texts]
