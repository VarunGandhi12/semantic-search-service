"""BM25 keyword index.

A compact, dependency-free implementation of Okapi BM25 — the lexical half of
hybrid search. It rewards exact term matches that dense vectors often miss
(names, IDs, rare words).
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._ids: list[str] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_freqs: list[Counter[str]] = []
        self._df: Counter[str] = Counter()
        self._avg_len: float = 0.0

    def add(self, doc_id: str, text: str) -> None:
        tokens = _tokenize(text)
        freqs = Counter(tokens)
        self._ids.append(doc_id)
        self._doc_tokens.append(tokens)
        self._doc_freqs.append(freqs)
        for term in freqs:
            self._df[term] += 1
        total = sum(len(t) for t in self._doc_tokens)
        self._avg_len = total / len(self._doc_tokens)

    def _idf(self, term: str) -> float:
        n = len(self._ids)
        df = self._df.get(term, 0)
        # BM25 idf with +1 smoothing to stay non-negative.
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if not self._ids or k <= 0:
            return []
        q_terms = _tokenize(query)
        scores: list[tuple[str, float]] = []
        for doc_id, freqs, tokens in zip(self._ids, self._doc_freqs, self._doc_tokens, strict=True):
            dl = len(tokens)
            score = 0.0
            for term in q_terms:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self._avg_len)
                score += self._idf(term) * (tf * (self.k1 + 1)) / denom
            if score > 0.0:
                scores.append((doc_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def __len__(self) -> int:
        return len(self._ids)
