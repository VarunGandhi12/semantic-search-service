"""Hybrid retrieval via Reciprocal Rank Fusion (RRF).

Dense (vector) and sparse (BM25) retrieval have complementary strengths. RRF
combines their *rankings* rather than their raw scores, so we never have to
reconcile incompatible score scales. Score for a doc is the sum over each ranked
list of ``1 / (rrf_k + rank)``.

Reference: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and
individual rank learning methods" (SIGIR 2009).
"""

from __future__ import annotations

from enum import StrEnum


class SearchMode(StrEnum):
    HYBRID = "hybrid"
    VECTOR = "vector"
    KEYWORD = "keyword"


def _rrf_scores(ranking: list[str], rrf_k: int) -> dict[str, float]:
    return {doc_id: 1.0 / (rrf_k + rank) for rank, doc_id in enumerate(ranking)}


def fuse(
    vector_ranking: list[str],
    keyword_ranking: list[str],
    *,
    mode: SearchMode = SearchMode.HYBRID,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse two rankings (lists of doc ids, best first) into one scored list."""
    combined: dict[str, float] = {}

    if mode in (SearchMode.HYBRID, SearchMode.VECTOR):
        for doc_id, score in _rrf_scores(vector_ranking, rrf_k).items():
            combined[doc_id] = combined.get(doc_id, 0.0) + score
    if mode in (SearchMode.HYBRID, SearchMode.KEYWORD):
        for doc_id, score in _rrf_scores(keyword_ranking, rrf_k).items():
            combined[doc_id] = combined.get(doc_id, 0.0) + score

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return ranked
