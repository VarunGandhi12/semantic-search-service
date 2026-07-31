from __future__ import annotations

from app.search.keyword import BM25Index


def _index() -> BM25Index:
    idx = BM25Index()
    idx.add("a", "the quick brown fox jumps over the lazy dog")
    idx.add("b", "a fast auburn fox leaps above a sleepy hound")
    idx.add("c", "database indexing with inverted files and postings")
    return idx


def test_exact_term_ranks_first() -> None:
    results = _index().search("database indexing", k=3)
    assert results[0][0] == "c"


def test_no_match_returns_empty() -> None:
    assert _index().search("zzz nonexistent", k=3) == []


def test_k_limits_results() -> None:
    idx = _index()
    assert len(idx.search("fox", k=1)) == 1


def test_rarer_term_scores_higher() -> None:
    idx = BM25Index()
    idx.add("common", "fox fox fox common common")
    idx.add("rare", "fox pangolin")
    # "pangolin" appears in one doc only -> high idf; that doc should win for it.
    results = idx.search("pangolin", k=2)
    assert results[0][0] == "rare"
