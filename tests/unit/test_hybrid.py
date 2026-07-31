from __future__ import annotations

from app.search.hybrid import SearchMode, fuse


def test_hybrid_rewards_agreement() -> None:
    # "x" is ranked highly by both retrievers -> should top the fused list.
    vector = ["x", "y", "z"]
    keyword = ["x", "w", "v"]
    fused = fuse(vector, keyword, mode=SearchMode.HYBRID)
    assert fused[0][0] == "x"


def test_vector_only_ignores_keyword_ranking() -> None:
    fused = fuse(["a", "b"], ["c", "d"], mode=SearchMode.VECTOR)
    ids = {doc_id for doc_id, _ in fused}
    assert ids == {"a", "b"}


def test_keyword_only_ignores_vector_ranking() -> None:
    fused = fuse(["a", "b"], ["c", "d"], mode=SearchMode.KEYWORD)
    ids = {doc_id for doc_id, _ in fused}
    assert ids == {"c", "d"}


def test_rrf_scores_decrease_with_rank() -> None:
    fused = dict(fuse(["a", "b", "c"], [], mode=SearchMode.VECTOR))
    assert fused["a"] > fused["b"] > fused["c"]
