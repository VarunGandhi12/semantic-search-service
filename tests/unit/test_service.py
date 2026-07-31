from __future__ import annotations

from app.domain.models import Document
from app.search.hybrid import SearchMode
from app.service import SearchService


def _docs() -> list[Document]:
    return [
        Document(id="d1", text="postgres pgvector stores embeddings for search", source="s"),
        Document(id="d2", text="bm25 ranks documents by term frequency", source="s"),
        Document(id="d3", text="cats and dogs are common household pets", source="s"),
    ]


def test_index_is_idempotent_on_id(service: SearchService) -> None:
    assert service.index(_docs()) == 3
    assert service.index(_docs()) == 0  # same ids -> nothing new
    assert service.size == 3


def test_search_finds_relevant_document(service: SearchService) -> None:
    service.index(_docs())
    hits = service.search("vector database embeddings", k=1)
    assert hits[0].document.id == "d1"


def test_empty_query_returns_nothing(service: SearchService) -> None:
    service.index(_docs())
    assert service.search("   ", k=3) == []


def test_search_on_empty_index_returns_nothing(service: SearchService) -> None:
    assert service.search("anything", k=3) == []


def test_keyword_mode_matches_exact_term(service: SearchService) -> None:
    service.index(_docs())
    hits = service.search("bm25", k=1, mode=SearchMode.KEYWORD)
    assert hits[0].document.id == "d2"
