"""Core domain types, framework-free.

Everything below the API layer speaks in these types, never in Pydantic or HTTP
objects — so the retrieval engine could be reused from a CLI or a worker
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    source: str = ""


@dataclass(frozen=True)
class ScoredDocument:
    document: Document
    score: float


@dataclass(frozen=True)
class Answer:
    """A RAG answer plus the documents it was grounded in."""

    text: str
    sources: list[ScoredDocument]
