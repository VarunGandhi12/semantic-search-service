"""Composition root — build the object graph once and hand it to routes.

This is the single place that chooses concrete adapters. Switching to real
semantic embeddings or a generative LLM is a one-line change here; nothing
downstream cares.
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.domain.corpus import load_sample_corpus
from app.embeddings.hashing import HashingEmbedder
from app.rag.extractive import ExtractiveAnswerer
from app.rag.service import RagService
from app.search.keyword import BM25Index
from app.service import SearchService
from app.stores.memory import InMemoryVectorStore


@lru_cache
def get_search_service() -> SearchService:
    settings = get_settings()
    embedder = HashingEmbedder(dim=settings.embedding_dim)
    service = SearchService(
        embedder=embedder,
        store=InMemoryVectorStore(dim=embedder.dim),
        keyword_index=BM25Index(),
    )
    if settings.load_sample_corpus:
        service.index(load_sample_corpus())
    return service


@lru_cache
def get_rag_service() -> RagService:
    return RagService(search=get_search_service(), llm=ExtractiveAnswerer())
