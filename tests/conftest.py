from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.embeddings.hashing import HashingEmbedder
from app.main import create_app
from app.search.keyword import BM25Index
from app.service import SearchService
from app.stores.memory import InMemoryVectorStore

DIM = 256


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=DIM)


@pytest.fixture
def service(embedder: HashingEmbedder) -> SearchService:
    return SearchService(
        embedder=embedder,
        store=InMemoryVectorStore(dim=embedder.dim),
        keyword_index=BM25Index(),
    )


@pytest.fixture
def client() -> TestClient:
    # Uses the real composition root, which loads the bundled sample corpus.
    return TestClient(create_app())
