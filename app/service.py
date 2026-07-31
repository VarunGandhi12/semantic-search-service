"""SearchService — wires embeddings, the vector store and the BM25 index into a
single index/search facade. This is the seam the API and the RAG layer depend on.
"""

from __future__ import annotations

from app.domain.models import Document, ScoredDocument
from app.embeddings.base import EmbeddingProvider
from app.search import hybrid
from app.search.hybrid import SearchMode
from app.search.keyword import BM25Index
from app.stores.base import VectorStore

# Over-fetch from each retriever before fusion so RRF has enough candidates to
# work with; a good rule of thumb is a few times the requested k.
_CANDIDATE_MULTIPLIER = 5
_MIN_CANDIDATES = 20


class SearchService:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStore,
        keyword_index: BM25Index,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._keyword = keyword_index
        self._docs: dict[str, Document] = {}

    @property
    def size(self) -> int:
        return len(self._docs)

    def index(self, documents: list[Document]) -> int:
        new = [d for d in documents if d.id not in self._docs]
        if not new:
            return 0
        vectors = self._embedder.embed_batch([d.text for d in new])
        self._store.add(new, vectors)
        for doc in new:
            self._keyword.add(doc.id, doc.text)
            self._docs[doc.id] = doc
        return len(new)

    def search(
        self, query: str, k: int = 5, mode: SearchMode = SearchMode.HYBRID
    ) -> list[ScoredDocument]:
        if not query.strip() or self.size == 0:
            return []
        candidates = max(_MIN_CANDIDATES, k * _CANDIDATE_MULTIPLIER)

        vector_ranking = [
            doc_id for doc_id, _ in self._store.search(self._embedder.embed(query), candidates)
        ]
        keyword_ranking = [doc_id for doc_id, _ in self._keyword.search(query, candidates)]

        fused = hybrid.fuse(vector_ranking, keyword_ranking, mode=mode)
        return [
            ScoredDocument(document=self._docs[doc_id], score=score)
            for doc_id, score in fused[:k]
            if doc_id in self._docs
        ]
