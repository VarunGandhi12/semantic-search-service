"""RAG orchestration: retrieve with the search service, then answer with an
LLM provider. The two responsibilities stay separate and independently testable.
"""

from __future__ import annotations

from app.domain.models import Answer
from app.rag.base import LLMProvider
from app.search.hybrid import SearchMode
from app.service import SearchService


class RagService:
    def __init__(self, search: SearchService, llm: LLMProvider) -> None:
        self._search = search
        self._llm = llm

    def ask(self, question: str, k: int = 4) -> Answer:
        retrieved = self._search.search(question, k=k, mode=SearchMode.HYBRID)
        text = self._llm.answer(question, retrieved)
        return Answer(text=text, sources=retrieved)
