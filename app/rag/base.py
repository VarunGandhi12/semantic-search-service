"""LLM provider port for the RAG layer.

The default adapter (`ExtractiveAnswerer`) needs no API key and runs in CI. A
real generative adapter (Gemini, OpenAI, a local model) implements the same
protocol — the RAG service doesn't change.
"""

from __future__ import annotations

from typing import Protocol

from app.domain.models import ScoredDocument


class LLMProvider(Protocol):
    def answer(self, question: str, context: list[ScoredDocument]) -> str:
        """Produce an answer grounded in the retrieved context."""
        ...
