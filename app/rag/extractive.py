"""Extractive answerer — the offline default LLM adapter.

Rather than call a generative model, it returns the most relevant retrieved
passage with lightweight citation. This keeps the RAG endpoint fully functional
and deterministic in tests, while demonstrating the retrieve-then-answer shape.
Swap in a generative `LLMProvider` for fluent answers.
"""

from __future__ import annotations

from app.domain.models import ScoredDocument


class ExtractiveAnswerer:
    def answer(self, question: str, context: list[ScoredDocument]) -> str:
        if not context:
            return "I couldn't find anything relevant in the indexed corpus."
        best = context[0].document
        cite = f" [source: {best.source}]" if best.source else ""
        return f"{best.text}{cite}"
