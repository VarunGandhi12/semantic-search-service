from __future__ import annotations

import pytest

from app.embeddings.sentence_transformer import SentenceTransformerEmbedder


def test_adapter_raises_helpful_error_without_the_extra() -> None:
    # The `semantic` extra is not installed in the default/CI environment, so the
    # adapter should fail fast with an actionable message rather than a raw
    # ModuleNotFoundError.
    with pytest.raises(ImportError, match="sentence-transformers"):
        SentenceTransformerEmbedder()
