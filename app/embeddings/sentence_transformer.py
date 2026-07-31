"""Optional production adapter: real semantic embeddings.

Requires the `semantic` extra (`pip install ".[semantic]"`), which pulls in
`sentence-transformers`. Import is deferred so the core service stays lightweight
and installable without a multi-hundred-MB ML stack.

This class is intentionally not imported anywhere by default — wire it up in
`app/api/deps.py` when you want true semantic vectors.
"""

from __future__ import annotations

import numpy as np

from app.embeddings.base import Vector


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only with extra
            raise ImportError(
                "sentence-transformers is not installed. "
                'Install the extra: pip install ".[semantic]"'
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> Vector:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[Vector]:
        raw = self._model.encode(texts, normalize_embeddings=True)
        arr = np.asarray(raw, dtype=np.float32)
        return list(arr)
