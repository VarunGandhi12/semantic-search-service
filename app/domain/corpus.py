"""Loader for the bundled sample corpus."""

from __future__ import annotations

import json
from pathlib import Path

from app.domain.models import Document

_CORPUS_PATH = Path(__file__).resolve().parents[2] / "data" / "corpus.json"


def load_sample_corpus(path: Path = _CORPUS_PATH) -> list[Document]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        Document(id=item["id"], text=item["text"], source=item.get("source", "")) for item in raw
    ]
