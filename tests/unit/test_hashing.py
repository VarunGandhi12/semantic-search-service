from __future__ import annotations

import numpy as np
import pytest

from app.embeddings.hashing import HashingEmbedder


def test_output_dim_and_normalization() -> None:
    emb = HashingEmbedder(dim=128)
    v = emb.embed("hello world")
    assert v.shape == (128,)
    assert v.dtype == np.float32
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-6)


def test_deterministic() -> None:
    emb = HashingEmbedder(dim=128)
    assert np.array_equal(emb.embed("same text"), emb.embed("same text"))


def test_similar_texts_more_similar_than_unrelated() -> None:
    emb = HashingEmbedder(dim=512)
    a = emb.embed("the cat sat on the mat")
    b = emb.embed("the cat sat on a mat")
    c = emb.embed("quantum chromodynamics lagrangian")
    assert float(a @ b) > float(a @ c)


def test_empty_text_is_zero_vector() -> None:
    v = HashingEmbedder(dim=64).embed("")
    assert np.linalg.norm(v) == 0.0


def test_invalid_dim_raises() -> None:
    with pytest.raises(ValueError):
        HashingEmbedder(dim=0)
