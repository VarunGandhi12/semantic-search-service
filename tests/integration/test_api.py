from __future__ import annotations


def test_health_reports_indexed_corpus(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["documents_indexed"] >= 12  # bundled sample corpus


def test_search_returns_relevant_hits(client) -> None:
    resp = client.post("/v1/search", json={"query": "how does hybrid search work", "k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "hybrid"
    assert len(body["hits"]) == 3
    ids = {h["id"] for h in body["hits"]}
    assert ids & {"hybrid", "bm25", "rrf"}  # at least one retrieval-notes doc


def test_ask_returns_grounded_answer(client) -> None:
    resp = client.post("/v1/ask", json={"question": "what is reciprocal rank fusion", "k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert len(body["sources"]) >= 1
    assert body["sources"][0]["id"] == "rrf"


def test_index_new_document_then_find_it(client) -> None:
    resp = client.post(
        "/v1/documents",
        json={
            "documents": [
                {"id": "custom-1", "text": "zebrafish neuroscience research", "source": "x"}
            ]
        },
    )
    assert resp.status_code == 200
    assert resp.json()["indexed"] == 1

    found = client.post("/v1/search", json={"query": "zebrafish neuroscience", "k": 1})
    assert found.json()["hits"][0]["id"] == "custom-1"


def test_search_validates_k_upper_bound(client) -> None:
    resp = client.post("/v1/search", json={"query": "x", "k": 999})
    assert resp.status_code == 422


def test_empty_query_is_rejected(client) -> None:
    resp = client.post("/v1/search", json={"query": "", "k": 3})
    assert resp.status_code == 422
