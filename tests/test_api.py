"""API contract tests — the POST /query shape is frozen for the
external eval harness, so these assert the exact key sets."""


def test_health_reports_index_state(app_client):
    r = app_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["index_loaded"] is True
    assert body["chunks"] >= 4
    assert body["embedder"] == "tfidf"
    assert body["llm"] == "stub"


def test_query_contract_shape(app_client):
    r = app_client.post("/query", json={"question": "How do I avoid deadlock?"})
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"answer", "sources"}
    assert isinstance(body["answer"], str) and body["answer"]
    assert len(body["sources"]) >= 1
    for src in body["sources"]:
        assert set(src.keys()) == {"file", "section", "score"}
        assert isinstance(src["score"], float)
    # Most relevant fixture doc should surface first for this question.
    assert body["sources"][0]["file"] == "concurrency.md"


def test_query_rejects_blank_question(app_client):
    r = app_client.post("/query", json={"question": "   "})
    assert r.status_code == 422


def test_query_without_index_returns_503(no_index_client):
    r = no_index_client.post("/query", json={"question": "anything?"})
    assert r.status_code == 503


def test_stub_answer_cites_sources(app_client):
    r = app_client.post("/query", json={"question": "What are the four caching patterns?"})
    body = r.json()
    assert "caching.md" in body["answer"]


def test_top_k_is_bounded(app_client):
    r = app_client.post("/query", json={"question": "deadlock", "top_k": 99})
    assert r.status_code == 422
