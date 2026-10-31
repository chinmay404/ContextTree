"""Embeddings through NVIDIA NIM. Off until EMBEDDING_PROVIDER=nim; failures stay zero vectors."""
import json
import math

import httpx
import pytest

from app.agent.utils import embeddings
from app.core.config import settings
from scripts.reembed import reembed
from tests.v3_support import _vec, db, new_id, requires_db, seed_board, seed_card, seed_message, unit_emb, user  # noqa: F401


class Wire:
    def __init__(self, status=200, reject_over=None):
        self.status, self.reject_over, self.requests = status, reject_over, []

    def __call__(self, request):
        body = json.loads(request.content)
        self.requests.append((request, body))
        if self.status != 200:
            return httpx.Response(self.status, json={"detail": "nope"})
        if self.reject_over and any(len(t) > self.reject_over for t in body["input"]):
            return httpx.Response(400, json={"detail": "too long"})
        data = [{"index": i, "embedding": [float(len(t) + 1)] + [1.0] * 2047} for i, t in enumerate(body["input"])]
        return httpx.Response(200, json={"data": list(reversed(data))})


@pytest.fixture
def wire(monkeypatch):
    w = Wire()
    monkeypatch.setattr(settings, "ENGINE_FAKE_LLM", False)
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "nim")
    monkeypatch.setattr(settings, "NVIDIA_API_KEY", "nvapi-secret")
    monkeypatch.setattr(embeddings, "_nim_transport", lambda: httpx.MockTransport(w))
    return w


def test_google_stays_the_default():
    assert settings.EMBEDDING_PROVIDER == "google"


def test_request_shape_and_auth(wire):
    embeddings.get_embedding("hello")
    request, body = wire.requests[0]
    assert str(request.url) == "https://integrate.api.nvidia.com/v1/embeddings"
    assert request.headers["authorization"] == "Bearer nvapi-secret"
    assert body["model"] == settings.NIM_EMBEDDING_MODEL and body["input"] == ["hello"]


def test_a_question_and_a_stored_text_are_marked_differently(wire):
    """Without this flag the live model scored 3 of 12 on a small check; with it, 12 of 12."""
    embeddings.get_embedding("a question")
    embeddings.get_embeddings(["a document"])
    embeddings.get_passage_embedding("an answer to keep")
    assert [b["input_type"] for _, b in wire.requests] == ["query", "passage", "passage"]


def test_stored_messages_are_embedded_as_passages(wire, monkeypatch):
    from app.agent import main
    monkeypatch.setattr(main, "get_embedding", lambda t: (_ for _ in ()).throw(AssertionError("query path")))
    assert any(main._stored_embedding("an answer to keep"))
    assert wire.requests[-1][1]["input_type"] == "passage"


def test_with_gemini_stored_text_keeps_the_one_kind_of_call(monkeypatch):
    from app.agent import main
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "google")
    monkeypatch.setattr(main, "get_embedding", lambda t: [1.0] * 768)
    assert main._stored_embedding("text") == [1.0] * 768


def test_very_long_text_is_clipped_not_refused(wire):
    wire.reject_over = embeddings.NIM_MAX_CHARS
    assert any(embeddings.get_embedding("x" * 80_000))
    assert len(wire.requests[-1][1]["input"][0]) == embeddings.NIM_MAX_CHARS


def test_vector_fits_the_768_column_and_is_a_unit_vector(wire):
    vector = embeddings.get_embedding("hello")
    assert len(vector) == 768
    assert math.isclose(sum(x * x for x in vector), 1.0, rel_tol=1e-9)
    assert all(x == 0.0 for x in vector[embeddings.NIM_KEEP_DIMS:])
    assert any(x != 0.0 for x in vector[:embeddings.NIM_KEEP_DIMS])


def test_results_follow_input_order_even_if_the_reply_does_not(wire):
    a, b = embeddings.get_embeddings(["x", "much longer text"])
    assert a[0] != b[0] and a[0] < b[0]


def test_large_lists_go_in_batches(wire):
    assert len(embeddings.get_embeddings(["t"] * (embeddings.NIM_BATCH * 2 + 1))) == embeddings.NIM_BATCH * 2 + 1
    assert [len(b["input"]) for _, b in wire.requests] == [embeddings.NIM_BATCH, embeddings.NIM_BATCH, 1]


def test_one_refused_text_does_not_lose_its_batchmates(wire):
    wire.reject_over = 10
    out = embeddings.get_embeddings(["short", "x" * 50, "also"])
    assert len(out) == 3
    assert any(v != 0.0 for v in out[0]) and any(v != 0.0 for v in out[2])
    assert all(v == 0.0 for v in out[1])


def test_empty_text_is_not_sent_empty(wire):
    embeddings.get_embeddings(["", "a"])
    assert all(t.strip() for t in wire.requests[0][1]["input"])


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_failure_gives_zero_vectors_so_keyword_search_takes_over(wire, status, caplog):
    wire.status = status
    assert not any(embeddings.get_embedding("hello"))
    assert all(not any(v) for v in embeddings.get_embeddings(["a", "b"]))
    assert "nvapi-secret" not in caplog.text


def test_no_key_never_calls_out(wire, monkeypatch):
    monkeypatch.setattr(settings, "NVIDIA_API_KEY", None)
    assert not any(embeddings.get_embedding("hello"))
    assert wire.requests == []


def test_nim_never_falls_back_to_google(wire, monkeypatch):
    """Vectors from two models can't be compared, so a fallback would quietly break search."""
    def boom(*a, **k):
        raise AssertionError("google must not be called")
    monkeypatch.setattr(embeddings, "_get_embedding_model", boom)
    wire.status = 500
    embeddings.get_embedding("hello")


# ── Re-embedding what is already stored ──────────────────────────────────────

@pytest.fixture
def stored(db, user):
    b = seed_board(db, user)
    card = seed_card(db, user, b, kind="source", source_type="pdf", status="ready")
    chat = seed_card(db, user, b, kind="chat")
    fid = new_id("f")
    with db.cursor() as cur:
        cur.execute("INSERT INTO external_files (id, node_id, canvas_id, user_email, file_name, file_type) "
                    "VALUES (%s,%s,%s,%s,'f','pdf')", (fid, card, b, user))
        for i in range(3):
            cur.execute("INSERT INTO file_chunks (file_id, chunk_index, chunk_text, metadata, embedding) "
                        "VALUES (%s,%s,%s,'{}'::jsonb,%s::vector)", (fid, i, f"chunk {i}", _vec(unit_emb(1))))
    seed_message(db, user, chat, "user", "a message", 1, embedding=unit_emb(1))
    return dict(fid=fid, chat=chat)


def _chunk_vectors(db, fid):
    with db.cursor() as cur:
        cur.execute("SELECT embedding::text FROM file_chunks WHERE file_id = %s ORDER BY chunk_index", (fid,))
        return [r[0] for r in cur.fetchall()]


@requires_db
def test_dry_run_counts_and_changes_nothing(db, stored, wire):
    before = _chunk_vectors(db, stored["fid"])
    counts = reembed(db, apply=False)
    assert counts["file_chunks"] >= 3 and counts["messages"] >= 1
    assert _chunk_vectors(db, stored["fid"]) == before and wire.requests == []


@requires_db
def test_apply_replaces_chunks_and_messages(db, stored, wire):
    before = _chunk_vectors(db, stored["fid"])
    reembed(db, apply=True, batch=2)
    assert all(a != b for a, b in zip(before, _chunk_vectors(db, stored["fid"])))
    with db.cursor() as cur:
        cur.execute("SELECT embedding::text FROM messages WHERE node_id = %s", (stored["chat"],))
        assert cur.fetchone()[0] != _vec(unit_emb(1))


@requires_db
def test_a_failed_provider_stops_the_run_and_never_stores_zeros(db, stored, wire):
    before = _chunk_vectors(db, stored["fid"])
    wire.status = 403
    with pytest.raises(SystemExit):
        reembed(db, apply=True)
    assert _chunk_vectors(db, stored["fid"]) == before
