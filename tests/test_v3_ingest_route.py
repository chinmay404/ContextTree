"""POST /boards/{b}/cards/{id}/ingest: ingest runs inside the request (contract amendment 2026-10-21).

On Vercel, work after a response is best effort, so the upload/link routes only create the card
and the client calls this route right after. A lease stamp in the card's engine-only data keeps
two calls from running the same ingest twice; a lease older than 300 s is taken over.
"""
import inspect
import uuid

import pytest

from app.agent import ingest
from app.api.v1.endpoints import sources
from tests.v3_support import (  # noqa: F401
    FakeLLM, api, db, make_board, new_id, requires_db, seed_board, seed_card, unit_emb, user,
)

pytestmark = requires_db

LEASE = "ingestLease"


@pytest.fixture
def rec(monkeypatch):
    """Ingest stand-ins that record the call and finish the card the way the real ones do."""
    calls = []

    def finish(kind):
        def run(card_id, email):
            calls.append((kind, card_id, email))
            ingest.set_card_status(ingest.store, card_id, "ready", {"summary": f"{kind} done"}, email)
        return run

    monkeypatch.setattr(sources, "ingest_file_source", finish("file"))
    monkeypatch.setattr(sources, "ingest_web_source", finish("web"))
    return calls


def _up(api, b, name="notes.txt", content=b"hello world " * 50, ctype="text/plain"):
    r = api.post(f"/boards/{b}/sources/file", data={"id": new_id("c"), "x": "0", "y": "0"},
                 files={"file": (name, content, ctype)})
    assert r.status_code == 202, r.text
    return r.json()


def _ingest(api, b, cid):
    return api.post(f"/boards/{b}/cards/{cid}/ingest")


def _set_lease(db, cid, age_seconds):
    with db.cursor() as cur:
        cur.execute(
            "UPDATE nodes SET data = data || jsonb_build_object(%s, extract(epoch from now()) - %s) "
            "WHERE id = %s", (LEASE, age_seconds, cid))


def _status(db, cid):
    with db.cursor() as cur:
        cur.execute("SELECT status FROM nodes WHERE id = %s", (cid,))
        return cur.fetchone()[0]


def test_upload_and_link_no_longer_start_ingest(api, rec, monkeypatch):
    b = make_board(api)["id"]
    card = _up(api, b)
    assert card["status"] == "processing"
    monkeypatch.setattr(sources, "host_is_public", lambda h: True)
    r = api.post(f"/boards/{b}/sources/link", json={"id": new_id("c"), "url": "https://harbor.dev/x",
                                                     "x": 0, "y": 0})
    assert r.status_code == 202 and r.json()["status"] == "processing"
    assert rec == []


def test_ingest_runs_the_file_ingest_inline_and_returns_the_finished_card(api, rec):
    b = make_board(api)["id"]
    card = _up(api, b)
    r = _ingest(api, b, card["id"])
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["id"] == card["id"] and out["status"] == "ready" and out["data"]["summary"] == "file done"
    assert LEASE not in out["data"]  # engine-only
    assert rec == [("file", card["id"], api.email)]


def test_ingest_runs_the_web_ingest_for_a_link(api, rec, monkeypatch):
    b = make_board(api)["id"]
    monkeypatch.setattr(sources, "host_is_public", lambda h: True)
    cid = api.post(f"/boards/{b}/sources/link", json={"id": new_id("c"), "url": "https://harbor.dev/x",
                                                       "x": 0, "y": 0}).json()["id"]
    r = _ingest(api, b, cid)
    assert r.status_code == 200 and r.json()["status"] == "ready"
    assert rec == [("web", cid, api.email)]


def test_real_ingest_inline_end_to_end(api, db, monkeypatch):
    monkeypatch.setattr(ingest, "get_embeddings", lambda texts: [unit_emb(i % 700) for i, _ in enumerate(texts)])
    monkeypatch.setattr(ingest, "get_embedding", lambda t: unit_emb(0))
    monkeypatch.setattr(ingest, "get_llm", lambda *a, **k: FakeLLM("An overview."))
    b = make_board(api)["id"]
    card = _up(api, b, content=b"The harbour opens at dawn. " * 200)
    r = _ingest(api, b, card["id"])
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["status"] == "ready" and out["data"]["summary"] == "An overview."
    assert out["data"]["excerpts"] and LEASE not in out["data"]
    passages = api.get(f"/files/{card['id']}/passages").json()["passages"]
    assert passages and passages[0]["text"].startswith("The harbour")


def test_a_fresh_lease_answers_202_processing_without_a_second_run(api, db, rec):
    b = make_board(api)["id"]
    card = _up(api, b)
    _set_lease(db, card["id"], 10)
    r = _ingest(api, b, card["id"])
    assert r.status_code == 202 and r.json() == {"status": "processing"}
    assert rec == [] and _status(db, card["id"]) == "processing"


def test_an_expired_lease_is_taken_over(api, db, rec):
    b = make_board(api)["id"]
    card = _up(api, b)
    _set_lease(db, card["id"], 301)
    r = _ingest(api, b, card["id"])
    assert r.status_code == 200 and r.json()["status"] == "ready"
    assert len(rec) == 1


def test_a_malformed_lease_is_taken_over(api, db, rec):
    b = make_board(api)["id"]
    card = _up(api, b)
    with db.cursor() as cur:
        cur.execute("UPDATE nodes SET data = data || '{\"ingestLease\": \"soon\"}'::jsonb WHERE id = %s",
                    (card["id"],))
    assert _ingest(api, b, card["id"]).status_code == 200 and len(rec) == 1


def test_the_lease_is_claimed_once(api, db):
    from app.agent.store.board_store import store
    b = make_board(api)["id"]
    card = _up(api, b)
    with store.tx() as cur:
        first = sources.claim_ingest_lease(cur, card["id"], api.email)
    with store.tx() as cur:
        second = sources.claim_ingest_lease(cur, card["id"], api.email)
    assert (first, second) == (True, False)


def test_a_second_call_while_running_is_202(api, db, monkeypatch):
    """The lease is written before ingest starts, so a call that lands mid-run does not run it again."""
    seen = []

    def slow(card_id, email):
        seen.append(_ingest(api, b, card_id).status_code)  # a second request while this one runs
        ingest.set_card_status(ingest.store, card_id, "ready", {}, email)

    monkeypatch.setattr(sources, "ingest_file_source", slow)
    b = make_board(api)["id"]
    card = _up(api, b)
    assert _ingest(api, b, card["id"]).status_code == 200
    assert seen == [202]


@pytest.mark.parametrize("status", ["ready", "failed", "soon"])
def test_finished_or_soon_sources_return_as_is(api, db, rec, status):
    b = make_board(api)["id"]
    cid = seed_card(db, api.email, b, kind="source", source_type="pdf", status=status,
                    data={"title": "T"})
    r = _ingest(api, b, cid)
    assert r.status_code == 200 and r.json()["status"] == status and r.json()["id"] == cid
    assert rec == []


def test_a_note_returns_as_is_and_a_chat_is_400(api, db, rec):
    b = make_board(api)["id"]
    note = seed_card(db, api.email, b, kind="source", source_type="note", status="ready",
                     data={"title": "N", "text": "t"})
    assert _ingest(api, b, note).json()["status"] == "ready"
    r = _ingest(api, b, f"{b}_q1")
    assert r.status_code == 400 and r.json()["error"] == "invalid"
    assert rec == []


def test_ingest_is_owner_scoped_and_skips_deleted(api, db, rec):
    other = f"other_{uuid.uuid4().hex[:10]}@example.com"
    with db.cursor() as cur:
        cur.execute("INSERT INTO users (email, name) VALUES (%s, 'O')", (other,))
    try:
        theirs_board = seed_board(db, other)
        theirs = seed_card(db, other, theirs_board, kind="source", source_type="pdf", status="processing")
        mine = make_board(api)["id"]
        dead = seed_card(db, api.email, mine, kind="source", source_type="pdf", status="processing",
                         deleted_at="2026-01-01")
        elsewhere = make_board(api)["id"]
        card = _up(api, elsewhere)
        for board, cid in ((theirs_board, theirs), (mine, theirs), (mine, dead), (mine, card["id"]),
                           (mine, "c_missing")):
            r = _ingest(api, board, cid)
            assert r.status_code == 404 and r.json()["error"] == "not_found", (board, cid)
        assert rec == []
        assert _status(db, theirs) == "processing"
    finally:
        with db.cursor() as cur:
            cur.execute("DELETE FROM nodes WHERE user_email = %s", (other,))
            cur.execute("DELETE FROM canvases WHERE user_email = %s", (other,))
            cur.execute("DELETE FROM users WHERE email = %s", (other,))


def test_the_lease_never_reaches_the_client(api, db):
    b = make_board(api)["id"]
    card = _up(api, b)
    _set_lease(db, card["id"], 5)
    board = api.get(f"/boards/{b}").json()
    [c] = [c for c in board["cards"] if c["id"] == card["id"]]
    assert LEASE not in c["data"]
    r = api.patch(f"/boards/{b}/cards/{card['id']}", json={"data": {LEASE: 0}})
    assert r.status_code == 200 and LEASE not in r.json()["data"]
    with db.cursor() as cur:  # a client cannot reset the lease
        cur.execute("SELECT (data->>%s)::float8 > extract(epoch from now()) - 60 FROM nodes WHERE id = %s",
                    (LEASE, card["id"]))
        assert cur.fetchone()[0] is True


def test_the_route_runs_in_the_threadpool():
    # A sync route: FastAPI runs it in the threadpool, so ingest's blocking DB, embedding and LLM
    # calls never sit on the event loop (pool-gate rule).
    assert not inspect.iscoroutinefunction(sources.api_ingest_source)


def test_lease_seconds_match_the_vercel_limit():
    assert sources.INGEST_LEASE_SECONDS == 300
