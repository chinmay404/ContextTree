import pytest

from app.api.v1.endpoints import files
from tests.v3_support import requires_db, db, user, api, new_id, seed_board, seed_card  # noqa: F401

pytestmark = requires_db


class Rec:
    def __init__(self):
        self.calls = []


@pytest.fixture
def rec(monkeypatch):
    r = Rec()
    monkeypatch.setattr(files, "process_file_background", lambda req: r.calls.append(req))
    return r


def _seed_file(db, owner_email, fid, node_from=None):
    b = seed_board(db, owner_email)
    c = seed_card(db, owner_email, b, kind="source", source_type="pdf")
    if node_from:
        c = node_from
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO external_files (id, user_email, node_id, canvas_id, file_name, file_type, "
            "file_size, data, mime_type, processed) VALUES (%s,%s,%s,%s,'a.pdf','application/pdf',3,%s,"
            "'application/pdf',false)", (fid, owner_email, c, b, b"abc"))


def test_ingest_other_users_file_404_not_scheduled(api, db, rec):
    other = f"other_{new_id('u')}@example.com"
    with db.cursor() as cur:
        cur.execute("INSERT INTO users (email, name) VALUES (%s, 'O')", (other,))
    fid = new_id("file")
    try:
        _seed_file(db, other, fid)
        r = api.post("/files/ingest", json={"file_id": fid})
        assert r.status_code == 404 and r.json()["error"] == "not_found"
        assert rec.calls == []
    finally:
        with db.cursor() as cur:
            cur.execute("DELETE FROM external_files WHERE user_email = %s", (other,))
            for t in ("nodes", "canvases"):
                cur.execute(f"DELETE FROM {t} WHERE user_email = %s", (other,))
            cur.execute("DELETE FROM users WHERE email = %s", (other,))


def test_ingest_unknown_file_404(api, rec):
    r = api.post("/files/ingest", json={"file_id": new_id("nofile")})
    assert r.status_code == 404 and r.json()["error"] == "not_found"
    assert rec.calls == []


def test_ingest_own_file_202_scheduled(api, db, rec):
    fid = new_id("file")
    _seed_file(db, api.email, fid)
    r = api.post("/files/ingest", json={"file_id": fid})
    assert r.status_code == 202
    assert r.json() == {"status": "processing_started", "file_id": fid}
    assert len(rec.calls) == 1


def _other_user(db):
    other = f"other_{new_id('u')}@example.com"
    with db.cursor() as cur:
        cur.execute("INSERT INTO users (email, name) VALUES (%s, 'O')", (other,))
    return other


def _drop_other(db, other):
    with db.cursor() as cur:
        cur.execute("DELETE FROM external_files WHERE user_email = %s", (other,))
        for t in ("nodes", "canvases"):
            cur.execute(f"DELETE FROM {t} WHERE user_email = %s", (other,))
        cur.execute("DELETE FROM users WHERE email = %s", (other,))


def _node_of(db, fid):
    with db.cursor() as cur:
        cur.execute("SELECT node_id, canvas_id FROM external_files WHERE id = %s", (fid,))
        return cur.fetchone()


def test_ingest_own_file_foreign_body_node_id_404(api, db, rec):
    other = _other_user(db)
    try:
        vb = seed_board(db, other)
        victim = seed_card(db, other, vb, kind="source", source_type="pdf")
        fid = new_id("file")
        _seed_file(db, api.email, fid)
        r = api.post("/files/ingest", json={"file_id": fid, "node_id": victim})
        assert r.status_code == 404 and r.json()["error"] == "not_found"
        assert rec.calls == []
    finally:
        _drop_other(db, other)


def test_ingest_own_file_row_pointing_at_foreign_node_404(api, db, rec):
    other = _other_user(db)
    try:
        vb = seed_board(db, other)
        victim = seed_card(db, other, vb, kind="source", source_type="pdf")
        fid = new_id("file")
        _seed_file(db, api.email, fid, node_from=victim)
        r = api.post("/files/ingest", json={"file_id": fid})
        assert r.status_code == 404 and rec.calls == []
    finally:
        _drop_other(db, other)


def test_ingest_matching_node_id_202_uses_stored_target(api, db, rec):
    fid = new_id("file")
    _seed_file(db, api.email, fid)
    node, canvas = _node_of(db, fid)
    r = api.post("/files/ingest", json={"file_id": fid, "node_id": node, "canvas_id": "evil"})
    assert r.status_code == 202 and len(rec.calls) == 1
    assert (rec.calls[0].node_id, rec.calls[0].canvas_id, rec.calls[0].user_email) == (node, canvas, api.email)


def test_ingest_body_omits_node_id_uses_stored_node(api, db, rec):
    fid = new_id("file")
    _seed_file(db, api.email, fid)
    node, _ = _node_of(db, fid)
    assert api.post("/files/ingest", json={"file_id": fid, "user_email": "x@evil.com"}).status_code == 202
    assert rec.calls[0].node_id == node and rec.calls[0].user_email == api.email


def test_node_updates_scoped_by_user_email(db, user):
    from app.agent.store.PostgresStore import PostgresConversationStore
    b = seed_board(db, user)
    node = seed_card(db, user, b, kind="source", source_type="pdf")
    st = PostgresConversationStore()
    assert st.update_node_data_content(node, "x", "someone@else.com") is False
    assert st.update_node_processing_error(node, "boom", "someone@else.com") is False
    assert st.update_node_data_content(node, "x", user) is True
    assert st.update_node_processing_error(node, "boom", user) is True
    assert st.update_node_data_content(node, "y") is True  # V2 call shape unchanged


# ── Vercel (contract amendment 2026-10-21): V2 ingest runs inside the request ──

def test_ingest_runs_inside_the_request_not_in_background_tasks():
    import inspect
    assert not any(p.annotation.__name__ == "BackgroundTasks"
                   for p in inspect.signature(files.ingest_file).parameters.values()
                   if isinstance(p.annotation, type))


def test_ingest_is_done_off_the_event_loop_before_the_202(api, db, monkeypatch):
    import asyncio
    seen = []

    def process(req):
        try:
            asyncio.get_running_loop()
            seen.append("on the event loop")
        except RuntimeError:
            seen.append("worker thread")

    monkeypatch.setattr(files, "process_file_background", process)
    fid = new_id("file")
    _seed_file(db, api.email, fid)
    r = api.post("/files/ingest", json={"file_id": fid})
    assert r.status_code == 202 and r.json() == {"status": "processing_started", "file_id": fid}
    assert seen == ["worker thread"]


def test_an_ingest_crash_keeps_the_same_202(api, db, monkeypatch):
    def boom(req):
        raise RuntimeError("loader exploded")

    monkeypatch.setattr(files, "process_file_background", boom)
    fid = new_id("file")
    _seed_file(db, api.email, fid)
    r = api.post("/files/ingest", json={"file_id": fid})
    assert r.status_code == 202 and r.json() == {"status": "processing_started", "file_id": fid}
