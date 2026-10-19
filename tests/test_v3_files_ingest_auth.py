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


def _seed_file(db, owner_email, fid):
    b = seed_board(db, owner_email)
    c = seed_card(db, owner_email, b, kind="source", source_type="pdf")
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
