"""Marks: pen, arrow and highlight on the board and on note pages."""
import pytest

from app.agent.reads import resolve_reads
from app.agent.store.board_store import store
from tests.v3_support import api, db, new_id, requires_db, seed_board, seed_card, user  # noqa: F401


@pytest.fixture
def board(db, user):
    b = seed_board(db, user)
    chat = seed_card(db, user, b, kind="chat", data={"name": "Q"})
    note = seed_card(db, user, b, kind="source", source_type="note", status="ready", data={"text": "hi"})
    pdf = seed_card(db, user, b, kind="source", source_type="pdf", status="ready", data={"title": "P"})
    return dict(b=b, chat=chat, note=note, pdf=pdf, email=user)


def _mark(**over):
    return {"id": new_id("mk"), "kind": "pen", "color": "ink", "points": [[0, 0], [10, 5.5, 0.4]], **over}


def _post(api, bd, **over):
    return api.post(f"/boards/{bd['b']}/marks", json=_mark(**over))


@requires_db
def test_a_mark_is_saved_and_listed(api, board):
    r = _post(api, board, touches=[board["pdf"]])
    assert r.status_code == 201
    body = r.json()
    assert body["points"] == [[0, 0], [10, 5.5, 0.4]] and body["touches"] == [board["pdf"]]
    assert body["anchorId"] is None and body["noteId"] is None
    assert [m["id"] for m in api.get(f"/boards/{board['b']}/marks").json()["marks"]] == [body["id"]]


@requires_db
def test_saving_twice_is_one_mark(api, board):
    m = _mark()
    first = api.post(f"/boards/{board['b']}/marks", json=m)
    second = api.post(f"/boards/{board['b']}/marks", json=m)
    assert (first.status_code, second.status_code) == (201, 200)
    assert len(api.get(f"/boards/{board['b']}/marks").json()["marks"]) == 1


@requires_db
def test_erased_marks_are_gone_and_erasing_twice_is_404(api, board):
    mid = _post(api, board).json()["id"]
    assert api.delete(f"/boards/{board['b']}/marks/{mid}").status_code == 204
    assert api.get(f"/boards/{board['b']}/marks").json()["marks"] == []
    assert api.delete(f"/boards/{board['b']}/marks/{mid}").status_code == 404


@requires_db
def test_a_mark_goes_and_comes_back_with_its_card(api, board, db):
    anchored = _post(api, board, anchorId=board["pdf"]).json()["id"]
    on_note = _post(api, board, noteId=board["note"]).json()["id"]
    loose = _post(api, board).json()["id"]
    with db.cursor() as cur:
        cur.execute("UPDATE nodes SET deleted_at = now() WHERE id = ANY(%s)", ([board["pdf"], board["note"]],))
    assert [m["id"] for m in api.get(f"/boards/{board['b']}/marks").json()["marks"]] == [loose]
    with db.cursor() as cur:
        cur.execute("UPDATE nodes SET deleted_at = NULL WHERE id = ANY(%s)", ([board["pdf"], board["note"]],))
    assert {m["id"] for m in api.get(f"/boards/{board['b']}/marks").json()["marks"]} == {anchored, on_note, loose}


@requires_db
def test_note_id_must_be_a_note(api, board):
    assert _post(api, board, noteId=board["pdf"]).status_code == 400
    assert _post(api, board, noteId="nope").status_code == 404
    assert _post(api, board, anchorId="nope").status_code == 404


@requires_db
@pytest.mark.parametrize("over", [
    {"kind": "circle"},
    {"color": "#123456"},
    {"size": 0},
    {"size": 1000},
    {"points": []},
    {"points": [[1]]},
    {"points": [[1, 2, 3, 4]]},
    {"points": [["a", 2]]},
    {"points": [[0, 0]] * 5001},
    {"touches": ["x"] * 51},
])
def test_bad_marks_are_refused(api, board, over):
    assert _post(api, board, **over).status_code == 400


@requires_db
def test_another_users_board_is_not_found(api, db, board):
    other = f"o_{new_id('u')}@example.com"
    with db.cursor() as cur:
        cur.execute("INSERT INTO users (email, name) VALUES (%s, 'x')", (other,))
    b = seed_board(db, other)
    try:
        assert api.get(f"/boards/{b}/marks").status_code == 404
        assert api.post(f"/boards/{b}/marks", json=_mark()).status_code == 404
    finally:
        with db.cursor() as cur:
            cur.execute("DELETE FROM canvases WHERE id = %s", (b,))
            cur.execute("DELETE FROM users WHERE email = %s", (other,))


@requires_db
def test_a_mark_id_of_another_board_is_refused(api, board, db):
    mid = _post(api, board).json()["id"]
    b2 = seed_board(db, board["email"])
    assert api.post(f"/boards/{b2}/marks", json=_mark(id=mid)).status_code == 409


@requires_db
def test_the_ai_reads_none_of_this(api, board):
    before = resolve_reads(store, board["email"], board["b"], board["chat"]).to_json()
    _post(api, board, anchorId=board["chat"], touches=[board["chat"]])
    assert resolve_reads(store, board["email"], board["b"], board["chat"]).to_json() == before


@requires_db
def test_tables_are_closed_to_supabases_public_api(db):
    with db.cursor() as cur:
        cur.execute("SELECT relname, relrowsecurity FROM pg_class WHERE relname = ANY(%s)",
                    (["board_marks", "summary_blocks", "context_traces"],))
        assert dict(cur.fetchall()) == {"board_marks": True, "summary_blocks": True, "context_traces": True}
