import pytest

from tests.v3_support import requires_db, db, user, api, make_board, new_id, seed_board, seed_card, seed_edge  # noqa: F401

pytestmark = requires_db


class T:
    pass


@pytest.fixture
def g(api, db, user):
    t = T()
    t.b = seed_board(db, user)
    t.q1 = seed_card(db, user, t.b, kind="chat")
    t.a = seed_card(db, user, t.b, kind="chat", parent_id=t.q1)
    t.s1 = seed_card(db, user, t.b, kind="chat", parent_id=t.q1)
    t.src = seed_card(db, user, t.b, kind="source", source_type="note", status="ready")
    t.src_desk = seed_card(db, user, t.b, kind="source", source_type="note", status="ready", desk_id=t.a)
    return t


def _post(api, g, kind, frm, to, eid=None):
    return api.post(f"/boards/{g.b}/edges",
                    json={"id": eid or new_id("e"), "kind": kind, "from": frm, "to": to})


def test_wire_source_to_chat_201(api, g):
    eid = new_id("e")
    r = _post(api, g, "wire", g.src, g.a, eid)
    assert r.status_code == 201, r.text
    j = r.json()
    assert set(j) == {"id", "boardId", "kind", "from", "to", "createdAt"}
    assert (j["id"], j["boardId"], j["kind"], j["from"], j["to"]) == (eid, g.b, "wire", g.src, g.a)


def test_wire_from_source_on_same_desk_409(api, g):
    r = _post(api, g, "wire", g.src_desk, g.a)
    assert r.status_code == 409 and r.json()["error"] == "not_allowed"


def test_wire_wrong_kinds_409(api, g):
    for frm, to in ((g.a, g.s1), (g.src, g.src_desk)):
        r = _post(api, g, "wire", frm, to)
        assert r.status_code == 409 and r.json()["error"] == "not_allowed"


def test_answer_between_siblings_201(api, g):
    r = _post(api, g, "answer", g.a, g.s1)
    assert r.status_code == 201, r.text
    assert r.json()["kind"] == "answer"


def test_answer_to_ancestor_or_descendant_409(api, g):
    for frm, to in ((g.a, g.q1), (g.q1, g.a)):
        r = _post(api, g, "answer", frm, to)
        assert r.status_code == 409 and r.json()["error"] == "not_allowed"


def test_answer_to_self_409(api, g):
    r = _post(api, g, "answer", g.a, g.a)
    assert r.status_code == 409 and r.json()["error"] == "not_allowed"


def test_duplicate_kind_from_to_409(api, g):
    assert _post(api, g, "wire", g.src, g.a).status_code == 201
    r = _post(api, g, "wire", g.src, g.a)
    assert r.status_code == 409 and r.json()["error"] == "duplicate"


def test_same_id_idempotent_200(api, g):
    eid = new_id("e")
    assert _post(api, g, "wire", g.src, g.a, eid).status_code == 201
    r = _post(api, g, "wire", g.src, g.a, eid)
    assert r.status_code == 200 and r.json()["id"] == eid


def test_lineage_kind_rejected_400(api, g):
    r = _post(api, g, "lineage", g.q1, g.a)
    assert r.status_code == 400 and r.json()["error"] == "invalid"


def test_edge_to_card_on_other_board_404(api, db, user, g):
    other = seed_board(db, user)
    oc = seed_card(db, user, other, kind="chat")
    r = _post(api, g, "wire", g.src, oc)
    assert r.status_code == 404 and r.json()["error"] == "not_found"


def test_delete_edge_204_soft(api, db, g):
    eid = new_id("e")
    assert _post(api, g, "wire", g.src, g.a, eid).status_code == 201
    r = api.delete(f"/boards/{g.b}/edges/{eid}")
    assert r.status_code == 204
    with db.cursor() as cur:
        cur.execute("SELECT deleted_at FROM edges WHERE id = %s", (eid,))
        assert cur.fetchone()[0] is not None
    assert eid not in [e["id"] for e in api.get(f"/boards/{g.b}").json()["edges"]]
    assert _post(api, g, "wire", g.src, g.a).status_code == 201


def test_delete_lineage_edge_400(api, db, user, g):
    lid = seed_edge(db, user, g.b, g.q1, g.a, "lineage")
    r = api.delete(f"/boards/{g.b}/edges/{lid}")
    assert r.status_code == 400 and r.json()["error"] == "invalid"


def test_delete_other_users_edge_404(api, db, g):
    with db.cursor() as cur:
        cur.execute("INSERT INTO users (email, name) VALUES ('v3_other_edges@example.com','x') ON CONFLICT DO NOTHING")
    try:
        ob = seed_board(db, "v3_other_edges@example.com")
        oa = seed_card(db, "v3_other_edges@example.com", ob, kind="chat")
        os_ = seed_card(db, "v3_other_edges@example.com", ob, kind="source", source_type="note", status="ready")
        oe = seed_edge(db, "v3_other_edges@example.com", ob, os_, oa, "wire")
        assert api.delete(f"/boards/{ob}/edges/{oe}").status_code == 404
        assert api.delete(f"/boards/{g.b}/edges/{oe}").status_code == 404
    finally:
        with db.cursor() as cur:
            for t in ("edges", "nodes", "canvases"):
                cur.execute(f"DELETE FROM {t} WHERE user_email = 'v3_other_edges@example.com'")
            cur.execute("DELETE FROM users WHERE email = 'v3_other_edges@example.com'")


def test_delete_edge_takes_board_lock_and_keeps_existing_stamp(db, user, g):
    # delete_edge must serialise with delete_card (board lock) and never overwrite its stamp
    import threading
    import time
    from app.agent.store import edge_store
    from app.agent.store.board_store import lock_board, store
    from app.api.errors import ApiError

    eid = seed_edge(db, user, g.b, g.src, g.a, "wire")
    outcome = {}

    def run():
        try:
            with store.tx() as cur:
                edge_store.delete_edge(cur, user, g.b, eid)
            outcome["r"] = "deleted"
        except ApiError as e:
            outcome["r"] = e.status_code

    with store.tx() as holder:
        lock_board(holder, g.b)
        t = threading.Thread(target=run)
        t.start()
        time.sleep(0.5)
        blocked = "r" not in outcome
        with db.cursor() as cur:  # delete_card stamps the edge while the lock is held
            cur.execute("UPDATE edges SET deleted_at = '2026-01-01' WHERE id = %s", (eid,))
    t.join(5)
    assert blocked, "delete_edge did not wait for the board lock"
    assert outcome["r"] == 404
    with db.cursor() as cur:
        cur.execute("SELECT deleted_at::date::text FROM edges WHERE id = %s", (eid,))
        assert cur.fetchone()[0] == "2026-01-01"
