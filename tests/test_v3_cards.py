from contextlib import contextmanager

from app.schemas.board import DEFAULT_ADVANCED, DEFAULT_MODEL
from tests.v3_support import requires_db, db, user, api, make_board, new_id, seed_card, seed_message, seed_board  # noqa: F401

pytestmark = requires_db


def _src(api, bid, st="note", **kw):
    body = {"id": new_id("c"), "kind": "source", "sourceType": st, "x": 1, "y": 2, "data": {}}
    body.update(kw)
    r = api.post(f"/boards/{bid}/cards", json=body)
    assert r.status_code == 201, r.text
    return r.json()


@contextmanager
def other_user(db):
    other = f"{new_id('o')}@example.com"
    with db.cursor() as cur:
        cur.execute("INSERT INTO users (email, name) VALUES (%s, 'o')", (other,))
    try:
        yield other
    finally:
        with db.cursor() as cur:
            for t in ("messages", "edges", "nodes", "canvases"):
                cur.execute(f"DELETE FROM {t} WHERE user_email = %s", (other,))
            cur.execute("DELETE FROM users WHERE email = %s", (other,))


def test_create_note_is_ready(api):
    b = make_board(api)
    r = api.post(f"/boards/{b['id']}/cards", json={"id": new_id("c"), "kind": "source",
        "sourceType": "note", "x": 10, "y": 20, "data": {"text": "Costs ≈ $0.40"}})
    assert r.status_code == 201 and r.json()["status"] == "ready" and r.json()["data"]["text"] == "Costs ≈ $0.40"


def test_create_non_note_source_is_processing(api):
    b = make_board(api)
    assert _src(api, b["id"], "pdf")["status"] == "processing"


def test_create_card_idempotent(api, db):
    b = make_board(api)
    body = {"id": new_id("c"), "kind": "source", "sourceType": "note", "x": 0, "y": 0, "data": {}}
    assert api.post(f"/boards/{b['id']}/cards", json=body).status_code == 201
    r = api.post(f"/boards/{b['id']}/cards", json=body)
    assert r.status_code == 200 and r.json()["id"] == body["id"]
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM nodes WHERE id = %s", (body["id"],))
        assert cur.fetchone()[0] == 1


def test_duplicate_id_of_other_user_is_409(api, db):
    b = make_board(api)
    with other_user(db) as other:
        ob = seed_board(db, other)
        cid = seed_card(db, other, ob, kind="source", source_type="note", status="ready")
        r = api.post(f"/boards/{b['id']}/cards", json={"id": cid, "kind": "source",
                     "sourceType": "note", "x": 0, "y": 0, "data": {}})
        assert r.status_code == 409 and r.json()["error"] == "duplicate"


def test_chat_defaults_filled(api):
    b = make_board(api)
    r = api.post(f"/boards/{b['id']}/cards", json={"id": new_id("c"), "kind": "chat", "x": 0, "y": 0,
                 "data": {"name": "Q", "advanced": {"lastKMessages": 3}, "memoryFacts": ["x"]}})
    assert r.status_code == 201
    d = r.json()
    assert d["data"]["name"] == "Q" and d["data"]["inheritDesk"] is True
    assert d["data"]["carry"] == "summary" and d["data"]["hide"] == [] and d["data"]["model"] == DEFAULT_MODEL
    assert d["data"]["advanced"] == {**DEFAULT_ADVANCED, "lastKMessages": 3}
    assert "memoryFacts" not in d["data"] and d["status"] is None


def test_chat_with_source_type_is_400(api):
    b = make_board(api)
    r = api.post(f"/boards/{b['id']}/cards", json={"id": new_id("c"), "kind": "chat",
                 "sourceType": "pdf", "x": 0, "y": 0, "data": {}})
    assert r.status_code == 400 and r.json()["error"] == "invalid"


def test_source_without_source_type_is_400(api):
    b = make_board(api)
    r = api.post(f"/boards/{b['id']}/cards", json={"id": new_id("c"), "kind": "source", "x": 0, "y": 0})
    assert r.status_code == 400 and r.json()["error"] == "invalid"


def test_create_chat_with_parent_sets_lineage(api, db):
    b = make_board(api); bid = b["id"]; q1 = f"{bid}_q1"; cid = new_id("c")
    r = api.post(f"/boards/{bid}/cards", json={"id": cid, "kind": "chat", "x": 0, "y": 0,
                 "parentId": q1, "forkedFromMessageId": "m1_a", "data": {}})
    assert r.status_code == 201 and r.json()["parentId"] == q1
    with db.cursor() as cur:
        cur.execute("SELECT parent_node_id, forked_from_message_id, ancestor_ids, is_initialized, "
                    "is_primary FROM nodes WHERE id = %s", (cid,))
        assert cur.fetchone() == (q1, "m1_a", [q1], False, False)
    edges = api.get(f"/boards/{bid}").json()["edges"]
    assert any(e["id"] == f"lin_{cid}" and e["kind"] == "lineage" and e["from"] == q1 and e["to"] == cid
               for e in edges)
    gid = new_id("c")
    api.post(f"/boards/{bid}/cards", json={"id": gid, "kind": "chat", "x": 0, "y": 0, "parentId": cid, "data": {}})
    with db.cursor() as cur:
        cur.execute("SELECT ancestor_ids FROM nodes WHERE id = %s", (gid,))
        assert cur.fetchone()[0] == [q1, cid]


def test_root_chat_is_primary(api, db):
    b = make_board(api); cid = new_id("c")
    api.post(f"/boards/{b['id']}/cards", json={"id": cid, "kind": "chat", "x": 0, "y": 0, "data": {}})
    with db.cursor() as cur:
        cur.execute("SELECT is_primary FROM nodes WHERE id = %s", (cid,))
        assert cur.fetchone()[0] is True


def test_parent_on_other_board_is_404(api):
    b1 = make_board(api); b2 = make_board(api)
    r = api.post(f"/boards/{b2['id']}/cards", json={"id": new_id("c"), "kind": "chat", "x": 0, "y": 0,
                 "parentId": f"{b1['id']}_q1", "data": {}})
    assert r.status_code == 404


def test_parent_must_be_chat(api):
    b = make_board(api); note = _src(api, b["id"])
    r = api.post(f"/boards/{b['id']}/cards", json={"id": new_id("c"), "kind": "chat", "x": 0, "y": 0,
                 "parentId": note["id"], "data": {}})
    assert r.status_code == 400


def test_card_limit_409_at_50(api, db, user):
    b = make_board(api); bid = b["id"]
    for _ in range(49):
        seed_card(db, user, bid, kind="source", source_type="note", status="ready")
    r = api.post(f"/boards/{bid}/cards", json={"id": new_id("c"), "kind": "source", "sourceType": "note",
                 "x": 0, "y": 0, "data": {}})
    assert r.status_code == 409 and r.json()["error"] == "card_limit"


def test_patch_merges_data_keys(api):
    b = make_board(api); c = _src(api, b["id"], data={"title": "A", "desc": "x"})
    r = api.patch(f"/boards/{b['id']}/cards/{c['id']}", json={"data": {"desc": "y"}})
    assert r.status_code == 200 and r.json()["data"] == {"title": "A", "desc": "y"}


def test_patch_desk_must_be_live_chat_on_board(api):
    b = make_board(api); bid = b["id"]; q1 = f"{bid}_q1"; other = make_board(api)
    note = _src(api, bid); src = _src(api, bid, "pdf")
    url = f"/boards/{bid}/cards/{src['id']}"
    assert api.patch(url, json={"deskId": note["id"]}).status_code == 400
    assert api.patch(url, json={"deskId": f"{other['id']}_q1"}).status_code == 400
    assert api.patch(url, json={"deskId": "bad id!"}).status_code == 400
    r = api.patch(url, json={"deskId": q1})
    assert r.status_code == 200 and r.json()["deskId"] == q1
    r = api.patch(url, json={"deskId": None})
    assert r.status_code == 200 and r.json()["deskId"] is None


def test_patch_absent_key_leaves_desk(api):
    b = make_board(api); bid = b["id"]; q1 = f"{bid}_q1"
    c = _src(api, bid, deskId=q1)
    r = api.patch(f"/boards/{bid}/cards/{c['id']}", json={"data": {"a": 1}})
    assert r.json()["deskId"] == q1


def test_patch_pile_must_be_live_pile(api):
    b = make_board(api); bid = b["id"]
    note = _src(api, bid); pile = _src(api, bid, "pile"); src = _src(api, bid, "pdf")
    url = f"/boards/{bid}/cards/{src['id']}"
    assert api.patch(url, json={"pileId": note["id"]}).status_code == 400
    r = api.patch(url, json={"pileId": pile["id"]})
    assert r.status_code == 200 and r.json()["pileId"] == pile["id"]


def test_patch_other_users_card_404(api, db):
    b = make_board(api)
    with other_user(db) as other:
        ob = seed_board(db, other)
        cid = seed_card(db, other, ob, kind="source", source_type="note", status="ready")
        assert api.patch(f"/boards/{ob}/cards/{cid}", json={"data": {"a": 1}}).status_code == 404
        assert api.patch(f"/boards/{b['id']}/cards/{cid}", json={"data": {"a": 1}}).status_code == 404


def _xy(api, bid):
    return {c["id"]: (c["x"], c["y"]) for c in api.get(f"/boards/{bid}").json()["cards"]}


def test_positions_batch_updates_and_ignores_unknown(api):
    b = make_board(api); bid = b["id"]; a = _src(api, bid); c = _src(api, bid)
    r = api.patch(f"/boards/{bid}/positions", json={"positions": [
        {"id": a["id"], "x": 5, "y": 6}, {"id": c["id"], "x": 7.5, "y": 8}, {"id": "nope", "x": 1, "y": 1}]})
    assert r.status_code == 200 and r.json() == {"updated": 2}
    xy = _xy(api, bid)
    assert xy[a["id"]] == (5, 6) and xy[c["id"]] == (7.5, 8)


def test_positions_over_200_is_400(api):
    b = make_board(api)
    r = api.patch(f"/boards/{b['id']}/positions",
                  json={"positions": [{"id": f"c{i}", "x": 0, "y": 0} for i in range(201)]})
    assert r.status_code == 400 and r.json()["error"] == "invalid"


def test_positions_non_finite_is_400(api):
    b = make_board(api); bid = b["id"]; cid = _src(api, bid)["id"]
    for raw in ('NaN', 'Infinity', '1e309'):
        r = api.patch(f"/boards/{bid}/positions", content=f'{{"positions":[{{"id":"{cid}","x":{raw},"y":0}}]}}',
                      headers={**api.h, "Content-Type": "application/json"})
        assert r.status_code == 400 and r.json()["error"] == "invalid"


def test_positions_other_users_cards_untouched(api, db):
    b = make_board(api)
    with other_user(db) as other:
        ob = seed_board(db, other)
        cid = seed_card(db, other, ob, kind="source", source_type="note", status="ready", x=1, y=1)
        r = api.patch(f"/boards/{b['id']}/positions", json={"positions": [{"id": cid, "x": 99, "y": 99}]})
        assert r.status_code == 200 and r.json() == {"updated": 0}
        assert api.patch(f"/boards/{ob}/positions", json={"positions": []}).status_code == 404
        with db.cursor() as cur:
            cur.execute("SELECT x, y FROM nodes WHERE id = %s", (cid,))
            assert cur.fetchone() == (1, 1)


def test_messages_visible_only_ordered(api, db, user):
    b = make_board(api); bid = b["id"]; q1 = f"{bid}_q1"
    seed_message(db, user, q1, "user", "old", -5, inherited=True)
    seed_message(db, user, q1, "assistant", "a", 2, msg_id="x_a")
    seed_message(db, user, q1, "user", "u", 1, msg_id="x_u")
    msgs = api.get(f"/boards/{bid}/cards/{q1}/messages").json()["messages"]
    assert [m["id"] for m in msgs] == ["x_u", "x_a"]
    assert set(msgs[0]) == {"id", "role", "content", "position", "createdAt"}


def test_messages_of_source_is_400(api):
    b = make_board(api); n = _src(api, b["id"])
    r = api.get(f"/boards/{b['id']}/cards/{n['id']}/messages")
    assert r.status_code == 400 and r.json()["error"] == "invalid"
