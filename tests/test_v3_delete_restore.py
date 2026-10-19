import pytest

from tests.v3_support import requires_db, db, user, api, make_board, new_id, seed_card, seed_edge  # noqa: F401

pytestmark = requires_db


class T:
    pass


def _post(api, b, **kw):
    body = {"id": new_id("c"), "x": 0, "y": 0, "data": {}}
    body.update(kw)
    r = api.post(f"/boards/{b}/cards", json=body)
    assert r.status_code == 201, r.text
    return body["id"]


@pytest.fixture
def tree(api, db, user):
    t = T()
    board = make_board(api)
    t.b = board["id"]
    q1 = [c for c in board["cards"] if c["kind"] == "chat"][0]["id"]
    t.q1 = q1
    t.c2 = _post(api, t.b, kind="chat", parentId=q1)
    t.c3 = _post(api, t.b, kind="chat", parentId=t.c2)
    t.s = _post(api, t.b, kind="source", sourceType="note", deskId=t.c2)
    t.s2 = _post(api, t.b, kind="source", sourceType="note")
    t.w = seed_edge(db, user, t.b, t.s2, t.c3, "wire")
    t.p = _post(api, t.b, kind="source", sourceType="pile")
    t.m1 = seed_card(db, user, t.b, kind="source", source_type="note", status="ready", pile_id=t.p)
    t.m2 = seed_card(db, user, t.b, kind="source", source_type="note", status="ready", pile_id=t.p)
    return t


def _board(api, b):
    r = api.get(f"/boards/{b}")
    assert r.status_code == 200, r.text
    j = r.json()
    return {c["id"]: c for c in j["cards"]}, {e["id"] for e in j["edges"]}


def test_delete_chat_deletes_subtree_and_strings_not_sources(api, db, tree):
    t = tree
    r = api.delete(f"/boards/{t.b}/cards/{t.c2}")
    assert r.status_code == 200 and r.json() == {"deleted": sorted([t.c2, t.c3])}
    cards, edges = _board(api, t.b)
    assert t.c2 not in cards and t.c3 not in cards
    assert t.s in cards and cards[t.s]["deskId"] is None
    assert t.s2 in cards
    assert t.w not in edges and "lin_" + t.c2 not in edges and "lin_" + t.c3 not in edges
    with db.cursor() as cur:
        cur.execute("SELECT desk_id, deleted_at FROM nodes WHERE id = %s", (t.s,))
        desk, sdel = cur.fetchone()
        assert desk == t.c2 and sdel is None
        cur.execute("SELECT deleted_at FROM nodes WHERE id IN (%s,%s)", (t.c2, t.c3))
        stamps = {r[0] for r in cur.fetchall()}
        cur.execute("SELECT deleted_at FROM edges WHERE id = %s", (t.w,))
        stamps.add(cur.fetchone()[0])
        assert len(stamps) == 1 and None not in stamps


def test_restore_undoes_exactly_the_set(api, tree):
    t = tree
    api.delete(f"/boards/{t.b}/cards/{t.c2}")
    r = api.post(f"/boards/{t.b}/cards/{t.c2}/restore")
    assert r.status_code == 200 and r.json() == {"restored": sorted([t.c2, t.c3])}
    cards, edges = _board(api, t.b)
    assert t.c2 in cards and t.c3 in cards
    assert t.w in edges and "lin_" + t.c2 in edges
    assert cards[t.s]["deskId"] == t.c2


def test_restore_does_not_resurrect_earlier_deletes(api, tree):
    t = tree
    assert api.delete(f"/boards/{t.b}/cards/{t.c3}").status_code == 200
    assert api.delete(f"/boards/{t.b}/cards/{t.c2}").json() == {"deleted": [t.c2]}
    r = api.post(f"/boards/{t.b}/cards/{t.c2}/restore")
    assert r.json() == {"restored": [t.c2]}
    cards, _ = _board(api, t.b)
    assert t.c2 in cards and t.c3 not in cards


def test_restore_child_under_deleted_parent_is_409(api, tree):
    t = tree
    api.delete(f"/boards/{t.b}/cards/{t.c3}")
    api.delete(f"/boards/{t.b}/cards/{t.c2}")
    r = api.post(f"/boards/{t.b}/cards/{t.c3}/restore")
    assert r.status_code == 409 and r.json()["error"] == "not_allowed"


def test_restore_live_card_is_noop(api, tree):
    t = tree
    r = api.post(f"/boards/{t.b}/cards/{t.c2}/restore")
    assert r.status_code == 200 and r.json() == {"restored": []}


def test_delete_pile_members_show_loose(api, tree):
    t = tree
    r = api.delete(f"/boards/{t.b}/cards/{t.p}")
    assert r.status_code == 200 and r.json() == {"deleted": [t.p]}
    cards, _ = _board(api, t.b)
    assert cards[t.m1]["pileId"] is None and cards[t.m2]["pileId"] is None
    assert api.post(f"/boards/{t.b}/cards/{t.p}/restore").json() == {"restored": [t.p]}
    cards, _ = _board(api, t.b)
    assert cards[t.m1]["pileId"] == t.p


def test_deleted_card_patch_and_messages_404(api, tree):
    t = tree
    api.delete(f"/boards/{t.b}/cards/{t.c2}")
    assert api.patch(f"/boards/{t.b}/cards/{t.c2}", json={"data": {"a": 1}}).status_code == 404
    assert api.get(f"/boards/{t.b}/cards/{t.c2}/messages").status_code == 404


def test_delete_twice_404(api, tree):
    t = tree
    assert api.delete(f"/boards/{t.b}/cards/{t.s}").status_code == 200
    assert api.delete(f"/boards/{t.b}/cards/{t.s}").status_code == 404


def test_restore_unknown_card_404(api, tree):
    assert api.post(f"/boards/{tree.b}/cards/nope/restore").status_code == 404


def test_restore_over_limit_409(api, db, user):
    board = make_board(api)
    b = board["id"]
    q1 = board["cards"][0]["id"]
    a = _post(api, b, kind="chat", parentId=q1)
    a2 = _post(api, b, kind="chat", parentId=a)
    assert api.delete(f"/boards/{b}/cards/{a}").json() == {"deleted": sorted([a, a2])}
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM nodes WHERE canvas_id = %s AND kind IS NOT NULL AND deleted_at IS NULL", (b,))
        live = cur.fetchone()[0]
    for _ in range(50 - live - 1):
        seed_card(db, user, b, kind="source", source_type="note", status="ready")
    seed_card(db, user, b, kind="source", source_type="note", status="ready")
    seed_card(db, user, b, kind="source", source_type="note", status="ready")
    r = api.post(f"/boards/{b}/cards/{a}/restore")
    assert r.status_code == 409 and r.json()["error"] == "card_limit"
