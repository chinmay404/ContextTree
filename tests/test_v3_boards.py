import uuid

from tests.v3_support import requires_db, db, user, api, make_board, new_id, seed_board, seed_card  # noqa: F401

pytestmark = requires_db


def _other_user(db):
    email = f"other_{uuid.uuid4().hex[:10]}@example.com"
    with db.cursor() as cur:
        cur.execute("INSERT INTO users (email, name) VALUES (%s, 'O')", (email,))
    return email


def _cleanup_other(db, email):
    with db.cursor() as cur:
        for t in ("nodes", "canvases"):
            cur.execute(f"DELETE FROM {t} WHERE user_email = %s", (email,))
        cur.execute("DELETE FROM users WHERE email = %s", (email,))


def test_create_board_returns_board_with_one_chat(api):
    bid = new_id("b")
    r = api.post("/boards", json={"id": bid, "title": "Pricing"})
    assert r.status_code == 201
    b = r.json()
    assert b["id"] == bid and b["title"] == "Pricing" and b["viewport"] is None and b["edges"] == []
    [q1] = b["cards"]
    assert q1["id"] == f"{bid}_q1" and q1["kind"] == "chat" and (q1["x"], q1["y"]) == (0, 0)
    assert q1["boardId"] == bid
    assert q1["sourceType"] is None and q1["status"] is None and q1["parentId"] is None
    d = q1["data"]
    assert d["name"] == "Pricing" and d["carry"] == "summary" and d["inheritDesk"] is True
    assert d["model"] == "openai/gpt-5-mini"  # = MODELS.gpt; a board made with no key names a model the board knows
    assert d["hide"] == [] and d["advanced"]["externalContextTopK"] == 5 and d["advanced"]["lastKMessages"] == 6
    assert q1["messageCount"] == 0 and q1["lastMessage"] is None


def test_create_board_twice_is_idempotent_200(api, db):
    bid = new_id("b")
    r1 = api.post("/boards", json={"id": bid, "title": "T"})
    r2 = api.post("/boards", json={"id": bid, "title": "T"})
    assert r1.status_code == 201 and r2.status_code == 200
    assert r1.json()["createdAt"] == r2.json()["createdAt"]
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM canvases WHERE id = %s", (bid,))
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT count(*) FROM nodes WHERE canvas_id = %s", (bid,))
        assert cur.fetchone()[0] == 1


def test_create_board_id_owned_by_other_user_is_409_duplicate(api, db):
    other = _other_user(db)
    try:
        bid = seed_board(db, other, new_id("b"))
        r = api.post("/boards", json={"id": bid, "title": "x"})
        assert r.status_code == 409 and r.json()["error"] == "duplicate"
    finally:
        _cleanup_other(db, other)


def test_create_board_id_of_own_deleted_board_or_v2_canvas_is_409(api, db):
    gone = seed_board(db, api.email, new_id("b"), deleted=True)
    v2 = seed_board(db, api.email, new_id("b"), version=None)
    for bid in (gone, v2):
        r = api.post("/boards", json={"id": bid, "title": "x"})
        assert r.status_code == 409 and r.json()["error"] == "duplicate"


def test_bad_id_is_400_invalid(api):
    r = api.post("/boards", json={"id": "has space", "title": "x"})
    assert r.status_code == 400
    body = r.json()
    assert body["error"] == "invalid" and isinstance(body["detail"], str) and body["detail"]
    assert set(body) == {"error", "detail"}


def test_empty_title_is_400(api):
    assert api.post("/boards", json={"id": new_id("b"), "title": ""}).status_code == 400


def test_list_boards_only_v3_live_newest_first(api, db):
    seed_board(db, api.email, new_id("b"), version=None)
    seed_board(db, api.email, new_id("b"), deleted=True)
    a = make_board(api, "A")["id"]
    b = make_board(api, "B")["id"]
    assert api.patch(f"/boards/{a}", json={"title": "A2"}).status_code == 200
    r = api.get("/boards")
    assert r.status_code == 200
    boards = r.json()["boards"]
    assert [x["id"] for x in boards] == [a, b]
    for x in boards:
        assert set(x) == {"id", "title", "updatedAt", "cardCount", "preview"}
        assert x["cardCount"] == 1
    assert boards[0]["title"] == "A2"


def test_get_board_404_for_other_user_v2_canvas_and_missing(api, db):
    other = _other_user(db)
    try:
        theirs = seed_board(db, other, new_id("b"))
        v2 = seed_board(db, api.email, new_id("b"), version=None)
        for bid in (theirs, v2, "nope_missing"):
            r = api.get(f"/boards/{bid}")
            assert r.status_code == 404 and r.json()["error"] == "not_found"
    finally:
        _cleanup_other(db, other)


def test_get_board_filters_deleted_cards(api, db):
    b = make_board(api)
    seed_card(db, api.email, b["id"], "gone", kind="chat", deleted_at="2026-01-01")
    got = api.get(f"/boards/{b['id']}").json()
    assert [c["id"] for c in got["cards"]] == [f"{b['id']}_q1"]


def test_patch_board_title_and_viewport(api, db):
    b = make_board(api)["id"]
    r = api.patch(f"/boards/{b}", json={"title": "New", "viewport": {"x": 1.5, "y": -2, "zoom": 0.8}})
    assert r.status_code == 200 and r.json() == {"ok": True}
    got = api.get(f"/boards/{b}").json()
    assert got["title"] == "New" and got["viewport"] == {"x": 1.5, "y": -2, "zoom": 0.8}
    with db.cursor() as cur:
        cur.execute("SELECT title FROM canvases WHERE id = %s", (b,))
        assert cur.fetchone()[0] == "New"
    assert api.patch(f"/boards/{b}", json={"viewport": None}).status_code == 200
    got = api.get(f"/boards/{b}").json()
    assert got["viewport"] is None and got["title"] == "New"


def test_patch_board_404_other(api):
    assert api.patch("/boards/missing", json={"title": "x"}).status_code == 404


def test_delete_board_soft_204_then_404(api, db):
    b = make_board(api)["id"]
    assert api.delete(f"/boards/{b}").status_code == 204
    with db.cursor() as cur:
        cur.execute("SELECT deleted_at IS NOT NULL FROM canvases WHERE id = %s", (b,))
        assert cur.fetchone()[0] is True
    assert api.get(f"/boards/{b}").status_code == 404
    assert b not in [x["id"] for x in api.get("/boards").json()["boards"]]
    assert api.delete(f"/boards/{b}").status_code == 404


def test_errors_use_envelope(api):
    r = api.client.get("/api/v1/boards")
    assert r.status_code == 401
    assert r.json()["error"] == "unauthorized" and "detail" in r.json()


def test_board_routes_not_capped_by_default_ip_limit(api):
    assert all(api.get("/boards").status_code == 200 for _ in range(120))


def test_inserted_rows_satisfy_prod_constraints(api, db):
    bid = make_board(api)["id"]
    with db.cursor() as cur:
        cur.execute("SELECT data IS NOT NULL, board_version FROM canvases WHERE id = %s", (bid,))
        assert cur.fetchone() == (True, 3)
        cur.execute("SELECT data IS NOT NULL, canvas_id, user_email, kind, is_primary "
                    "FROM nodes WHERE id = %s", (f"{bid}_q1",))
        assert cur.fetchone() == (True, bid, api.email, "chat", True)


# ── GET /boards preview (app-shell plan Task 8) ──────────────────────────────

import json as _json

from tests.v3_support import seed_edge, seed_message

PREVIEW_CARD_KEYS = {"id", "kind", "sourceType", "x", "y", "deskId", "pileId", "parentId"}


def _listed(api, bid):
    r = api.get("/boards")
    assert r.status_code == 200, r.text
    return next(b for b in r.json()["boards"] if b["id"] == bid)


def _cards(preview):
    return {c["id"]: c for c in preview["cards"]}


def test_list_boards_preview_has_q1(api):
    bid = make_board(api, "Pricing")["id"]
    preview = _listed(api, bid)["preview"]
    assert preview == {"cards": [{"id": f"{bid}_q1", "kind": "chat", "sourceType": None, "x": 0, "y": 0,
                                  "deskId": None, "pileId": None, "parentId": None, "name": "Pricing"}],
                       "edges": []}


def test_list_boards_preview_wire_and_pile(api, db):
    bid = seed_board(db, api.email)
    q = seed_card(db, api.email, bid, kind="chat", data={"name": "Q"}, x=10, y=20)
    note = seed_card(db, api.email, bid, kind="source", source_type="note", status="ready",
                     desk_id=q, data={"title": "N", "text": "SECRET NOTE TEXT"})
    pile = seed_card(db, api.email, bid, kind="source", source_type="note", status="ready")
    member = seed_card(db, api.email, bid, kind="source", source_type="note", status="ready",
                       pile_id=pile)
    child = seed_card(db, api.email, bid, kind="chat", parent_id=q, data={"name": "Branch"})
    seed_edge(db, api.email, bid, note, child, "wire")
    seed_edge(db, api.email, bid, q, child, "answer")
    seed_edge(db, api.email, bid, member, q, "wire")  # one end is a pile member: left out
    preview = _listed(api, bid)["preview"]
    cards = _cards(preview)
    assert set(cards) == {q, note, pile, child}  # the pile member is absent
    assert cards[q] == {"id": q, "kind": "chat", "sourceType": None, "x": 10, "y": 20, "deskId": None,
                        "pileId": None, "parentId": None, "name": "Q"}
    assert cards[note] == {"id": note, "kind": "source", "sourceType": "note", "x": 0, "y": 0,
                           "deskId": q, "pileId": None, "parentId": None}  # sources carry no name
    assert cards[child]["parentId"] == q
    assert sorted((e["kind"], e["from"], e["to"]) for e in preview["edges"]) == sorted(
        [("wire", note, child), ("answer", q, child)])
    assert all(set(e) == {"kind", "from", "to"} for e in preview["edges"])


def test_list_boards_preview_skips_deleted(api, db):
    bid = seed_board(db, api.email)
    q = seed_card(db, api.email, bid, kind="chat", data={"name": "Q"})
    gone = seed_card(db, api.email, bid, kind="chat", deleted_at="2026-01-01")
    desk = seed_card(db, api.email, bid, kind="chat", data={"name": "Desk"})
    pile = seed_card(db, api.email, bid, kind="source", source_type="note", status="ready")
    on_desk = seed_card(db, api.email, bid, kind="source", source_type="note", status="ready",
                        desk_id=desk)
    in_pile = seed_card(db, api.email, bid, kind="source", source_type="note", status="ready",
                        pile_id=pile)
    seed_edge(db, api.email, bid, q, gone, "answer")
    seed_edge(db, api.email, bid, on_desk, q, "wire", deleted_at="2026-01-01")
    with db.cursor() as cur:
        cur.execute("UPDATE nodes SET deleted_at = now() WHERE id IN (%s, %s)", (desk, pile))
    preview = _listed(api, bid)["preview"]
    cards = _cards(preview)
    assert set(cards) == {q, on_desk, in_pile}
    assert cards[on_desk]["deskId"] is None  # linked to a deleted desk reads as NULL
    assert cards[in_pile]["pileId"] is None  # a deleted pile no longer hides its members
    assert preview["edges"] == []


def test_list_boards_preview_name_truncated(api, db):
    bid = seed_board(db, api.email)
    q = seed_card(db, api.email, bid, kind="chat", data={"name": "n" * 200})
    assert _cards(_listed(api, bid)["preview"])[q]["name"] == "n" * 60


def test_list_boards_preview_only_24(api, db):
    ids = []
    for i in range(25):
        bid = seed_board(db, api.email)
        seed_card(db, api.email, bid, kind="chat", data={"name": f"B{i}"})
        with db.cursor() as cur:
            cur.execute("UPDATE canvases SET updated_at = now() - %s * interval '1 minute' WHERE id = %s",
                        (i, bid))
        ids.append(bid)
    boards = api.get("/boards").json()["boards"]
    assert [b["id"] for b in boards] == ids  # newest first
    assert all("preview" in b for b in boards[:24])
    assert "preview" not in boards[24]


def test_list_boards_preview_never_carries_content(api, db):
    bid = seed_board(db, api.email)
    q = seed_card(db, api.email, bid, kind="chat",
                  data={"name": "Q", "systemPrompt": "SECRET PROMPT", "summary": "SECRET SUMMARY"})
    seed_card(db, api.email, bid, kind="source", source_type="note", status="ready",
              data={"title": "SECRET TITLE", "text": "SECRET NOTE", "overview": "SECRET OVERVIEW"})
    seed_card(db, api.email, bid, kind="source", source_type="csv", status="ready",
              data={"title": "t.csv", "rows": [["SECRET", "ROW"]], "keyPassages": ["SECRET PASSAGE"]})
    seed_message(db, api.email, q, "user", "SECRET MESSAGE", 0)
    with db.cursor() as cur:
        cur.execute("UPDATE nodes SET summary = 'SECRET THREAD SUMMARY' WHERE id = %s", (q,))
    preview = _listed(api, bid)["preview"]
    assert "SECRET" not in _json.dumps(preview)
    for c in preview["cards"]:
        assert set(c) - {"name"} == PREVIEW_CARD_KEYS


def test_list_boards_preview_is_owner_scoped(api, db):
    other = _other_user(db)
    try:
        bid = seed_board(db, api.email)
        mine = seed_card(db, api.email, bid, kind="chat", data={"name": "Mine"})
        # A row with another owner on my board id (cannot happen through the API) never leaks.
        theirs = seed_card(db, other, bid, kind="chat", data={"name": "Theirs"})
        seed_edge(db, other, bid, mine, theirs, "answer")
        foreign = seed_board(db, other)
        seed_card(db, other, foreign, kind="chat", data={"name": "Foreign"})
        boards = api.get("/boards").json()["boards"]
        assert [b["id"] for b in boards] == [bid]
        assert set(_cards(boards[0]["preview"])) == {mine}
        assert boards[0]["preview"]["edges"] == []
    finally:
        with db.cursor() as cur:
            cur.execute("DELETE FROM edges WHERE user_email = %s", (other,))
        _cleanup_other(db, other)


def test_list_boards_preview_at_the_card_cap_is_small(api, db):
    bid = seed_board(db, api.email)
    long = "x" * 100
    ids = [seed_card(db, api.email, bid, f"{long[:90]}{i:010d}", kind="chat",
                     data={"name": "n" * 500, "systemPrompt": "p" * 4000}) for i in range(50)]
    for a, b in zip(ids, ids[1:]):
        seed_edge(db, api.email, bid, a, b, "answer")
    preview = _listed(api, bid)["preview"]
    assert len(preview["cards"]) == 50 and len(preview["edges"]) == 49
    size = len(_json.dumps(preview))
    assert size < 32_000, size  # ids ≤ 100 chars, names ≤ 60: no card or edge grows past that


def test_list_boards_preview_uses_two_queries(api, db, monkeypatch):
    from contextlib import contextmanager

    from app.agent.store.board_store import store

    for i in range(30):
        bid = seed_board(db, api.email)
        a = seed_card(db, api.email, bid, kind="chat", data={"name": f"B{i}"})
        b = seed_card(db, api.email, bid, kind="chat", data={"name": "C"})
        seed_edge(db, api.email, bid, a, b, "answer")
    seen = []
    real_tx = store.tx

    class Counting:
        def __init__(self, cur):
            self._cur = cur

        def execute(self, sql, *a, **k):
            seen.append(sql)
            return self._cur.execute(sql, *a, **k)

        def __getattr__(self, name):
            return getattr(self._cur, name)

    @contextmanager
    def counted():
        with real_tx() as cur:
            yield Counting(cur)

    monkeypatch.setattr(store, "tx", counted)
    boards = api.get("/boards").json()["boards"]
    assert len(boards) == 30 and sum("preview" in b for b in boards) == 24
    assert len(seen) == 3, seen  # the board list, then the preview cards and edges
