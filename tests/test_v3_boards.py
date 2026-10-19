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
    assert d["model"] == "openai/gpt-oss-120b"
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
        assert set(x) == {"id", "title", "updatedAt", "cardCount"}
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
