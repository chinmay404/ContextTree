"""V3 chat turns: the card must exist, the server read set replaces contextNodeIds."""
from types import SimpleNamespace

import pytest

from tests.v3_support import (api, db, user, new_id, requires_db, seed_board, seed_card, TEST_DB)

pytestmark = requires_db
_ = (api, db, user)

STREAM = "/chat/stream"
SYNC = "/chat/"
OTHER = "other-user@example.com"


def body(node, **extra):
    return {"message": "hi", "message_id": new_id("m"), "nodeId": node,
            "model": "z-ai/glm-5.2", **extra}


@pytest.fixture
def env(monkeypatch):
    from app.agent.store.PostgresStore import PostgresConversationStore
    from app.api.v1.endpoints import chat

    captured, forks = {}, []

    async def fake(**kw):
        captured.update(kw)
        yield "data: [DONE]\n\n"

    def fake_sync(**kw):
        captured.update(kw)
        return "ok"

    def rec(msg, graph, transport, user_id, carry="summary"):
        forks.append({"parent": msg.parent_thread_id, "fork_at": msg.fork_at_message_id,
                      "carry": carry, "transport": transport})

    monkeypatch.setattr(chat, "graph", SimpleNamespace(
        mongo_store=PostgresConversationStore(db_url=TEST_DB),
        get_stream_response=fake, get_response=fake_sync))
    monkeypatch.setattr(chat, "graph_init", True)
    monkeypatch.setattr(chat, "validate_model_access", lambda *a, **k: None)
    monkeypatch.setattr(chat, "_init_fork_if_needed", rec)
    return SimpleNamespace(captured=captured, forks=forks)


def count(db, node):
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM nodes WHERE id = %s", (node,))
        return cur.fetchone()[0]


def other_user(db):
    with db.cursor() as cur:
        cur.execute("INSERT INTO users (email) VALUES (%s) ON CONFLICT DO NOTHING", (OTHER,))
    return OTHER


def test_v3_unknown_card_with_board_id_is_404_and_creates_nothing(api, db, user, env):
    bid = seed_board(db, user)
    r = api.post(STREAM, json=body("c_missing_x", boardId=bid))
    assert r.status_code == 404 and r.json()["error"] == "not_found"
    assert count(db, "c_missing_x") == 0
    assert env.captured == {}


def test_v3_deleted_card_is_404(api, db, user, env):
    bid = seed_board(db, user)
    c = seed_card(db, user, bid, kind="chat")
    with db.cursor() as cur:
        cur.execute("UPDATE nodes SET deleted_at = now() WHERE id = %s", (c,))
    for extra in ({"boardId": bid}, {}):
        r = api.post(STREAM, json=body(c, **extra))
        assert r.status_code == 404 and r.json()["error"] == "not_found"


def test_v3_other_users_card_is_404_not_403(api, db, user, env):
    other = other_user(db)
    bid = seed_board(db, other)
    c = seed_card(db, other, bid, kind="chat")
    r = api.post(STREAM, json=body(c, boardId=bid))
    assert r.status_code == 404 and r.json()["error"] == "not_found"


def test_v3_card_on_other_board_is_404(api, db, user, env):
    b1, b2 = seed_board(db, user), seed_board(db, user)
    c = seed_card(db, user, b1, kind="chat")
    r = api.post(STREAM, json=body(c, boardId=b2))
    assert r.status_code == 404 and r.json()["error"] == "not_found"


def test_v3_board_id_on_non_v3_or_deleted_board_is_404(api, db, user, env):
    v2 = seed_board(db, user, version=2)
    with db.cursor() as cur:
        cur.execute("INSERT INTO nodes (id, canvas_id, user_email, data) VALUES (%s,%s,%s,'{}'::jsonb)",
                    ("v2n_" + new_id("n"), v2, user))
        cur.execute("SELECT id FROM nodes WHERE canvas_id = %s", (v2,))
        v2n = cur.fetchone()[0]
    r = api.post(STREAM, json=body(v2n, boardId=v2))
    assert r.status_code == 404
    dead = seed_board(db, user, deleted=True)
    c = seed_card(db, user, dead, kind="chat")
    r = api.post(STREAM, json=body(c, boardId=dead))
    assert r.status_code == 404


def test_v3_source_card_is_400(api, db, user, env):
    bid = seed_board(db, user)
    c = seed_card(db, user, bid, kind="source", source_type="note", status="ready")
    r = api.post(STREAM, json=body(c, boardId=bid))
    assert r.status_code == 400 and r.json()["error"] == "invalid"


def test_v3_without_board_id_detected_by_kind(api, db, user, env):
    bid = seed_board(db, user)
    c = seed_card(db, user, bid, kind="chat")
    r = api.post(STREAM, json=body(c))
    assert r.status_code == 200
    assert env.captured["board_reads"] is not None


def test_v3_ignores_context_node_ids_and_passes_reads(api, db, user, env):
    bid = seed_board(db, user)
    c = seed_card(db, user, bid, kind="chat")
    r = api.post(STREAM, json=body(c, boardId=bid, contextNodeIds=["x"]))
    assert r.status_code == 200
    assert env.captured["context_node_ids"] is None
    assert env.captured["board_reads"].chat_id == c
    assert env.captured["user_email"] == user


def test_v3_card_row_beats_body_parent(api, db, user, env):
    bid = seed_board(db, user)
    parent = seed_card(db, user, bid, kind="chat")
    other = seed_card(db, user, bid, kind="chat")
    child = seed_card(db, user, bid, kind="chat", parent_id=parent)
    r = api.post(STREAM, json=body(child, boardId=bid, parentNodeId=other))
    assert r.status_code == 200
    assert env.forks[0]["parent"] == parent


@pytest.mark.parametrize("route", [STREAM, SYNC])
def test_v3_carry_and_budget_passed(api, db, user, env, route):
    bid = seed_board(db, user)
    c = seed_card(db, user, bid, kind="chat", data={
        "carry": "whole", "advanced": {"contextBudgetTokens": 1500, "externalContextTopK": 3}})
    r = api.post(route, json=body(c, boardId=bid, externalContextTopK=9))
    assert r.status_code == 200, r.text
    assert env.forks[0]["carry"] == "whole"
    assert env.captured["budget_tokens"] == 1500
    assert env.captured["last_k_messages"] == 50
    assert env.captured["external_context_top_k"] == 3


@pytest.mark.parametrize("route", [STREAM, SYNC])
def test_v3_non_whole_carry_keeps_body_last_k(api, db, user, env, route):
    bid = seed_board(db, user)
    c = seed_card(db, user, bid, kind="chat", data={"carry": "fresh"})
    r = api.post(route, json=body(c, boardId=bid, lastKMessages=7))
    assert r.status_code == 200, r.text
    assert env.forks[0]["carry"] == "fresh"
    assert env.captured["last_k_messages"] == 7


@pytest.mark.parametrize("route", [STREAM, SYNC])
def test_v2_chat_without_board_id_unchanged(api, db, user, env, route):
    v2 = seed_board(db, user, version=2)
    nid = new_id("n")
    with db.cursor() as cur:
        cur.execute("INSERT INTO nodes (id, canvas_id, user_email, data) VALUES (%s,%s,%s,'{}'::jsonb)",
                    (nid, v2, user))
    r = api.post(route, json=body(nid, contextNodeIds=["f1"]))
    assert r.status_code == 200, r.text
    assert env.captured["context_node_ids"] == ["f1"]
    assert env.captured["board_reads"] is None
    assert env.forks[0]["carry"] == "summary"


def test_v2_new_node_id_still_reaches_graph(api, db, user, env):
    nid = new_id("n")
    r = api.post(STREAM, json=body(nid))
    assert r.status_code == 200
    assert env.captured["context_node_ids"] is None and env.captured["board_reads"] is None
    assert env.captured["thread_id"] == nid
    assert count(db, nid) == 0  # the (faked) graph creates it, the gate never does


@pytest.mark.parametrize("route", [STREAM, SYNC])
def test_v3_other_users_card_without_board_id_is_404(api, db, user, env, route):
    other = other_user(db)
    bid = seed_board(db, other)
    c = seed_card(db, other, bid, kind="chat")
    r = api.post(route, json=body(c))
    assert r.status_code == 404 and r.json()["error"] == "not_found"
    assert env.captured == {}


def test_v3_gate_never_holds_two_connections(api, db, user, env, monkeypatch):
    from contextlib import contextmanager
    from app.agent.store.board_store import store as bs

    real = bs._get_conn
    state = {"open": 0, "max": 0, "email_open": None}

    @contextmanager
    def tracked():
        state["open"] += 1
        state["max"] = max(state["max"], state["open"])
        try:
            with real() as conn:
                yield conn
        finally:
            state["open"] -= 1

    real_resolve = type(bs).resolve_user_email

    def resolve(self, uid):
        state["email_open"] = state["open"]
        return real_resolve(self, uid)

    bid = seed_board(db, user)
    c = seed_card(db, user, bid, kind="chat")
    monkeypatch.setattr(bs, "_get_conn", tracked)
    monkeypatch.setattr(type(bs), "resolve_user_email", resolve)
    from app.api.v1.endpoints import chat
    from app.schemas.item import ChatMessage
    msg = ChatMessage(message="hi", message_id=new_id("m"), nodeId=c, boardId=bid)
    row, email = chat._resolve_v3_card(user, msg)
    assert row["id"] == c and email == user
    assert state["email_open"] == 0
    assert state["max"] == 1


def test_v3_sync_route_same_gate(api, db, user, env):
    bid = seed_board(db, user)
    r = api.post(SYNC, json=body("c_missing_y", boardId=bid))
    assert r.status_code == 404 and r.json()["error"] == "not_found"
    assert count(db, "c_missing_y") == 0
