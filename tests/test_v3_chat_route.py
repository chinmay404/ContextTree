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
            "model": "openai/gpt-5-mini", **extra}


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

    def rec(msg, graph, transport, user_id, carry="summary", turn_key=None):
        forks.append({"parent": msg.parent_thread_id, "fork_at": msg.fork_at_message_id,
                      "carry": carry, "transport": transport})

    from app.agent.helpers.v3_llm import TurnKey, v3_route

    monkeypatch.setattr(chat, "graph", SimpleNamespace(
        mongo_store=PostgresConversationStore(db_url=TEST_DB),
        get_stream_response=fake, get_response=fake_sync))
    monkeypatch.setattr(chat, "graph_init", True)
    monkeypatch.setattr(chat, "validate_model_access", lambda *a, **k: None)
    monkeypatch.setattr(chat, "_init_fork_if_needed", rec)
    # These tests are about the card gate; the key check has its own tests below.
    monkeypatch.setattr(chat, "_v3_turn_key",
                        lambda msg, email: TurnKey(v3_route(msg.model_name), "gate-test-key"))
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


# ── BYOK turns (app-shell plan Task 7): a V3 turn runs only on the user's own key ──
#
# These run the real graph, the real key store (encrypted rows in user_api_keys) and the real
# LangChain clients. Only the wire is fake: httpx transports answer as the provider would, and
# record every request so the tests can read which key and which model each call used.

import json
import secrets as _secrets

from tests.v3_support import seed_message, unit_emb

GPT = "openai/gpt-5-mini"
CLAUDE = "anthropic/claude-sonnet-4-20250514"
GEMINI = "gemini/gemini-3-flash-preview"
ENV_KEY = "env-key"


def _seal(plain: str, master: str) -> str:
    """Same format as lib/byok-crypto.ts: b64url(iv).b64url(tag).b64url(ciphertext)."""
    import base64
    import hashlib

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    def b64(b):
        return base64.urlsafe_b64encode(b).decode().rstrip("=")

    iv = _secrets.token_bytes(12)
    sealed = AESGCM(hashlib.sha256(master.encode()).digest()).encrypt(iv, plain.encode(), None)
    return ".".join([b64(iv), b64(sealed[-16:]), b64(sealed[:-16])])


def _completion(text: str, stream: bool) -> tuple[bytes, str]:
    base = {"id": "c1", "created": 1, "model": "gpt-5-mini"}
    if not stream:
        body = {**base, "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        return json.dumps(body).encode(), "application/json"
    chunks = [
        {**base, "object": "chat.completion.chunk",
         "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
        {**base, "object": "chat.completion.chunk",
         "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]},
        {**base, "object": "chat.completion.chunk",
         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return sse.encode(), "text/event-stream"


@pytest.fixture
def wire(monkeypatch):
    """Fake provider endpoints. `mode` is "ok" (answers `text`) or "reject" (401 for every call)."""
    import httpx

    state = SimpleNamespace(mode="ok", text="ok", calls=[], others=[])

    def reply(request):
        if request.url.host != "api.openai.com":
            state.others.append(str(request.url.host))
            return httpx.Response(500, json={"error": "unexpected host in test"}, request=request)
        body = json.loads(request.content.decode() or "{}")
        state.calls.append({"auth": request.headers.get("authorization"), "path": request.url.path,
                            "model": body.get("model"), "messages": body.get("messages") or []})
        if state.mode == "reject":
            return httpx.Response(401, json={"error": {"message": "Incorrect API key provided",
                                                       "type": "invalid_request_error",
                                                       "code": "invalid_api_key"}}, request=request)
        content, ctype = _completion(state.text, bool(body.get("stream")))
        return httpx.Response(200, content=content, headers={"content-type": ctype}, request=request)

    def handle(self, request):
        return reply(request)

    async def ahandle(self, request):
        return reply(request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", ahandle)
    return state


@pytest.fixture
def byok(user, db, monkeypatch, wire):
    """Real key store and real clients, fake wire. Env keys are set to prove they are never used."""
    from app.agent.helpers import v3_llm
    from app.api.v1.endpoints import chat
    from app.core.config import settings
    import app.agent.main as main

    master = _secrets.token_hex(16)
    monkeypatch.setenv("BYOK_ENCRYPTION_SECRET", master)
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY"):
        monkeypatch.setenv(name, ENV_KEY)
    monkeypatch.setattr(settings, "ENGINE_FAKE_LLM", False)
    monkeypatch.setattr(settings, "SUMMARY_MODEL_NAME", "openai/gpt-oss-120b")  # must be ignored on V3
    monkeypatch.setattr(main, "get_embedding", lambda *a, **k: unit_emb(3))
    monkeypatch.setattr(chat, "get_embedding", lambda *a, **k: unit_emb(3))

    lookups = []
    real_fetch = v3_llm.fetch_user_provider_api_key

    def counted(email, provider):
        lookups.append((email, provider))
        return real_fetch(email, provider)

    monkeypatch.setattr(v3_llm, "fetch_user_provider_api_key", counted)

    def put_key(provider, plain):
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO user_api_keys (user_email, provider, encrypted_key, created_at, updated_at) "
                "VALUES (%s, %s, %s, now(), now())", (user, provider, _seal(plain, master)))

    st = SimpleNamespace(lookups=lookups, put_key=put_key, wire=wire, chat=chat, main=main,
                         user_key=f"user-openai-{_secrets.token_hex(6)}")
    try:
        yield st
    finally:
        with db.cursor() as cur:
            cur.execute("DELETE FROM user_api_keys WHERE user_email = %s", (user,))


@pytest.fixture(params=["async", "sync"])
def graph_path(request, monkeypatch):
    """Run the stream on the async graph (astream_events), or on the sync fallback.

    The async Postgres checkpointer cannot start outside a running loop, so under pytest the
    app has no async graph; "async" builds one on an in-memory checkpointer.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    import app.agent.utils.saver as saver
    from app.api.v1.endpoints import chat

    g = chat._require_graph()
    if request.param == "async":
        async def pool_open():
            return True

        monkeypatch.setattr(g, "async_graph", g._make_builder().compile(checkpointer=InMemorySaver()))
        monkeypatch.setattr(saver, "ensure_async_pool_open", pool_open)
    else:
        monkeypatch.setattr(g, "async_graph", None)
    return request.param


def _frames(text: str) -> list:
    out = []
    for block in text.split("\n\n"):
        block = block.strip()
        if block.startswith("data: "):
            raw = block[len("data: "):]
            out.append(raw if raw == "[DONE]" else json.loads(raw))
    return out


def _text(frames) -> str:
    return "".join(f["message"] for f in frames if isinstance(f, dict) and "message" in f)


def _rows(db, node):
    with db.cursor() as cur:
        cur.execute("SELECT role, content FROM messages WHERE node_id = %s ORDER BY position", (node,))
        return cur.fetchall()


def _ask(api, node, board, model=GPT, route=STREAM, message="What is the plan?", **extra):
    return api.post(route, json={"message": message, "message_id": new_id("m"), "nodeId": node,
                                 "boardId": board, "model": model, **extra})


def _boom(*a, **k):
    raise AssertionError("must not be called on this turn")


@pytest.mark.parametrize("route", [STREAM, SYNC])
def test_v3_turn_without_user_key_is_missing_key(api, db, user, byok, route, monkeypatch):
    from app.agent.helpers import v3_llm
    monkeypatch.setattr(v3_llm, "_build_client", _boom)  # no client is ever built
    bid = seed_board(db, user)
    c = seed_card(db, user, bid, kind="chat")
    r = _ask(api, c, bid, model=CLAUDE, route=route)
    assert r.status_code == 400, r.text
    assert r.json() == {"error": "missing_key", "detail": "anthropic"}
    assert _rows(db, c) == [] and byok.wire.calls == [] and byok.wire.others == []


@pytest.mark.parametrize("route", [STREAM, SYNC])
@pytest.mark.parametrize("model", ["z-ai/glm-5.2", "litellm/x", "groq/compound"])
def test_v3_turn_unsupported_model(api, db, user, byok, route, model):
    bid = seed_board(db, user)
    c = seed_card(db, user, bid, kind="chat")
    r = _ask(api, c, bid, model=model, route=route)
    assert r.status_code == 400, r.text
    assert r.json() == {"error": "unsupported_model", "detail": model}
    assert byok.lookups == [] and _rows(db, c) == []


@pytest.mark.parametrize("route", [STREAM, SYNC])
def test_v3_key_store_down_is_503(api, db, user, byok, route, monkeypatch):
    from app.agent.helpers import v3_llm

    def down(email, provider):
        from app.agent.helpers.v3_keys import KeyStoreUnavailableError
        raise KeyStoreUnavailableError("key store unavailable (OperationalError)")

    monkeypatch.setattr(v3_llm, "fetch_user_provider_api_key", down)
    bid = seed_board(db, user)
    c = seed_card(db, user, bid, kind="chat")
    r = _ask(api, c, bid, route=route)
    assert r.status_code == 503, r.text
    assert r.json() == {"error": "key_store_unavailable", "detail": "openai"}
    assert _rows(db, c) == [] and byok.wire.calls == []


@pytest.mark.parametrize("route", [STREAM, SYNC])
def test_v3_server_config_error_is_a_generic_500(api, db, user, byok, route, monkeypatch):
    from app.agent.helpers import v3_llm

    def vertex(*a, **k):
        raise v3_llm.V3ConfigError("Gemini is switched to Vertex AI by the server environment")

    monkeypatch.setattr(v3_llm, "_build_client", vertex)
    byok.put_key("google", "user-google-key")
    bid = seed_board(db, user)
    c = seed_card(db, user, bid, kind="chat")
    r = _ask(api, c, bid, model=GEMINI, route=route)
    assert r.status_code == 500, r.text
    body = r.json()
    assert set(body) == {"error", "detail"} and body["error"] != "missing_key"
    assert "Vertex" not in r.text and "key" not in (body["detail"] or "").lower()
    assert _rows(db, c) == []


def test_v3_turn_uses_user_key(api, db, user, byok, graph_path):
    byok.put_key("openai", byok.user_key)
    bid = seed_board(db, user)
    c = seed_card(db, user, bid, kind="chat")
    r = _ask(api, c, bid)
    assert r.status_code == 200, r.text
    frames = _frames(r.text)
    assert _text(frames) == "ok" and frames[-1] == "[DONE]"
    assert byok.wire.calls and byok.wire.others == []
    assert {call["auth"] for call in byok.wire.calls} == {f"Bearer {byok.user_key}"}
    assert {call["model"] for call in byok.wire.calls} == {"gpt-5-mini"}
    assert byok.lookups == [(user, "openai")]  # looked up once for the whole turn
    assert [row[0] for row in _rows(db, c)] == ["user", "assistant"]


def test_v3_sync_route_uses_user_key(api, db, user, byok):
    byok.put_key("openai", byok.user_key)
    bid = seed_board(db, user)
    c = seed_card(db, user, bid, kind="chat")
    r = _ask(api, c, bid, route=SYNC)
    assert r.status_code == 200, r.text
    assert r.json()["message"] == "ok"
    assert {call["auth"] for call in byok.wire.calls} == {f"Bearer {byok.user_key}"}
    assert byok.lookups == [(user, "openai")]


def test_v3_summary_uses_chat_model_on_user_key(api, db, user, byok, monkeypatch):
    import app.agent.nodes.assistant_node as an
    from app.core.config import settings

    monkeypatch.setattr(settings, "MAX_MESSAGES_BEFORE_SUMMARY", 0)  # summarise after this answer
    monkeypatch.setattr(an, "get_llm", _boom)
    spy = []
    real = an.get_v3_llm

    def recording(model_id, user_email, *a, **k):
        route, key = k.get("resolved") or (None, None)
        spy.append({"model": model_id, "provider": getattr(route, "provider", None), "key": key})
        return real(model_id, user_email, *a, **k)

    monkeypatch.setattr(an, "get_v3_llm", recording)
    byok.put_key("openai", byok.user_key)
    bid = seed_board(db, user)
    c = seed_card(db, user, bid, kind="chat")
    byok.wire.text = "THE SUMMARY"
    r = _ask(api, c, bid)
    assert r.status_code == 200, r.text
    assert len(spy) == 2  # the answer, then the rolling summary
    assert all(s == {"model": GPT, "provider": "openai", "key": byok.user_key} for s in spy)
    assert len(byok.wire.calls) == 2
    assert {call["auth"] for call in byok.wire.calls} == {f"Bearer {byok.user_key}"}
    assert {call["model"] for call in byok.wire.calls} == {"gpt-5-mini"}  # not SUMMARY_MODEL_NAME
    assert byok.lookups == [(user, "openai")]
    with db.cursor() as cur:
        cur.execute("SELECT summary FROM nodes WHERE id = %s", (c,))
        assert cur.fetchone()[0] == "THE SUMMARY"


def _branch_with_history(db, user, bid) -> str:
    """A summary-carry branch forked at the last of six parent messages (four get summarised)."""
    parent = seed_card(db, user, bid, kind="chat")
    ids = [seed_message(db, user, parent, role, f"parent turn {i}", i)
           for i, role in enumerate(["user", "assistant"] * 3)]
    child = seed_card(db, user, bid, kind="chat", parent_id=parent, data={"carry": "summary"})
    with db.cursor() as cur:
        cur.execute("UPDATE nodes SET forked_from_message_id = %s WHERE id = %s", (ids[-1], child))
    return child


def test_v3_branch_start_summary_uses_chat_model_on_user_key(api, db, user, byok, monkeypatch):
    monkeypatch.setattr(byok.chat, "get_llm", _boom)
    byok.put_key("openai", byok.user_key)
    bid = seed_board(db, user)
    child = _branch_with_history(db, user, bid)
    byok.wire.text = "BRANCH SUMMARY"
    r = _ask(api, child, bid)
    assert r.status_code == 200, r.text
    first = byok.wire.calls[0]
    assert "parent turn 0" in json.dumps(first["messages"])  # the branch-start summary call
    assert {call["auth"] for call in byok.wire.calls} == {f"Bearer {byok.user_key}"}
    assert {call["model"] for call in byok.wire.calls} == {"gpt-5-mini"}
    assert byok.lookups == [(user, "openai")]
    with db.cursor() as cur:
        cur.execute("SELECT summary, is_initialized FROM nodes WHERE id = %s", (child,))
        assert cur.fetchone() == ("BRANCH SUMMARY", True)


def test_v3_provider_rejects_key_frame(api, db, user, byok, graph_path):
    byok.put_key("openai", byok.user_key)
    bid = seed_board(db, user)
    c = seed_card(db, user, bid, kind="chat")
    byok.wire.mode = "reject"
    r = _ask(api, c, bid, message="Will it work?")
    assert r.status_code == 200, r.text
    frames = _frames(r.text)
    assert frames == [{"error": "OpenAI turned this key down.", "code": "key_rejected",
                       "provider": "openai"}]
    assert "[DONE]" not in r.text and byok.user_key not in r.text and "Incorrect" not in r.text
    assert len(byok.wire.calls) == 1  # refused once, not re-run on the sync fallback
    # No assistant row, and no user row either: the board drops the failed bubble and re-asks
    # the same text with a fresh messageId once a key is saved, so a stored row would double it.
    assert _rows(db, c) == []

    # The re-ask after the key is fixed: one question, one answer, and the model sees the
    # question once (the refused turn left nothing behind).
    byok.wire.mode, byok.wire.calls[:] = "ok", []
    r = _ask(api, c, bid, message="Will it work?")
    assert r.status_code == 200, r.text
    assert _text(_frames(r.text)) == "ok"
    assert _rows(db, c) == [("user", "Will it work?"), ("assistant", "ok")]
    asked = [m for m in byok.wire.calls[0]["messages"] if m.get("role") == "user"]
    assert len(asked) == 1


def test_v3_sync_route_rejected_key_is_400(api, db, user, byok):
    byok.put_key("openai", byok.user_key)
    bid = seed_board(db, user)
    c = seed_card(db, user, bid, kind="chat")
    byok.wire.mode = "reject"
    r = _ask(api, c, bid, route=SYNC)
    assert r.status_code == 400, r.text
    assert r.json() == {"error": "key_rejected", "detail": "openai"}
    assert byok.user_key not in r.text and _rows(db, c) == []


def test_v3_branch_start_summary_rejected_key_frame(api, db, user, byok):
    byok.put_key("openai", byok.user_key)
    bid = seed_board(db, user)
    child = _branch_with_history(db, user, bid)
    byok.wire.mode = "reject"
    r = _ask(api, child, bid)
    assert r.status_code == 200, r.text
    assert _frames(r.text) == [{"error": "OpenAI turned this key down.", "code": "key_rejected",
                                "provider": "openai"}]
    assert len(byok.wire.calls) == 1  # the answer is not attempted on a refused key
    with db.cursor() as cur:
        cur.execute("SELECT is_initialized FROM nodes WHERE id = %s", (child,))
        assert cur.fetchone()[0] is False  # inheritance is retried on the next message
    assert _rows(db, child) == []


class _Status401(Exception):
    status_code = 401


def test_v3_web_search_401_is_not_key_rejected(api, db, user, byok, monkeypatch):
    def search(*a, **k):
        raise _Status401("web search provider: 401 unauthorized")

    monkeypatch.setattr(byok.main, "search_web", search)
    byok.put_key("openai", byok.user_key)
    bid = seed_board(db, user)
    c = seed_card(db, user, bid, kind="chat")
    r = _ask(api, c, bid, webSearch=True)
    assert r.status_code == 200, r.text
    assert not any(isinstance(f, dict) and f.get("code") == "key_rejected" for f in _frames(r.text))


def test_v3_non_llm_401_in_the_node_is_not_key_rejected(api, db, user, byok, monkeypatch):
    import app.agent.nodes.assistant_node as an

    def broken(*a, **k):
        raise _Status401("some other service said 401")

    monkeypatch.setattr(an, "build_memory_context_block", broken)
    byok.put_key("openai", byok.user_key)
    bid = seed_board(db, user)
    c = seed_card(db, user, bid, kind="chat")
    r = _ask(api, c, bid)
    assert r.status_code == 200, r.text
    assert not any(isinstance(f, dict) and f.get("code") == "key_rejected" for f in _frames(r.text))
    assert byok.wire.calls == []


def test_v3_fake_mode_answers_without_keys(api, db, user, byok, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "ENGINE_FAKE_LLM", True)
    bid = seed_board(db, user)
    c = seed_card(db, user, bid, kind="chat")
    r = _ask(api, c, bid, model=CLAUDE, message="Hello there")
    assert r.status_code == 200, r.text
    frames = _frames(r.text)
    assert _text(frames) == "Fake answer to: Hello there" and frames[-1] == "[DONE]"
    assert byok.lookups == [] and byok.wire.calls == []


@pytest.mark.parametrize("route", [STREAM, SYNC])
def test_v2_turn_unchanged(api, db, user, byok, route, monkeypatch):
    """No boardId on a V2 node: validate_model_access and get_llm, never the V3 resolver."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    import app.agent.nodes.assistant_node as an

    seen = {"access": [], "get_llm": []}

    def access(model, user_id):
        seen["access"].append(model)
        return None

    def v2_llm(model, *a, **k):
        seen["get_llm"].append(model)
        return FakeListChatModel(responses=["v2 answer"])

    monkeypatch.setattr(byok.chat, "validate_model_access", access)
    monkeypatch.setattr(an, "get_llm", v2_llm)
    monkeypatch.setattr(byok.chat, "v3_require_key", _boom, raising=False)
    monkeypatch.setattr(an, "get_v3_llm", _boom, raising=False)
    v2 = seed_board(db, user, version=2)
    nid = new_id("n")
    with db.cursor() as cur:
        cur.execute("INSERT INTO nodes (id, canvas_id, user_email, data) VALUES (%s,%s,%s,'{}'::jsonb)",
                    (nid, v2, user))
    r = api.post(route, json={"message": "hi", "message_id": new_id("m"), "nodeId": nid,
                              "model": "z-ai/glm-5.2"})
    assert r.status_code == 200, r.text
    assert "v2 answer" in r.text
    assert seen["access"] == ["z-ai/glm-5.2"] and seen["get_llm"][0] == "z-ai/glm-5.2"
    assert byok.lookups == [] and byok.wire.calls == []


# ── Task 7 review follow-ups: the daily quota on key errors, summaries without refused questions ──

def _quota(db, user) -> int:
    with db.cursor() as cur:
        cur.execute("SELECT msg_count FROM quotas WHERE user_key = %s AND day = CURRENT_DATE", (user,))
        row = cur.fetchone()
        return row[0] if row else 0


@pytest.mark.parametrize("route", [STREAM, SYNC])
def test_v3_missing_key_does_not_use_a_daily_message(api, db, user, byok, route):
    bid = seed_board(db, user)
    c = seed_card(db, user, bid, kind="chat")
    r = _ask(api, c, bid, model=CLAUDE, route=route)
    assert r.status_code == 400 and r.json()["error"] == "missing_key"
    assert _quota(db, user) == 0  # the key is checked before the quota is charged


@pytest.mark.parametrize("route", [STREAM, SYNC])
def test_v3_refused_key_refunds_the_daily_message(api, db, user, byok, route):
    byok.put_key("openai", byok.user_key)
    bid = seed_board(db, user)
    c = seed_card(db, user, bid, kind="chat")
    assert _ask(api, c, bid, route=route).status_code == 200  # one answered turn
    assert _quota(db, user) == 1
    byok.wire.mode = "reject"
    r = _ask(api, c, bid, route=route)
    if route == STREAM:
        assert _frames(r.text)[-1]["code"] == "key_rejected"
    else:
        assert r.status_code == 400 and r.json()["error"] == "key_rejected"
    assert _quota(db, user) == 1


def test_v3_refused_first_turn_refund_never_goes_below_zero(api, db, user, byok):
    byok.put_key("openai", byok.user_key)
    bid = seed_board(db, user)
    c = seed_card(db, user, bid, kind="chat")
    byok.wire.mode = "reject"
    _ask(api, c, bid)
    _ask(api, c, bid, route=SYNC)
    assert _quota(db, user) == 0


def test_v3_refused_branch_start_summary_refunds_the_daily_message(api, db, user, byok):
    byok.put_key("openai", byok.user_key)
    bid = seed_board(db, user)
    child = _branch_with_history(db, user, bid)
    byok.wire.mode = "reject"
    r = _ask(api, child, bid)
    assert _frames(r.text) == [{"error": "OpenAI turned this key down.", "code": "key_rejected",
                                "provider": "openai"}]
    assert _quota(db, user) == 0


@pytest.mark.parametrize("byok_only, kept", [(True, False), (False, True)])
def test_rolling_summary_leaves_out_a_refused_question(monkeypatch, byok_only, kept):
    from langchain_core.messages import AIMessage, HumanMessage

    import app.agent.nodes.assistant_node as an
    from tests.v3_support import FakeLLM

    llm = FakeLLM("S")
    monkeypatch.setattr(an, "get_v3_llm", lambda *a, **k: llm)
    monkeypatch.setattr(an, "get_llm", lambda *a, **k: llm)
    state = {
        "messages": [HumanMessage(content="REFUSED QUESTION", id="m1"),
                     HumanMessage(content="answered question", id="m2"),
                     AIMessage(content="the answer", id="a2")],
        "summary": "", "model": GPT, "user_id": "u@example.com", "byok_only": byok_only,
    }
    out = an.AgentNodes(mongo_store=None).summurize(state)
    assert out["summary"] == "S"
    prompt = str(llm.calls[0])
    assert "answered question" in prompt and "the answer" in prompt
    assert ("REFUSED QUESTION" in prompt) is kept  # V2 (byok_only off) is unchanged
