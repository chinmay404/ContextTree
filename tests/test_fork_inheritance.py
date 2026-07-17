"""
Fork inheritance & context isolation — the product's core guarantee
(V2/10-CONTEXT-PIPELINE.md): a branch starts with its parent lineage's
knowledge, and context flows DOWN a lineage only — never sideways to a
sibling, never past the fork point, never across tenants.

Runs against a real Postgres (TEST_DATABASE_URL, e.g. the local Docker
pgvector instance with migrations 001-004 applied). Skips if not set,
so it never touches prod.

    TEST_DATABASE_URL=postgresql://postgres:ctxdev@localhost:55432/forktest \
        .venv/Scripts/python.exe -m pytest tests/test_fork_inheritance.py -q

Layers covered:
  * store  — fork_thread / get_thread_ancestry_scopes /
             find_similar_by_message_id / get_fork_inheritance_payload /
             get_thread_messages_after (watermark hydration)
  * chat   — _init_fork_if_needed orchestration: first-message seeding,
             node-row metadata fallback, and the empty-parent retry
             (the 2026-07-13 "frozen blank fork" regression).

No LLM or embedding API is called: parents are kept at <= FORK_BUFFER(2)
messages on chat-layer tests (so nothing needs summarising) and embeddings
are deterministic unit basis vectors.
"""

import os
import uuid
from types import SimpleNamespace

import psycopg2
import pytest

TEST_DB = os.getenv("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(not TEST_DB, reason="TEST_DATABASE_URL not set")

if TEST_DB:
    # Defensive: make sure nothing imported below can dial a real DB.
    os.environ["DATABASE_URL"] = TEST_DB

EMB_DIM = 768


def unit_emb(i: int):
    """Deterministic embedding: basis vector e_i. Identical i => cosine 1.0,
    different i => cosine 0.0 — lets tests bait the retriever precisely."""
    v = [0.0] * EMB_DIM
    v[i % EMB_DIM] = 1.0
    return v


def _mk_ids(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def env():
    """One user + one canvas + a seeded lineage:

        root (m1..m4) ──fork@m2──> child_a (2 buffer msgs)
             │                        └─fork@buffer──> grandchild
             └────fork@m4──> child_b (sibling of child_a)

    plus a second tenant (bob) for cross-tenant checks.
    """
    from app.agent.store.PostgresStore import PostgresConversationStore

    store = PostgresConversationStore(db_url=TEST_DB)
    conn = psycopg2.connect(TEST_DB)

    alice = f"alice_{uuid.uuid4().hex[:6]}@example.com"
    bob = f"bob_{uuid.uuid4().hex[:6]}@example.com"
    canvas = _mk_ids("canvas")

    with conn.cursor() as cur:
        for email in (alice, bob):
            cur.execute(
                "INSERT INTO users (id, email) VALUES (gen_random_uuid(), %s) "
                "ON CONFLICT (email) DO NOTHING",
                (email,),
            )
        cur.execute(
            "INSERT INTO canvases (id, user_email, data) VALUES (%s, %s, '{}'::jsonb)",
            (canvas, alice),
        )
    conn.commit()

    root = _mk_ids("root")
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO nodes (id, canvas_id, user_email, summary, is_primary) "
            "VALUES (%s, %s, %s, %s, true)",
            (root, canvas, alice, "root summary: trip planning for Japan"),
        )
    conn.commit()

    # m1..m4 on the root, each with a distinct basis embedding.
    msg_ids = []
    for i, text in enumerate(
        ["plan a 10-day japan trip", "kyoto food itinerary",
         "root secret three", "root secret four"]
    ):
        mid = _mk_ids(f"m{i + 1}")
        store.add_message(
            user_id=alice, thread_id=root, message_id=mid,
            role="user" if i % 2 == 0 else "assistant",
            text=text, embedding=unit_emb(i),
        )
        msg_ids.append(mid)
    m1, m2, m3, m4 = msg_ids

    def fork(source, new_id, at_msg, summary):
        payload = store.get_fork_inheritance_payload(alice, source, at_msg, 6)
        buffer = payload["messages"][-2:]
        assert store.fork_thread(
            user_id=alice, source_thread_id=source, new_thread_id=new_id,
            fork_at_message_id=at_msg, summary=summary,
            summary_embedding=None, initial_messages=buffer,
            memory_facts={"user_profile": {"name": "Chinmay"}},
        )

    child_a = _mk_ids("child_a")
    fork(root, child_a, m2, "inherited: japan trip, kyoto focus")

    child_b = _mk_ids("child_b")
    fork(root, child_b, m4, "inherited: full root context")

    # child_b gets its own message with a unique embedding — sibling bait.
    b_msg = _mk_ids("bmsg")
    store.add_message(
        user_id=alice, thread_id=child_b, message_id=b_msg,
        role="user", text="sibling-only secret: budget rework",
        embedding=unit_emb(10),
    )

    # grandchild forks off child_a at child_a's last buffer message.
    a_msgs = store.get_thread_messages(alice, child_a)
    grandchild = _mk_ids("grand")
    fork(child_a, grandchild, a_msgs[-1]["message_id"],
         "inherited: down the a-lineage")

    yield SimpleNamespace(
        store=store, conn=conn, alice=alice, bob=bob, canvas=canvas,
        root=root, m=msg_ids, child_a=child_a, child_b=child_b,
        b_msg=b_msg, grandchild=grandchild,
    )

    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM messages WHERE node_id IN "
            "(SELECT id FROM nodes WHERE user_email IN (%s, %s))",
            (alice, bob),
        )
        cur.execute("DELETE FROM nodes WHERE user_email IN (%s, %s)", (alice, bob))
        cur.execute("DELETE FROM canvases WHERE user_email IN (%s, %s)", (alice, bob))
        cur.execute("DELETE FROM quotas WHERE user_key IN (%s, %s)", (alice, bob))
        cur.execute("DELETE FROM users WHERE email IN (%s, %s)", (alice, bob))
    conn.commit()
    conn.close()


# ── Store layer: fork seeding ──────────────────────────────────────────────────


def test_fork_seeds_summary_buffer_and_ancestry(env):
    s = env.store
    with env.conn.cursor() as cur:
        cur.execute(
            "SELECT parent_node_id, forked_from_message_id, summary, "
            "ancestor_ids, is_initialized, is_primary FROM nodes WHERE id = %s",
            (env.child_a,),
        )
        parent_id, fork_at, summary, ancestors, initialized, primary = cur.fetchone()
    assert parent_id == env.root
    assert fork_at == env.m[1]  # m2
    assert "inherited" in summary
    assert list(ancestors) == [env.root]
    assert initialized is True
    assert primary is False

    buffer = s.get_thread_messages(env.alice, env.child_a)
    assert len(buffer) == 2, "FORK_BUFFER(2) messages must be copied verbatim"
    assert [b["position"] for b in buffer] == [1, 2], (
        "buffer messages must get monotonic non-NULL positions "
        "(NULL positions silently break all hydration — migration 004)"
    )
    assert buffer[-1]["text"] == "kyoto food itinerary"  # up to fork point m2

    facts = s.get_thread_memory_facts(env.alice, env.child_a)
    assert facts.get("user_profile", {}).get("name") == "Chinmay"


def test_fork_of_fork_walks_full_lineage(env):
    scopes = env.store.get_thread_ancestry_scopes(env.grandchild)
    ids = [t for t, _ in scopes]
    assert ids == [env.grandchild, env.child_a, env.root], (
        "grandchild scope must be self → parent → root"
    )
    caps = dict(scopes)
    assert caps[env.child_a] is not None, "parent must be capped at grandchild's fork point"
    assert caps[env.root] == env.m[1], "root must be capped at child_a's fork point (m2)"


def test_sibling_never_in_ancestry_scope(env):
    for node in (env.child_a, env.grandchild):
        ids = [t for t, _ in env.store.get_thread_ancestry_scopes(node)]
        assert env.child_b not in ids, f"sibling {env.child_b} leaked into {node} scope"


# ── Store layer: retrieval isolation ───────────────────────────────────────────


def test_retrieval_excludes_sibling_even_when_most_similar(env):
    """Bait: query embedding IDENTICAL to the sibling's message (cosine 1.0).
    Scope fencing must still exclude it."""
    s = env.store
    scopes = s.get_thread_ancestry_scopes(env.child_a)
    hits = s.find_similar_by_message_id(
        env.alice, scopes, unit_emb(10), top_k=10, min_score=0.0
    )
    hit_ids = {h["message_id"] for h in hits}
    assert env.b_msg not in hit_ids, "sibling message retrieved despite scope fence"
    lineage_ids = {m["message_id"] for m in s.get_thread_messages(env.alice, env.root)}
    lineage_ids |= {m["message_id"] for m in s.get_thread_messages(env.alice, env.child_a)}
    assert hit_ids <= lineage_ids, f"retrieval escaped the lineage: {hit_ids - lineage_ids}"


def test_retrieval_respects_fork_point_cap(env):
    """child_a forked at m2 — root messages AFTER m2 (m3, m4) must be invisible,
    even when the query embedding matches them exactly."""
    s = env.store
    scopes = s.get_thread_ancestry_scopes(env.child_a)
    for bait_idx, banned in ((2, env.m[2]), (3, env.m[3])):
        hits = s.find_similar_by_message_id(
            env.alice, scopes, unit_emb(bait_idx), top_k=10, min_score=0.0
        )
        assert banned not in {h["message_id"] for h in hits}, (
            f"post-fork-point parent message {banned} leaked into the branch"
        )


def test_retrieval_before_fork_point_is_visible(env):
    """Sanity inverse: pre-fork-point parent knowledge (m1) IS retrievable."""
    s = env.store
    scopes = s.get_thread_ancestry_scopes(env.child_a)
    hits = s.find_similar_by_message_id(
        env.alice, scopes, unit_emb(0), top_k=10, min_score=0.5
    )
    assert env.m[0] in {h["message_id"] for h in hits}, (
        "pre-fork-point ancestor message must be retrievable from the branch"
    )


# ── Store layer: inheritance payload ───────────────────────────────────────────


def test_inheritance_payload_scoped_to_fork_point(env):
    p = env.store.get_fork_inheritance_payload(env.alice, env.root, env.m[1], 6)
    assert p["mode"] == "scoped-fork-point"
    texts = [m["text"] for m in p["messages"]]
    assert texts == ["plan a 10-day japan trip", "kyoto food itinerary"], (
        "payload must stop at the fork point — no later parent messages"
    )


def test_inheritance_payload_cross_tenant_is_empty(env):
    """bob asking for alice's thread must get nothing (tenant fence)."""
    p = env.store.get_fork_inheritance_payload(env.bob, env.root, env.m[1], 6)
    assert p["messages"] == []
    assert not (p["summary"] or "").strip()


# ── Store layer: watermark hydration (the NULL-position regression class) ─────


def test_watermark_hydration_and_advance(env):
    s = env.store
    node = _mk_ids("hydra")
    for i in range(3):
        s.add_message(
            user_id=env.alice, thread_id=node, message_id=_mk_ids(f"h{i}"),
            role="user", text=f"turn {i}", embedding=unit_emb(20 + i),
        )
    tail = s.get_thread_messages_after(env.alice, node, 0, 10)
    assert len(tail) == 3, (
        "position > 0 must match fresh rows — NULL positions break hydration"
    )
    assert s.update_thread_summary(env.alice, node, "folded turns 0-1",
                                   summarized_up_to_position=2)
    st = s.get_thread_summary_state(env.alice, node)
    assert st["summarized_up_to_position"] == 2
    tail = s.get_thread_messages_after(env.alice, node, st["summarized_up_to_position"], 10)
    assert [m["text"] for m in tail] == ["turn 2"], (
        "messages at/below the watermark must never be re-fed"
    )


# ── Store layer: idempotency & repair ─────────────────────────────────────────


def test_fork_thread_is_idempotent(env):
    s = env.store
    before = s.get_thread_message_count(env.child_a)
    payload = s.get_fork_inheritance_payload(env.alice, env.root, env.m[1], 6)
    assert s.fork_thread(
        user_id=env.alice, source_thread_id=env.root, new_thread_id=env.child_a,
        fork_at_message_id=env.m[1], summary="should be ignored",
        summary_embedding=None, initial_messages=payload["messages"][-2:],
        memory_facts={},
    )
    assert s.get_thread_message_count(env.child_a) == before, (
        "re-running fork_thread must not duplicate buffer messages"
    )


def test_frozen_fork_marked_initialized_without_ancestry_is_repaired(env):
    """The legacy freeze: a row stamped is_initialized=true with empty
    ancestor_ids (created before the self-heal fix). fork_thread must repair
    it instead of skipping."""
    s = env.store
    frozen = _mk_ids("frozen")
    with env.conn.cursor() as cur:
        cur.execute(
            "INSERT INTO nodes (id, canvas_id, user_email, is_initialized, is_primary) "
            "VALUES (%s, %s, %s, true, false)",
            (frozen, env.canvas, env.alice),
        )
    env.conn.commit()

    payload = s.get_fork_inheritance_payload(env.alice, env.root, env.m[1], 6)
    assert s.fork_thread(
        user_id=env.alice, source_thread_id=env.root, new_thread_id=frozen,
        fork_at_message_id=env.m[1], summary="repaired inheritance",
        summary_embedding=None, initial_messages=payload["messages"][-2:],
        memory_facts={},
    )
    with env.conn.cursor() as cur:
        cur.execute("SELECT ancestor_ids, summary FROM nodes WHERE id = %s", (frozen,))
        ancestors, summary = cur.fetchone()
    assert list(ancestors) == [env.root], "repair must materialize the ancestry"
    assert summary == "repaired inheritance"


# ── Chat layer: _init_fork_if_needed orchestration ────────────────────────────


def _chat_message(node_id, parent_id=None, fork_at=None):
    from app.schemas.item import ChatMessage

    body = {"message": "hello branch", "message_id": str(uuid.uuid4()),
            "nodeId": node_id}
    if parent_id:
        body["parentNodeId"] = parent_id
    if fork_at:
        body["forkedFromMessageId"] = fork_at
    return ChatMessage(**body)


def _small_parent(env, n_messages):
    """Parent with <= FORK_BUFFER messages so fork-init needs no summarizer LLM."""
    parent = _mk_ids("cparent")
    with env.conn.cursor() as cur:
        cur.execute(
            "INSERT INTO nodes (id, canvas_id, user_email, is_primary) "
            "VALUES (%s, %s, %s, true)",
            (parent, env.canvas, env.alice),
        )
    env.conn.commit()
    for i in range(n_messages):
        env.store.add_message(
            user_id=env.alice, thread_id=parent, message_id=_mk_ids(f"cp{i}"),
            role="user" if i % 2 == 0 else "assistant",
            text=f"parent fact {i}", embedding=unit_emb(30 + i),
        )
    return parent


def test_first_message_initializes_fork_from_request_pointers(env):
    from app.api.v1.endpoints.chat import _init_fork_if_needed

    parent = _small_parent(env, 2)
    child = _mk_ids("cchild")
    graph = SimpleNamespace(mongo_store=env.store)

    _init_fork_if_needed(_chat_message(child, parent_id=parent), graph, "test", env.alice)

    assert env.store.thread_exists(child)
    assert [t for t, _ in env.store.get_thread_ancestry_scopes(child)] == [child, parent]
    texts = [m["text"] for m in env.store.get_thread_messages(env.alice, child)]
    assert texts == ["parent fact 0", "parent fact 1"], (
        "first message must seed the branch with the parent buffer"
    )


def test_first_message_falls_back_to_node_row_metadata(env):
    """The frontend persists parent_node_id via canvas sync but the chat body
    may arrive without parentNodeId (stale client snapshot). The backend must
    recover the lineage from the child's own node row."""
    from app.api.v1.endpoints.chat import _init_fork_if_needed

    parent = _small_parent(env, 2)
    child = _mk_ids("cchild")
    with env.conn.cursor() as cur:
        cur.execute(
            "INSERT INTO nodes (id, canvas_id, user_email, parent_node_id, "
            "is_initialized, is_primary) VALUES (%s, %s, %s, %s, false, false)",
            (child, env.canvas, env.alice, parent),
        )
    env.conn.commit()

    graph = SimpleNamespace(mongo_store=env.store)
    _init_fork_if_needed(_chat_message(child), graph, "test", env.alice)  # no parent in body

    with env.conn.cursor() as cur:
        cur.execute("SELECT ancestor_ids, is_initialized FROM nodes WHERE id = %s", (child,))
        ancestors, initialized = cur.fetchone()
    assert list(ancestors) == [parent], "lineage must self-heal from the node row"
    assert initialized is True
    assert env.store.get_thread_message_count(child) == 2


def test_empty_parent_leaves_fork_uninitialized_then_retries(env):
    """2026-07-13 regression: forking while the parent has NOTHING persisted
    must NOT stamp the branch initialized-empty forever. The next message
    (after the parent has content) must complete inheritance."""
    from app.api.v1.endpoints.chat import _init_fork_if_needed

    parent = _small_parent(env, 0)  # node row exists, zero messages
    child = _mk_ids("cchild")
    graph = SimpleNamespace(mongo_store=env.store)

    _init_fork_if_needed(_chat_message(child, parent_id=parent), graph, "test", env.alice)
    assert not env.store.thread_exists(child), (
        "nothing to inherit — the fork must stay uninitialized so it can retry"
    )

    for i in range(2):
        env.store.add_message(
            user_id=env.alice, thread_id=parent, message_id=_mk_ids(f"late{i}"),
            role="user", text=f"late parent fact {i}", embedding=unit_emb(40 + i),
        )
    _init_fork_if_needed(_chat_message(child, parent_id=parent), graph, "test", env.alice)

    assert env.store.thread_exists(child), "retry must succeed once the parent has content"
    texts = [m["text"] for m in env.store.get_thread_messages(env.alice, child)]
    assert texts == ["late parent fact 0", "late parent fact 1"]
