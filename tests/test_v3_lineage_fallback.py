from app.agent.store.PostgresStore import PostgresConversationStore
from tests.v3_support import TEST_DB, requires_db, db, user, seed_board, seed_card, seed_edge  # noqa: F401

pytestmark = requires_db


def _meta(db, child):
    return PostgresConversationStore(db_url=TEST_DB).get_thread_fork_metadata(child)


def _setup(db, user):
    b = seed_board(db, user)
    other = seed_card(db, user, b)
    parent = seed_card(db, user, b)
    child = seed_card(db, user, b)
    return b, other, parent, child


def test_fallback_ignores_wire_edge(db, user):
    b, other, parent, child = _setup(db, user)
    seed_edge(db, user, b, other, child, "wire")
    seed_edge(db, user, b, parent, child, "lineage")
    assert _meta(db, child)["parent_node_id"] == parent
    c2 = seed_card(db, user, b)
    seed_edge(db, user, b, other, c2, "wire")
    assert _meta(db, c2)["parent_node_id"] is None


def test_fallback_ignores_answer_edge(db, user):
    b, other, parent, child = _setup(db, user)
    seed_edge(db, user, b, other, child, "answer")
    assert _meta(db, child)["parent_node_id"] is None


def test_fallback_uses_lineage_edge(db, user):
    b, other, parent, child = _setup(db, user)
    seed_edge(db, user, b, parent, child, "lineage")
    assert _meta(db, child)["parent_node_id"] == parent


def test_fallback_uses_legacy_null_kind_fork_edge(db, user):
    b, other, parent, child = _setup(db, user)
    seed_edge(db, user, b, parent, child, None, meta={"condition": "Fork"})
    assert _meta(db, child)["parent_node_id"] == parent
    c2 = seed_card(db, user, b)
    seed_edge(db, user, b, parent, c2, None, data={"meta": {"condition": "Fork"}})
    assert _meta(db, c2)["parent_node_id"] == parent


def test_fallback_ignores_null_kind_context_edge(db, user):
    b, other, parent, child = _setup(db, user)
    seed_edge(db, user, b, parent, child, None, meta={"condition": "Context"})
    assert _meta(db, child)["parent_node_id"] is None


def test_fallback_ignores_deleted_lineage_edge(db, user):
    b, other, parent, child = _setup(db, user)
    with db.cursor() as cur:
        cur.execute("SELECT now()")
        now = cur.fetchone()[0]
    seed_edge(db, user, b, parent, child, "lineage", deleted_at=now)
    assert _meta(db, child)["parent_node_id"] is None
