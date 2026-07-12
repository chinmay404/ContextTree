"""
Tenancy isolation — the launch blocker (V2 architecture §7 row 1):
user A must never read or mutate user B's data.

Runs against a real Postgres (TEST_DATABASE_URL, e.g. the local Docker
pgvector instance). Skips if that is not set, so it never touches prod.

    TEST_DATABASE_URL=postgresql://postgres:ctxdev@localhost:55432/tenancy \
        .venv/Scripts/python.exe -m pytest tests/test_tenancy.py -q
"""

import os
import uuid

import psycopg2
import pytest

TEST_DB = os.getenv("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(not TEST_DB, reason="TEST_DATABASE_URL not set")


def _seed(conn, email, node_id):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (id, email) VALUES (gen_random_uuid(), %s) "
            "ON CONFLICT (email) DO NOTHING",
            (email,),
        )
        cur.execute(
            "INSERT INTO canvases (id, user_email, data) VALUES (%s, %s, '{}'::jsonb)",
            (f"canvas_{node_id}", email),
        )
        cur.execute(
            "INSERT INTO nodes (id, canvas_id, user_email, summary, is_primary) "
            "VALUES (%s, %s, %s, %s, true)",
            (node_id, f"canvas_{node_id}", email, f"secret summary of {email}"),
        )
        cur.execute(
            "INSERT INTO messages (id, node_id, user_email, role, content, position) "
            "VALUES (%s, %s, %s, 'user', %s, 1)",
            (f"msg_{node_id}", node_id, email, f"secret message of {email}"),
        )
    conn.commit()


@pytest.fixture(scope="module")
def store_and_data():
    # Point the app store at the test DB.
    os.environ["DATABASE_URL"] = TEST_DB
    from app.agent.store.PostgresStore import PostgresConversationStore

    conn = psycopg2.connect(TEST_DB)
    alice = "alice@example.com"
    bob = "bob@example.com"
    a_node = f"node_alice_{uuid.uuid4().hex[:8]}"
    b_node = f"node_bob_{uuid.uuid4().hex[:8]}"
    _seed(conn, alice, a_node)
    _seed(conn, bob, b_node)

    store = PostgresConversationStore()
    yield store, alice, bob, a_node, b_node

    with conn.cursor() as cur:
        for n in (a_node, b_node):
            cur.execute("DELETE FROM messages WHERE node_id = %s", (n,))
            cur.execute("DELETE FROM nodes WHERE id = %s", (n,))
            cur.execute("DELETE FROM canvases WHERE id = %s", (f"canvas_{n}",))
        cur.execute("DELETE FROM users WHERE email IN (%s, %s)", (alice, bob))
    conn.commit()
    conn.close()


def test_owner_reads_own_summary(store_and_data):
    store, alice, _bob, a_node, _b_node = store_and_data
    assert "alice@example.com" in store.get_thread_summary(alice, a_node)


def test_cross_tenant_summary_is_empty(store_and_data):
    store, alice, _bob, _a_node, b_node = store_and_data
    # Alice asks for Bob's node → must get nothing, not Bob's secret.
    assert store.get_thread_summary(alice, b_node) == ""


def test_cross_tenant_messages_empty(store_and_data):
    store, alice, _bob, _a_node, b_node = store_and_data
    assert store.get_thread_messages(alice, b_node) == []


def test_cross_tenant_recent_messages_empty(store_and_data):
    store, alice, _bob, _a_node, b_node = store_and_data
    assert store.get_thread_recent_messages(alice, b_node, 10) == []


def test_cross_tenant_summary_state_empty(store_and_data):
    store, alice, _bob, _a_node, b_node = store_and_data
    st = store.get_thread_summary_state(alice, b_node)
    assert st["summary"] == ""


def test_cross_tenant_update_summary_denied(store_and_data):
    store, alice, bob, _a_node, b_node = store_and_data
    # Alice tries to overwrite Bob's summary → no row updated.
    assert store.update_thread_summary(alice, b_node, "hacked by alice") is False
    # Bob's summary is intact.
    assert "bob@example.com" in store.get_thread_summary(bob, b_node)


def test_owner_gate_resolves_owner(store_and_data):
    store, _alice, bob, _a_node, b_node = store_and_data
    assert store.get_thread_owner(b_node) == bob
    assert store.get_thread_owner("does_not_exist") is None


def test_daily_quota_increments_and_isolates(store_and_data):
    store, alice, bob, _a, _b = store_and_data
    a1 = store.increment_daily_quota(alice)
    a2 = store.increment_daily_quota(alice)
    b1 = store.increment_daily_quota(bob)
    assert a2 == a1 + 1
    assert b1 < a2  # bob's bucket is independent of alice's
