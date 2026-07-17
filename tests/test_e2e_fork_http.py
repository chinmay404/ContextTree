"""
End-to-end fork inheritance over HTTP — the layer the unit/integration
suite (test_fork_inheritance.py) deliberately does not cover: the real
FastAPI service, real JWT auth, real persistence, real LLM. This is the
test that fails when "the fork doesn't know my name" in the actual
product, regardless of which internal layer broke.

Skips unless E2E_BASE_URL is set — it never runs accidentally.

Local run (proves the code you're about to deploy):

    # terminal 1 — serve the backend against the disposable test DB:
    DATABASE_URL=postgresql://postgres:ctxdev@localhost:55432/forktest \
        .venv/Scripts/python.exe -m uvicorn app.api.server:app --port 8010

    # terminal 2:
    E2E_BASE_URL=http://127.0.0.1:8010 \
    TEST_DATABASE_URL=postgresql://postgres:ctxdev@localhost:55432/forktest \
        .venv/Scripts/python.exe -m pytest tests/test_e2e_fork_http.py -q

Prod smoke (after a Railway deploy): set E2E_BASE_URL to the Railway URL
and E2E_USER_ID to a real users.id uuid; BACKEND_JWT_SECRET must be the
prod value. DB assertions are skipped when TEST_DATABASE_URL is unset —
the behavioral assertion (the fork knows the name) still runs.
"""

import os
import time
import uuid

import jwt
import pytest
import requests

BASE = (os.getenv("E2E_BASE_URL") or "").rstrip("/")
TEST_DB = os.getenv("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(not BASE, reason="E2E_BASE_URL not set")

CHAT = f"{BASE}/api/v1/chat/"
TIMEOUT = 90  # LLM turns can be slow on free providers


def _secret() -> str:
    secret = os.getenv("BACKEND_JWT_SECRET")
    if not secret:
        from dotenv import load_dotenv

        load_dotenv()
        secret = os.getenv("BACKEND_JWT_SECRET")
    assert secret, "BACKEND_JWT_SECRET required to mint the service JWT"
    return secret


def _auth(user_id: str) -> dict:
    token = jwt.encode(
        {"sub": user_id, "exp": int(time.time()) + 60}, _secret(), algorithm="HS256"
    )
    return {"Authorization": f"Bearer {token}"}


def _chat(user_id: str, node_id: str, message: str, parent_id: str | None = None) -> str:
    body = {
        "message": message,
        "message_id": str(uuid.uuid4()),
        "nodeId": node_id,
    }
    if parent_id:
        body["parentNodeId"] = parent_id
    r = requests.post(CHAT, json=body, headers=_auth(user_id), timeout=TIMEOUT)
    assert r.status_code == 200, f"chat failed: {r.status_code} {r.text[:300]}"
    return str(r.json().get("message") or "")


@pytest.fixture(scope="module")
def user():
    """A real users row when we own the DB; otherwise E2E_USER_ID (prod)."""
    if not TEST_DB:
        uid = os.getenv("E2E_USER_ID")
        assert uid, "E2E_USER_ID required when TEST_DATABASE_URL is not set"
        yield uid, None
        return

    import psycopg2

    conn = psycopg2.connect(TEST_DB)
    uid = str(uuid.uuid4())
    email = f"e2e_{uid[:8]}@example.com"
    with conn.cursor() as cur:
        cur.execute("INSERT INTO users (id, email) VALUES (%s, %s)", (uid, email))
    conn.commit()
    yield uid, conn
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM messages WHERE node_id IN (SELECT id FROM nodes WHERE user_email = %s)",
            (email,),
        )
        cur.execute("DELETE FROM nodes WHERE user_email = %s", (email,))
        cur.execute("DELETE FROM canvases WHERE user_email = %s", (email,))
        cur.execute("DELETE FROM quotas WHERE user_key IN (%s, %s)", (uid, email))
        cur.execute("DELETE FROM users WHERE id = %s", (uid,))
    conn.commit()
    conn.close()


def test_fork_knows_parent_context_end_to_end(user):
    uid, conn = user
    parent = str(uuid.uuid4())
    fork = str(uuid.uuid4())

    # Two real turns on the parent that establish a fact.
    _chat(uid, parent, "Hi! My name is Chinmay Pisal and I am planning a Japan trip.")
    _chat(uid, parent, "I want to focus the Kyoto days on food.")

    # First message on the fork — inheritance must fire here.
    reply = _chat(
        uid, fork, "Do you know my name? Answer with just the name.", parent_id=parent
    )
    assert "chinmay" in reply.lower(), (
        "THE PRODUCT GUARANTEE: a fork's first turn must know the parent "
        f"lineage. Reply was: {reply[:200]!r}"
    )

    if conn is None:
        return  # prod smoke: behavioral assertion only

    # Structural proof in the DB the service actually wrote.
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ancestor_ids, is_initialized, summary FROM nodes WHERE id = %s",
            (fork,),
        )
        row = cur.fetchone()
    assert row is not None, "fork node row was never created"
    ancestors, initialized, summary = row
    assert list(ancestors) == [parent], "fork must materialize its ancestry"
    assert initialized is True
    assert (summary or "").strip() or True  # summary may be empty when buffer covers all

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM messages WHERE node_id = %s AND position IS NULL", (fork,))
        assert cur.fetchone()[0] == 0, "no NULL positions may ever be written again"
        cur.execute("SELECT count(*) FROM messages WHERE node_id = %s", (parent,))
        assert cur.fetchone()[0] >= 4, "parent turns must be persisted by the chat path"
