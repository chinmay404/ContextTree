"""Shared fixtures and seed helpers for the V3 board tests (need TEST_DATABASE_URL)."""
import json
import os
import random
import string
import time
import uuid

import psycopg2
import pytest
from langchain_core.messages import AIMessage

TEST_DB = os.getenv("TEST_DATABASE_URL")
requires_db = pytest.mark.skipif(not TEST_DB, reason="TEST_DATABASE_URL not set")

_B36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def _base36(n: int) -> str:
    out = ""
    while n:
        n, r = divmod(n, 36)
        out = _B36[r] + out
    return out or "0"


def new_id(prefix: str) -> str:
    rand = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{prefix}_{_base36(int(time.time() * 1000))}_{rand}"


def unit_emb(i: int) -> list[float]:
    """768-dim basis vector e_i (deterministic stand-in for an embedding)."""
    v = [0.0] * 768
    v[i] = 1.0
    return v


def _vec(emb: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in emb) + "]"


class FakeLLM:
    def __init__(self, text: str = "FAKE SUMMARY"):
        self.text = text
        self.calls: list = []

    def with_config(self, *a, **k):
        return self

    def invoke(self, prompt):
        self.calls.append(prompt)
        return AIMessage(content=self.text)


def raising_llm(*a, **k):
    raise AssertionError("LLM must not be called")


@pytest.fixture
def db():
    conn = psycopg2.connect(TEST_DB)
    conn.autocommit = True
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def user(db):
    email = f"v3_{uuid.uuid4().hex[:12]}@example.com"
    with db.cursor() as cur:
        cur.execute("INSERT INTO users (email, name) VALUES (%s, %s)", (email, "V3 Test"))
    try:
        yield email
    finally:
        with db.cursor() as cur:
            cur.execute(
                "DELETE FROM file_chunks WHERE file_id IN "
                "(SELECT id FROM external_files WHERE user_email = %s)", (email,))
            for table in ("external_files", "messages", "edges", "nodes", "canvases"):
                cur.execute(f"DELETE FROM {table} WHERE user_email = %s", (email,))
            cur.execute("DELETE FROM quotas WHERE user_key = %s", (email,))
            cur.execute("DELETE FROM users WHERE email = %s", (email,))


def seed_board(db, email, board_id=None, *, version=3, title="Test", deleted=False) -> str:
    board_id = board_id or new_id("b")
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO canvases (id, user_email, data, title, board_version, deleted_at) "
            "VALUES (%s, %s, %s::jsonb, %s, %s, CASE WHEN %s THEN now() END)",
            (board_id, email, '{"viewport": null}', title, version, deleted))
    return board_id


def seed_card(db, email, board_id, card_id=None, *, kind="chat", source_type=None, status=None,
              desk_id=None, pile_id=None, parent_id=None, data=None, x=0.0, y=0.0,
              deleted_at=None, created_at=None) -> str:
    card_id = card_id or new_id("c")
    payload = json.dumps(data if data is not None else {})
    with db.cursor() as cur:
        if parent_id:
            cur.execute("SELECT ancestor_ids FROM nodes WHERE id = %s", (parent_id,))
            ancestors = list(cur.fetchone()[0] or []) + [parent_id]
            parent_node_id, is_initialized, is_primary = parent_id, False, False
        else:
            ancestors, parent_node_id, is_initialized = [], None, True
            is_primary = kind == "chat"
        cur.execute(
            "INSERT INTO nodes (id, canvas_id, user_email, data, kind, source_type, status, "
            "desk_id, pile_id, x, y, deleted_at, parent_node_id, ancestor_ids, is_initialized, "
            "is_primary, created_at) "
            "VALUES (%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,COALESCE(%s, now()))",
            (card_id, board_id, email, payload, kind, source_type, status, desk_id, pile_id,
             x, y, deleted_at, parent_node_id, ancestors, is_initialized, is_primary, created_at))
    return card_id


def seed_edge(db, email, board_id, from_id, to_id, kind, edge_id=None, *, meta=None, data=None,
              deleted_at=None) -> str:
    edge_id = edge_id or new_id("e")
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO edges (id, canvas_id, user_email, from_node, to_node, kind, meta, data, "
            "deleted_at) VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s)",
            (edge_id, board_id, email, from_id, to_id, kind,
             json.dumps(meta) if meta is not None else None,
             json.dumps(data) if data is not None else None, deleted_at))
    return edge_id


def seed_message(db, email, node_id, role, content, position, *, inherited=False, msg_id=None,
                 embedding=None) -> str:
    msg_id = msg_id or new_id("m")
    with db.cursor() as cur:
        cur.execute("SELECT canvas_id FROM nodes WHERE id = %s", (node_id,))
        row = cur.fetchone()
        canvas_id = row[0] if row else None
        cur.execute(
            "INSERT INTO messages (id, node_id, canvas_id, user_email, role, content, position, "
            "inherited, embedding) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)",
            (msg_id, node_id, canvas_id, email, role, content, position, inherited,
             _vec(embedding) if embedding is not None else None))
    return msg_id
