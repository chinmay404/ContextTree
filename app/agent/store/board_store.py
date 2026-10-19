"""
Board-level persistence for the V3 engine: transactions, card/edge row mapping,
and the live-row guards every board route shares.

Tenancy key is user_email. A missing, deleted or foreign row is a 404 (never 403).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, ContextManager, Optional

from pgvector.psycopg2 import register_vector
from psycopg2.extras import RealDictCursor

from app.agent.store.PostgresStore import PostgresConversationStore
from app.api.errors import ApiError
from app.agent.utils.text import clip_text

LAST_MESSAGE_LIMIT = 280

# Card columns plus desk/pile ids masked to NULL while the target card is deleted
# (the stored column keeps the id so a restore puts the link back).
CARD_SELECT = (
    "SELECT n.id, n.canvas_id, n.user_email, n.kind, n.source_type, n.x, n.y, "
    "n.parent_node_id, n.forked_from_message_id, n.status, n.data, "
    "n.created_at, n.updated_at, n.deleted_at, n.desk_id, n.pile_id, "
    "CASE WHEN d.deleted_at IS NULL THEN n.desk_id END AS \"deskId\", "
    "CASE WHEN p.deleted_at IS NULL THEN n.pile_id END AS \"pileId\" "
    "FROM nodes n "
    "LEFT JOIN nodes d ON d.id = n.desk_id "
    "LEFT JOIN nodes p ON p.id = n.pile_id"
)

# Keys the engine keeps inside nodes.data that must never reach the client.
_INTERNAL_DATA_KEYS = ("memoryFacts",)


def _iso(value) -> Optional[str]:
    return value.isoformat() if value is not None else None


class BoardStore(PostgresConversationStore):
    @contextmanager
    def tx(self) -> ContextManager[RealDictCursor]:
        """One transaction on a pooled connection; commits on clean exit."""
        with self._get_conn() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            try:
                yield cur
            finally:
                cur.close()

    def email_for(self, user_id: str) -> str:
        return self.resolve_user_email(user_id)


store = BoardStore()


# ── Row mapping ───────────────────────────────────────────────────────────────

def card_to_json(row: dict) -> dict:
    data = dict(row.get("data") or {})
    for k in _INTERNAL_DATA_KEYS:
        data.pop(k, None)
    out = {
        "id": row["id"],
        "boardId": row["canvas_id"],
        "kind": row["kind"],
        "sourceType": row.get("source_type"),
        "x": row["x"] if row.get("x") is not None else 0,
        "y": row["y"] if row.get("y") is not None else 0,
        "deskId": row.get("deskId"),
        "pileId": row.get("pileId"),
        "parentId": row.get("parent_node_id"),
        "forkedFromMessageId": row.get("forked_from_message_id"),
        "status": row.get("status"),
        "data": data,
        "createdAt": _iso(row.get("created_at")),
        "updatedAt": _iso(row.get("updated_at")),
    }
    if "messageCount" in row:
        out["messageCount"] = row["messageCount"]
    if "lastMessage" in row:
        out["lastMessage"] = row["lastMessage"]
    return out


def edge_to_json(row: dict) -> dict:
    return {
        "id": row["id"],
        "boardId": row["canvas_id"],
        "kind": row["kind"],
        "from": row["from_node"],
        "to": row["to_node"],
        "createdAt": _iso(row.get("created_at")),
    }


# ── Guards ────────────────────────────────────────────────────────────────────

def require_board(cur, board_id: str, user_email: str) -> dict:
    cur.execute(
        "SELECT * FROM canvases WHERE id = %s AND user_email = %s "
        "AND board_version = 3 AND deleted_at IS NULL",
        (board_id, user_email),
    )
    row = cur.fetchone()
    if not row:
        raise ApiError(404, "not_found")
    return row


def require_card(cur, board_id: str, card_id: str, user_email: str) -> dict:
    cur.execute(
        "SELECT * FROM nodes WHERE id = %s AND canvas_id = %s AND user_email = %s "
        "AND kind IS NOT NULL AND deleted_at IS NULL",
        (card_id, board_id, user_email),
    )
    row = cur.fetchone()
    if not row:
        raise ApiError(404, "not_found")
    return row


def get_card_row(cur, card_id: str) -> Optional[dict]:
    """Raw row, no filters; the caller checks owner, board and deleted_at."""
    cur.execute("SELECT * FROM nodes WHERE id = %s", (card_id,))
    return cur.fetchone()


# ── Reads ─────────────────────────────────────────────────────────────────────

def fetch_card_json(cur, board_id: str, card_id: str) -> dict:
    cur.execute(
        CARD_SELECT + " WHERE n.id = %s AND n.canvas_id = %s AND n.kind IS NOT NULL "
        "AND n.deleted_at IS NULL",
        (card_id, board_id),
    )
    row = cur.fetchone()
    if not row:
        raise ApiError(404, "not_found")
    if row["kind"] == "chat":
        _attach_message_info(cur, [row])
    return card_to_json(row)


def _attach_message_info(cur, rows: list[dict]) -> None:
    ids = [r["id"] for r in rows if r["kind"] == "chat"]
    for r in rows:
        if r["kind"] == "chat":
            r["messageCount"] = 0
            r["lastMessage"] = None
    if not ids:
        return
    cur.execute(
        "SELECT node_id, count(*) AS n FROM messages "
        "WHERE node_id = ANY(%s) AND inherited = false GROUP BY node_id",
        (ids,),
    )
    counts = {r["node_id"]: r["n"] for r in cur.fetchall()}
    cur.execute(
        "SELECT DISTINCT ON (node_id) node_id, role, content FROM messages "
        "WHERE node_id = ANY(%s) AND inherited = false "
        "ORDER BY node_id, position DESC",
        (ids,),
    )
    last = {r["node_id"]: r for r in cur.fetchall()}
    for r in rows:
        if r["kind"] != "chat":
            continue
        r["messageCount"] = int(counts.get(r["id"], 0))
        m = last.get(r["id"])
        if m:
            r["lastMessage"] = {
                "role": "user" if m["role"] == "user" else "assistant",
                "content": clip_text(m["content"] or "", LAST_MESSAGE_LIMIT),
            }


def list_cards_json(cur, board_id: str) -> list[dict]:
    cur.execute(
        CARD_SELECT + " WHERE n.canvas_id = %s AND n.kind IS NOT NULL AND n.deleted_at IS NULL "
        "ORDER BY n.created_at, n.id",
        (board_id,),
    )
    rows = cur.fetchall()
    _attach_message_info(cur, rows)
    return [card_to_json(r) for r in rows]


def list_edges_json(cur, board_id: str) -> list[dict]:
    cur.execute(
        "SELECT e.id, e.canvas_id, e.kind, e.from_node, e.to_node, e.created_at FROM edges e "
        "JOIN nodes a ON a.id = e.from_node AND a.deleted_at IS NULL "
        "JOIN nodes b ON b.id = e.to_node AND b.deleted_at IS NULL "
        "WHERE e.canvas_id = %s AND e.deleted_at IS NULL AND e.kind IS NOT NULL "
        "ORDER BY e.created_at, e.id",
        (board_id,),
    )
    return [edge_to_json(r) for r in cur.fetchall()]


def get_board_json(cur, board_id: str) -> dict:
    cur.execute("SELECT * FROM canvases WHERE id = %s", (board_id,))
    b = cur.fetchone()
    if not b:
        raise ApiError(404, "not_found")
    return {
        "id": b["id"],
        "title": b["title"],
        "viewport": (b.get("data") or {}).get("viewport"),
        "cards": list_cards_json(cur, board_id),
        "edges": list_edges_json(cur, board_id),
        "createdAt": _iso(b["created_at"]),
        "updatedAt": _iso(b["updated_at"]),
    }


# ── Write helpers ─────────────────────────────────────────────────────────────

def lock_board(cur, board_id: str) -> None:
    cur.execute("SELECT pg_advisory_xact_lock(hashtext('board:' || %s))", (board_id,))


def live_card_count(cur, board_id: str) -> int:
    cur.execute(
        "SELECT count(*) AS n FROM nodes WHERE canvas_id = %s "
        "AND kind IS NOT NULL AND deleted_at IS NULL",
        (board_id,),
    )
    return int(cur.fetchone()["n"])


def touch_board(cur, board_id: str) -> None:
    cur.execute("UPDATE canvases SET updated_at = now() WHERE id = %s", (board_id,))
