"""Edge writes for the V3 board: wire and answer strings (canPlug rules).

Functions take an open transaction cursor (BoardStore.tx()) and scope every query
by user_email. Missing, deleted or foreign rows are 404 (never 403).
"""
from __future__ import annotations

from app.agent.store.board_store import edge_to_json, lock_board, require_board, touch_board
from app.api.errors import ApiError
from app.schemas.board import EdgeCreate

_EDGE_COLS = "id, canvas_id, kind, from_node, to_node, created_at"


def _existing_own(cur, user_email: str, board_id: str, edge_id: str):
    cur.execute(f"SELECT {_EDGE_COLS}, user_email, deleted_at FROM edges WHERE id = %s", (edge_id,))
    row = cur.fetchone()
    if not row:
        return None
    if (row["user_email"] == user_email and row["canvas_id"] == board_id
            and row["deleted_at"] is None and row["kind"] in ("wire", "answer")):
        return edge_to_json(row)
    raise ApiError(409, "duplicate", "edge id is not available")


def _live_card(cur, user_email: str, board_id: str, card_id: str) -> dict:
    cur.execute(
        "SELECT id, kind, desk_id, ancestor_ids FROM nodes WHERE id = %s AND canvas_id = %s "
        "AND user_email = %s AND kind IS NOT NULL AND deleted_at IS NULL",
        (card_id, board_id, user_email),
    )
    row = cur.fetchone()
    if not row:
        raise ApiError(404, "not_found", "card not found")
    return row


def _can_plug(kind: str, src: dict, dst: dict) -> bool:
    if kind == "wire":
        return (src["kind"] == "source" and dst["kind"] == "chat"
                and src["desk_id"] != dst["id"])
    return (src["kind"] == "chat" and dst["kind"] == "chat" and src["id"] != dst["id"]
            and dst["id"] not in (src["ancestor_ids"] or [])
            and src["id"] not in (dst["ancestor_ids"] or []))


def create_edge(cur, user_email: str, board_id: str, body: EdgeCreate) -> tuple[dict, bool]:
    require_board(cur, board_id, user_email)
    existing = _existing_own(cur, user_email, board_id, body.id)
    if existing is not None:
        return existing, False

    # the lock makes card liveness stable and serialises concurrent duplicate posts
    lock_board(cur, board_id)
    existing = _existing_own(cur, user_email, board_id, body.id)
    if existing is not None:
        return existing, False

    src = _live_card(cur, user_email, board_id, body.from_)
    dst = _live_card(cur, user_email, board_id, body.to)
    if not _can_plug(body.kind, src, dst):
        raise ApiError(409, "not_allowed", f"a {body.kind} cannot connect these cards")

    cur.execute(
        "SELECT 1 FROM edges WHERE canvas_id = %s AND user_email = %s AND kind = %s "
        "AND from_node = %s AND to_node = %s AND deleted_at IS NULL",
        (board_id, user_email, body.kind, body.from_, body.to),
    )
    if cur.fetchone():
        raise ApiError(409, "duplicate", "this string already exists")

    cur.execute(
        "INSERT INTO edges (id, canvas_id, user_email, from_node, to_node, kind, data) "
        "VALUES (%s,%s,%s,%s,%s,%s,'{}'::jsonb) ON CONFLICT (id) DO NOTHING "
        f"RETURNING {_EDGE_COLS}",
        (body.id, board_id, user_email, body.from_, body.to, body.kind),
    )
    row = cur.fetchone()
    if row is None:
        raise ApiError(409, "duplicate", "edge id is not available")
    touch_board(cur, board_id)
    return edge_to_json(row), True


def delete_edge(cur, user_email: str, board_id: str, edge_id: str) -> None:
    require_board(cur, board_id, user_email)
    # serialise with delete_card/restore so we never overwrite a stamp restore relies on
    lock_board(cur, board_id)
    cur.execute(
        "SELECT kind FROM edges WHERE id = %s AND canvas_id = %s AND user_email = %s "
        "AND deleted_at IS NULL",
        (edge_id, board_id, user_email),
    )
    row = cur.fetchone()
    if not row:
        raise ApiError(404, "not_found")
    if row["kind"] not in ("wire", "answer"):
        raise ApiError(400, "invalid", "lineage edges cannot be deleted")
    cur.execute("UPDATE edges SET deleted_at = now() WHERE id = %s AND deleted_at IS NULL", (edge_id,))
    touch_board(cur, board_id)
