"""Marks for the V3 board: pen, arrow and highlight, on the board or on a note page.

Same rules as edges: functions take an open transaction cursor, every query is scoped by user_email,
and missing, deleted or foreign rows are 404. A mark on a deleted card is hidden with it and comes
back if the card is restored.
"""
from __future__ import annotations

from app.agent.store.board_store import _iso, dumps_json, require_board, touch_board
from app.api.errors import ApiError
from app.schemas.board import MarkCreate

MAX_MARKS_PER_BOARD = 2000
_COLS = "id, canvas_id, kind, color, size, points, anchor_id, note_id, touches, created_at"


def mark_to_json(row: dict) -> dict:
    return {
        "id": row["id"],
        "boardId": row["canvas_id"],
        "kind": row["kind"],
        "color": row["color"],
        "size": row["size"],
        "points": row["points"],
        "anchorId": row["anchor_id"],
        "noteId": row["note_id"],
        "touches": list(row["touches"] or []),
        "createdAt": _iso(row["created_at"]),
    }


def list_marks(cur, user_email: str, board_id: str) -> list[dict]:
    require_board(cur, board_id, user_email)
    cur.execute(
        "SELECT m.id, m.canvas_id, m.kind, m.color, m.size, m.points, m.anchor_id, m.note_id, m.touches, m.created_at "
        "FROM board_marks m "
        "LEFT JOIN nodes a ON a.id = m.anchor_id LEFT JOIN nodes n ON n.id = m.note_id "
        "WHERE m.canvas_id = %s AND m.user_email = %s AND m.deleted_at IS NULL "
        "AND a.deleted_at IS NULL AND n.deleted_at IS NULL ORDER BY m.created_at, m.id",
        (board_id, user_email),
    )
    return [mark_to_json(r) for r in cur.fetchall()]


def _require_card(cur, user_email: str, board_id: str, card_id: str, note: bool = False) -> None:
    cur.execute(
        "SELECT source_type FROM nodes WHERE id = %s AND canvas_id = %s AND user_email = %s "
        "AND kind IS NOT NULL AND deleted_at IS NULL",
        (card_id, board_id, user_email),
    )
    row = cur.fetchone()
    if not row:
        raise ApiError(404, "not_found", "card not found")
    if note and row["source_type"] != "note":
        raise ApiError(400, "invalid", "noteId must be a note")


def create_mark(cur, user_email: str, board_id: str, body: MarkCreate) -> tuple[dict, bool]:
    require_board(cur, board_id, user_email)
    cur.execute(f"SELECT {_COLS}, user_email, deleted_at FROM board_marks WHERE id = %s", (body.id,))
    existing = cur.fetchone()
    if existing:  # a retry of the same save
        if existing["user_email"] == user_email and existing["canvas_id"] == board_id and existing["deleted_at"] is None:
            return mark_to_json(existing), False
        raise ApiError(409, "duplicate", "mark id is not available")
    if body.anchor_id:
        _require_card(cur, user_email, board_id, body.anchor_id)
    if body.note_id:
        _require_card(cur, user_email, board_id, body.note_id, note=True)
    cur.execute("SELECT count(*) AS n FROM board_marks WHERE canvas_id = %s AND deleted_at IS NULL", (board_id,))
    if cur.fetchone()["n"] >= MAX_MARKS_PER_BOARD:
        raise ApiError(409, "mark_limit", f"a board holds at most {MAX_MARKS_PER_BOARD} marks")
    cur.execute(
        "INSERT INTO board_marks (id, canvas_id, user_email, kind, color, size, points, anchor_id, note_id, touches) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s) ON CONFLICT (id) DO NOTHING "
        f"RETURNING {_COLS}",
        (body.id, board_id, user_email, body.kind, body.color, body.size, dumps_json(body.points),
         body.anchor_id, body.note_id, body.touches),
    )
    row = cur.fetchone()
    if row is None:
        raise ApiError(409, "duplicate", "mark id is not available")
    touch_board(cur, board_id)
    return mark_to_json(row), True


def delete_mark(cur, user_email: str, board_id: str, mark_id: str) -> None:
    require_board(cur, board_id, user_email)
    cur.execute(
        "UPDATE board_marks SET deleted_at = now() WHERE id = %s AND canvas_id = %s AND user_email = %s "
        "AND deleted_at IS NULL",
        (mark_id, board_id, user_email),
    )
    if cur.rowcount == 0:
        raise ApiError(404, "not_found", "mark not found")
    touch_board(cur, board_id)
