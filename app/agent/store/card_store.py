"""Card writes for the V3 board: create, patch, positions batch, message list.

All functions take an open transaction cursor (BoardStore.tx()) and scope every
query by user_email. Missing, deleted or foreign rows are 404 (never 403).
"""
from __future__ import annotations

import json
from typing import Optional

from psycopg2.extras import execute_values

from app.agent.store.board_store import (
    _INTERNAL_DATA_KEYS,
    fetch_card_json,
    live_card_count,
    lock_board,
    require_board,
    require_card,
    touch_board,
)
from app.api.errors import ApiError
from app.schemas.board import (
    MAX_CARDS_PER_BOARD,
    MAX_POSITIONS,
    CardCreate,
    CardPatch,
    Position,
    default_chat_data,
)


def _public_data(data: dict) -> dict:
    return {k: v for k, v in data.items() if k not in _INTERNAL_DATA_KEYS}


def _live_card(cur, user_email: str, board_id: str, card_id: str) -> Optional[dict]:
    cur.execute(
        "SELECT * FROM nodes WHERE id = %s AND canvas_id = %s AND user_email = %s "
        "AND kind IS NOT NULL AND deleted_at IS NULL",
        (card_id, board_id, user_email),
    )
    return cur.fetchone()


def _check_desk(cur, user_email: str, board_id: str, desk_id: str) -> None:
    row = _live_card(cur, user_email, board_id, desk_id)
    if not row or row["kind"] != "chat":
        raise ApiError(400, "invalid", "deskId must be a live chat on this board")


def _check_pile(cur, user_email: str, board_id: str, pile_id: str) -> None:
    row = _live_card(cur, user_email, board_id, pile_id)
    if not row or row["kind"] != "source" or row["source_type"] != "pile":
        raise ApiError(400, "invalid", "pileId must be a live pile on this board")


def _validate_create(body: CardCreate) -> None:
    if body.kind == "chat":
        if body.source_type is not None or body.status is not None:
            raise ApiError(400, "invalid", "a chat takes no sourceType or status")
    else:
        if body.source_type is None:
            raise ApiError(400, "invalid", "a source needs sourceType")
        if body.parent_id is not None or body.forked_from_message_id is not None:
            raise ApiError(400, "invalid", "only a chat can have a parent")


def _chat_data(data: dict) -> dict:
    data = _public_data(data)
    if "advanced" in data and not isinstance(data["advanced"], dict):
        raise ApiError(400, "invalid", "advanced must be an object")
    merged = default_chat_data(data.get("name", ""))
    for k, v in data.items():
        if k == "advanced":
            merged["advanced"].update(v)  # one level deep over the defaults
        else:
            merged[k] = v
    return merged


def _existing_own(cur, user_email: str, board_id: str, card_id: str) -> Optional[dict]:
    """The caller's live card with this id on this board; 409 if the id is taken otherwise."""
    cur.execute(
        "SELECT canvas_id, user_email, kind, deleted_at FROM nodes WHERE id = %s", (card_id,)
    )
    row = cur.fetchone()
    if not row:
        return None
    if (row["user_email"] == user_email and row["canvas_id"] == board_id
            and row["kind"] is not None and row["deleted_at"] is None):
        return fetch_card_json(cur, board_id, card_id)
    raise ApiError(409, "duplicate", "card id is not available")


def create_card(cur, user_email: str, board_id: str, body: CardCreate) -> tuple[dict, bool]:
    _validate_create(body)
    require_board(cur, board_id, user_email)

    existing = _existing_own(cur, user_email, board_id, body.id)
    if existing is not None:
        return existing, False

    lock_board(cur, board_id)
    # a concurrent POST of the same id may have committed while we waited for the lock
    existing = _existing_own(cur, user_email, board_id, body.id)
    if existing is not None:
        return existing, False
    if live_card_count(cur, board_id) >= MAX_CARDS_PER_BOARD:
        raise ApiError(409, "card_limit", f"a board holds at most {MAX_CARDS_PER_BOARD} cards")

    if body.desk_id is not None:
        _check_desk(cur, user_email, board_id, body.desk_id)
    if body.pile_id is not None:
        _check_pile(cur, user_email, board_id, body.pile_id)
    parent = None
    if body.parent_id is not None:
        parent = _live_card(cur, user_email, board_id, body.parent_id)
        if not parent:
            raise ApiError(404, "not_found", "parent chat not found")
        if parent["kind"] != "chat":
            raise ApiError(400, "invalid", "parentId must be a chat")

    if body.kind == "chat":
        data = _chat_data(body.data)
        status = None
    else:
        data = _public_data(body.data)
        status = "ready" if body.source_type in ("note", "pile") else (body.status or "processing")

    ancestors: list[str] = []
    parent_id = forked = None
    is_initialized, is_primary = True, body.kind == "chat"
    if parent:
        ancestors = list(parent.get("ancestor_ids") or []) + [parent["id"]]
        parent_id, forked = parent["id"], body.forked_from_message_id
        is_initialized, is_primary = False, False

    cur.execute(
        "INSERT INTO nodes (id, canvas_id, user_email, data, kind, source_type, status, x, y, "
        "desk_id, pile_id, parent_node_id, forked_from_message_id, ancestor_ids, "
        "is_initialized, is_primary) "
        "VALUES (%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
        "ON CONFLICT (id) DO NOTHING RETURNING id",
        (body.id, board_id, user_email, json.dumps(data), body.kind, body.source_type, status,
         body.x, body.y, body.desk_id, body.pile_id, parent_id, forked, ancestors,
         is_initialized, is_primary),
    )
    if cur.fetchone() is None:
        raise ApiError(409, "duplicate", "card id is not available")
    if parent:
        cur.execute(
            "INSERT INTO edges (id, canvas_id, user_email, from_node, to_node, kind, data) "
            "VALUES (%s,%s,%s,%s,%s,'lineage','{}'::jsonb) ON CONFLICT (id) DO NOTHING",
            ("lin_" + body.id, board_id, user_email, parent["id"], body.id),
        )
    touch_board(cur, board_id)
    return fetch_card_json(cur, board_id, body.id), True


def patch_card(cur, user_email: str, board_id: str, card_id: str, body: CardPatch) -> dict:
    require_board(cur, board_id, user_email)
    card = require_card(cur, board_id, card_id, user_email)
    fields = body.model_fields_set
    sets: list[str] = []
    args: list = []
    if "desk_id" in fields:
        if body.desk_id is not None:
            _check_desk(cur, user_email, board_id, body.desk_id)
        sets.append("desk_id = %s")
        args.append(body.desk_id)
    if "pile_id" in fields:
        if body.pile_id is not None:
            _check_pile(cur, user_email, board_id, body.pile_id)
        sets.append("pile_id = %s")
        args.append(body.pile_id)
    if "status" in fields:
        if card["kind"] == "chat":
            raise ApiError(400, "invalid", "a chat has no status")
        if body.status is None:
            raise ApiError(400, "invalid", "status cannot be null")
        sets.append("status = %s")
        args.append(body.status)
    if "data" in fields:
        if body.data is None:
            raise ApiError(400, "invalid", "data cannot be null")
        patch = _public_data(body.data)
        if card["kind"] == "chat" and "advanced" in patch and not isinstance(patch["advanced"], dict):
            raise ApiError(400, "invalid", "advanced must be an object")
        if card["kind"] == "chat" and "advanced" in patch:
            # advanced merges one level deep so a partial patch keeps the other keys
            sets.append(
                "data = jsonb_set(coalesce(data, '{}'::jsonb) || (%s::jsonb - 'advanced'), "
                "'{advanced}', CASE WHEN jsonb_typeof(data->'advanced') = 'object' THEN data->'advanced' ELSE '{}'::jsonb END || %s::jsonb)"
            )
            args.append(json.dumps(patch))
            args.append(json.dumps(patch["advanced"]))
        else:
            sets.append("data = coalesce(data, '{}'::jsonb) || %s::jsonb")
            args.append(json.dumps(patch))
    if sets:
        cur.execute(
            f"UPDATE nodes SET {', '.join(sets)}, updated_at = now() "
            "WHERE id = %s AND canvas_id = %s AND user_email = %s",
            (*args, card_id, board_id, user_email),
        )
        touch_board(cur, board_id)
    return fetch_card_json(cur, board_id, card_id)


def save_positions(cur, user_email: str, board_id: str, positions: list[Position]) -> int:
    require_board(cur, board_id, user_email)
    if not positions:
        return 0
    # execute_values owns the single %s; bind board/user up front (escaping any %).
    scope = cur.mogrify(
        "WHERE n.id = v.id AND n.canvas_id = %s AND n.user_email = %s AND n.deleted_at IS NULL",
        (board_id, user_email),
    ).decode().replace("%", "%%")
    execute_values(
        cur,
        "UPDATE nodes n SET x = v.x, y = v.y, updated_at = now() "
        "FROM (VALUES %s) AS v(id, x, y) " + scope,
        [(p.id, p.x, p.y) for p in positions],
        template="(%s, %s::double precision, %s::double precision)",
        page_size=MAX_POSITIONS,  # one page, so rowcount covers every row
    )
    return cur.rowcount



def list_messages(cur, chat_id: str, user_email: str) -> list[dict]:
    cur.execute(
        'SELECT id, role, content, position, "timestamp" AS created_at FROM messages '
        "WHERE node_id = %s AND user_email = %s AND inherited = false ORDER BY position, id",
        (chat_id, user_email),
    )
    return [
        {
            "id": r["id"],
            "role": "user" if r["role"] == "user" else "assistant",
            "content": r["content"] or "",
            "position": r["position"],
            "createdAt": r["created_at"].isoformat() if r["created_at"] is not None else None,
        }
        for r in cur.fetchall()
    ]
