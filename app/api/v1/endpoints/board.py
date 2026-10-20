"""Board CRUD routes (contract Routes rows 1-5)."""
import json

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse

from app.agent.store.board_store import (
    fail_stale_sources,
    get_board_json,
    require_board,
    store,
    touch_board,
)
from app.api.auth import get_current_user_id
from app.api.errors import ApiError
from app.api.limiter import BOARD_RATE, limiter, user_key
from app.schemas.board import BoardCreate, BoardPatch, default_chat_data

router = APIRouter()


@router.get("")
@limiter.limit(BOARD_RATE, key_func=user_key)
def api_list_boards(request: Request, user_id: str = Depends(get_current_user_id)):
    email = store.email_for(user_id)
    with store.tx() as cur:
        cur.execute(
            "SELECT c.id, c.title, c.updated_at, "
            "(SELECT count(*) FROM nodes n WHERE n.canvas_id = c.id "
            " AND n.kind IS NOT NULL AND n.deleted_at IS NULL) AS card_count "
            "FROM canvases c WHERE c.user_email = %s AND c.board_version = 3 "
            "AND c.deleted_at IS NULL ORDER BY c.updated_at DESC, c.id",
            (email,),
        )
        rows = cur.fetchall()
    return {
        "boards": [
            {
                "id": r["id"],
                "title": r["title"],
                "updatedAt": r["updated_at"].isoformat(),
                "cardCount": int(r["card_count"]),
            }
            for r in rows
        ]
    }


@router.post("")
@limiter.limit(BOARD_RATE, key_func=user_key)
def api_create_board(
    body: BoardCreate, request: Request, user_id: str = Depends(get_current_user_id)
):
    email = store.email_for(user_id)
    with store.tx() as cur:
        cur.execute(
            "INSERT INTO canvases (id, user_email, data, title, board_version) "
            "VALUES (%s, %s, %s::jsonb, %s, 3) ON CONFLICT (id) DO NOTHING RETURNING id",
            (body.id, email, '{"viewport": null}', body.title),
        )
        created = cur.fetchone() is not None
        if created:
            cur.execute(
                "INSERT INTO nodes (id, canvas_id, user_email, data, kind, x, y, "
                "is_primary, is_initialized) "
                "VALUES (%s, %s, %s, %s::jsonb, 'chat', 0, 0, true, true) "
                "ON CONFLICT (id) DO NOTHING RETURNING id",
                (f"{body.id}_q1", body.id, email, json.dumps(default_chat_data(body.title))),
            )
            if cur.fetchone() is None:
                # the first-chat id is taken by someone else; roll the whole board back
                raise ApiError(409, "duplicate", "board id is not available")
        else:
            # Re-post of the caller's own live V3 board is idempotent; anything else
            # (another user's id, own deleted board, V2 canvas) is a duplicate.
            cur.execute(
                "SELECT 1 FROM canvases WHERE id = %s AND user_email = %s "
                "AND board_version = 3 AND deleted_at IS NULL",
                (body.id, email),
            )
            if cur.fetchone() is None:
                raise ApiError(409, "duplicate", "board id is not available")
        board = get_board_json(cur, body.id)
    return JSONResponse(board, status_code=201 if created else 200)


@router.get("/{board_id}")
@limiter.limit(BOARD_RATE, key_func=user_key)
def api_get_board(
    board_id: str, request: Request, user_id: str = Depends(get_current_user_id)
):
    email = store.email_for(user_id)
    with store.tx() as cur:
        require_board(cur, board_id, email)
        fail_stale_sources(cur, board_id)
        return get_board_json(cur, board_id)


@router.patch("/{board_id}")
@limiter.limit(BOARD_RATE, key_func=user_key)
def api_patch_board(
    board_id: str,
    body: BoardPatch,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    email = store.email_for(user_id)
    with store.tx() as cur:
        require_board(cur, board_id, email)
        if "title" in body.model_fields_set and body.title is not None:
            cur.execute("UPDATE canvases SET title = %s WHERE id = %s", (body.title, board_id))
        if "viewport" in body.model_fields_set:
            viewport = body.viewport.model_dump() if body.viewport is not None else None
            cur.execute(
                "UPDATE canvases SET data = jsonb_set(coalesce(data, '{}'::jsonb), "
                "'{viewport}', %s::jsonb) WHERE id = %s",
                (json.dumps(viewport), board_id),
            )
        touch_board(cur, board_id)
    return {"ok": True}


@router.delete("/{board_id}", status_code=204)
@limiter.limit(BOARD_RATE, key_func=user_key)
def api_delete_board(
    board_id: str, request: Request, user_id: str = Depends(get_current_user_id)
):
    email = store.email_for(user_id)
    with store.tx() as cur:
        require_board(cur, board_id, email)
        cur.execute(
            "UPDATE canvases SET deleted_at = now(), updated_at = now() WHERE id = %s",
            (board_id,),
        )
    return Response(status_code=204)
