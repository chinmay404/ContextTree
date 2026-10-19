"""Card routes (contract Routes rows 6-7, 10, 13)."""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.agent.store import card_store
from app.agent.store.board_store import require_board, require_card, store
from app.api.auth import get_current_user_id
from app.api.errors import ApiError
from app.api.limiter import BOARD_RATE, limiter, user_key
from app.schemas.board import CardCreate, CardPatch, PositionsPatch

router = APIRouter()


@router.post("/{board_id}/cards")
@limiter.limit(BOARD_RATE, key_func=user_key)
def api_create_card(
    board_id: str, body: CardCreate, request: Request,
    user_id: str = Depends(get_current_user_id),
):
    email = store.email_for(user_id)
    with store.tx() as cur:
        card, created = card_store.create_card(cur, email, board_id, body)
    return JSONResponse(card, status_code=201 if created else 200)


@router.patch("/{board_id}/cards/{card_id}")
@limiter.limit(BOARD_RATE, key_func=user_key)
def api_patch_card(
    board_id: str, card_id: str, body: CardPatch, request: Request,
    user_id: str = Depends(get_current_user_id),
):
    email = store.email_for(user_id)
    with store.tx() as cur:
        return card_store.patch_card(cur, email, board_id, card_id, body)


@router.patch("/{board_id}/positions")
@limiter.limit(BOARD_RATE, key_func=user_key)
def api_save_positions(
    board_id: str, body: PositionsPatch, request: Request,
    user_id: str = Depends(get_current_user_id),
):
    email = store.email_for(user_id)
    with store.tx() as cur:
        n = card_store.save_positions(cur, email, board_id, body.positions)
    return {"updated": n}


@router.get("/{board_id}/cards/{card_id}/messages")
@limiter.limit(BOARD_RATE, key_func=user_key)
def api_card_messages(
    board_id: str, card_id: str, request: Request,
    user_id: str = Depends(get_current_user_id),
):
    email = store.email_for(user_id)
    with store.tx() as cur:
        require_board(cur, board_id, email)
        card = require_card(cur, board_id, card_id, email)
        if card["kind"] != "chat":
            raise ApiError(400, "invalid", "messages exist only on chats")
        return {"messages": card_store.list_messages(cur, card_id, email)}
