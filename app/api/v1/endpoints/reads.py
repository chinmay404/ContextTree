"""Read set route (contract Routes row 14): what a chat reads."""
from fastapi import APIRouter, Depends, Request

from app.agent.reads import resolve_reads
from app.agent.store.board_store import store
from app.api.auth import get_current_user_id
from app.api.limiter import BOARD_RATE, limiter, user_key

router = APIRouter()


@router.get("/{board_id}/cards/{card_id}/reads")
@limiter.limit(BOARD_RATE, key_func=user_key)
def api_get_reads(
    board_id: str, card_id: str, request: Request,
    user_id: str = Depends(get_current_user_id),
):
    email = store.email_for(user_id)
    return resolve_reads(store, email, board_id, card_id).to_json()
