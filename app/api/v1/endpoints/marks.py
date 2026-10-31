"""Mark routes: pen, arrow and highlight on the board and on note pages."""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response

from app.agent.store import mark_store
from app.agent.store.board_store import store
from app.api.auth import get_current_user_id
from app.api.limiter import BOARD_RATE, limiter, user_key
from app.schemas.board import MarkCreate

router = APIRouter()


@router.get("/{board_id}/marks")
@limiter.limit(BOARD_RATE, key_func=user_key)
def api_list_marks(board_id: str, request: Request, user_id: str = Depends(get_current_user_id)):
    email = store.email_for(user_id)
    with store.tx() as cur:
        return {"marks": mark_store.list_marks(cur, email, board_id)}


@router.post("/{board_id}/marks")
@limiter.limit(BOARD_RATE, key_func=user_key)
def api_create_mark(board_id: str, body: MarkCreate, request: Request, user_id: str = Depends(get_current_user_id)):
    email = store.email_for(user_id)
    with store.tx() as cur:
        mark, created = mark_store.create_mark(cur, email, board_id, body)
    return JSONResponse(mark, status_code=201 if created else 200)


@router.delete("/{board_id}/marks/{mark_id}")
@limiter.limit(BOARD_RATE, key_func=user_key)
def api_delete_mark(board_id: str, mark_id: str, request: Request, user_id: str = Depends(get_current_user_id)):
    email = store.email_for(user_id)
    with store.tx() as cur:
        mark_store.delete_mark(cur, email, board_id, mark_id)
    return Response(status_code=204)
