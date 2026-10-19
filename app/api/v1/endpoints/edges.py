"""Edge routes (contract Routes rows 11-12)."""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response

from app.agent.store import edge_store
from app.agent.store.board_store import store
from app.api.auth import get_current_user_id
from app.api.limiter import BOARD_RATE, limiter, user_key
from app.schemas.board import EdgeCreate

router = APIRouter()


@router.post("/{board_id}/edges")
@limiter.limit(BOARD_RATE, key_func=user_key)
def api_create_edge(
    board_id: str, body: EdgeCreate, request: Request,
    user_id: str = Depends(get_current_user_id),
):
    email = store.email_for(user_id)
    with store.tx() as cur:
        edge, created = edge_store.create_edge(cur, email, board_id, body)
    return JSONResponse(edge, status_code=201 if created else 200)


@router.delete("/{board_id}/edges/{edge_id}")
@limiter.limit(BOARD_RATE, key_func=user_key)
def api_delete_edge(
    board_id: str, edge_id: str, request: Request,
    user_id: str = Depends(get_current_user_id),
):
    email = store.email_for(user_id)
    with store.tx() as cur:
        edge_store.delete_edge(cur, email, board_id, edge_id)
    return Response(status_code=204)
