"""
Fork utilities — read-only summary endpoint.

The actual fork initialisation happens automatically inside the chat endpoint
(_init_fork_if_needed) on the first message of any branched node.
The separate POST /fork endpoint was removed as dead code.

Identity comes from the verified service JWT (app/api/auth.py); the old
/summary/{user_id}/{thread_id} path — which let any caller name any user —
is gone.
"""
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.agent.store.PostgresStore import PostgresConversationStore
from app.api.auth import get_current_user_id
from app.api.limiter import limiter
from app.core.logger import logger

router = APIRouter()
store = PostgresConversationStore()


@router.get("/summary/{thread_id}")
@limiter.limit("20/minute")
async def get_thread_summary(
    thread_id: str,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    """
    Return the rolling summary stored for one of the caller's threads.
    """
    try:
        summary = store.get_thread_summary(user_id, thread_id)
        return {"thread_id": thread_id, "summary": summary}
    except Exception as e:
        logger.error(f"Error retrieving thread summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve summary",
        )
