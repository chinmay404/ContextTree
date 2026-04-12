"""
Fork utilities — read-only summary endpoint.

The actual fork initialisation happens automatically inside the chat endpoint
(_init_fork_if_needed) on the first message of any branched node.
The separate POST /fork endpoint was removed as dead code.
"""
from fastapi import APIRouter, HTTPException, status, Request
from app.agent.store.PostgresStore import PostgresConversationStore
from app.api.limiter import limiter
from app.core.logger import logger

router = APIRouter()
mongo_store = PostgresConversationStore()


@router.get("/summary/{user_id}/{thread_id}")
@limiter.limit("20/minute")
async def get_thread_summary(user_id: str, thread_id: str, request: Request):
    """
    Return the rolling summary stored for a given thread.
    Useful for debugging and for the frontend's summary display.
    """
    try:
        summary = mongo_store.get_thread_summary(user_id, thread_id)
        return {"user_id": user_id, "thread_id": thread_id, "summary": summary}
    except Exception as e:
        logger.error(f"Error retrieving thread summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve summary: {str(e)}",
        )
