"""
Fork thread endpoint.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from app.agent.store.MongoStore import MongoConversationStore
from app.api.limiter import limiter
from fastapi import Request
from app.core.logger import logger


router = APIRouter()
mongo_store = MongoConversationStore()


class ForkRequest(BaseModel):
    """Schema for forking a thread."""
    user_id: str = Field(..., description="User identifier")
    source_thread_id: str = Field(..., description="Source thread to fork from")
    new_thread_id: str = Field(..., description="New thread identifier")
    fork_at_message_id: str = Field(..., description="Message ID to fork at (inclusive)")


@router.post("/")
@limiter.limit("10/minute")
async def fork_thread(fork_request: ForkRequest, request: Request):
    """
    Fork a thread at a specific message.
    
    This creates a new thread with:
    - All messages up to and including the fork point
    - The summary from the source thread
    - Parent/child thread relationships
    
    Returns:
        dict: Success status and new thread details
    """
    try:
        success = mongo_store.fork_thread(
            user_id=fork_request.user_id,
            source_thread_id=fork_request.source_thread_id,
            new_thread_id=fork_request.new_thread_id,
            fork_at_message_id=fork_request.fork_at_message_id
        )
        
        if success:
            logger.info(
                f"Successfully forked thread {fork_request.source_thread_id} "
                f"to {fork_request.new_thread_id} at message {fork_request.fork_at_message_id}"
            )
            return {
                "status": "success",
                "message": "Thread forked successfully",
                "new_thread_id": fork_request.new_thread_id,
                "source_thread_id": fork_request.source_thread_id,
                "fork_point": fork_request.fork_at_message_id
            }
        else:
            logger.warning(
                f"Failed to fork thread {fork_request.source_thread_id}: "
                f"Thread or message not found"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source thread or fork message not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error forking thread: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fork thread: {str(e)}"
        )


@router.get("/summary/{user_id}/{thread_id}")
@limiter.limit("20/minute")
async def get_thread_summary(user_id: str, thread_id: str, request: Request):
    """
    Get the summary for a specific thread.
    
    Args:
        user_id: User identifier
        thread_id: Thread identifier
        
    Returns:
        dict: Thread summary
    """
    try:
        summary = mongo_store.get_thread_summary(user_id, thread_id)
        return {
            "user_id": user_id,
            "thread_id": thread_id,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error retrieving thread summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve summary: {str(e)}"
        )
