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
from app.agent.helpers.get_llm import get_groq_llm
from app.agent.prompts.prompt_formation import get_formated_summury_prompt
from langchain_core.messages import HumanMessage, AIMessage
from app.agent.utils.embeddings import get_embedding


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
    - The summary from the source thread updated with the tail messages
    - Parent/child thread relationships
    
    Returns:
        dict: Success status and new thread details
    """
    try:
        # 1. Get messages and existing summary up to the fork point
        existing_summary, messages_data = mongo_store.get_messages_until(
            fork_request.user_id,
            fork_request.source_thread_id,
            fork_request.fork_at_message_id
        )

        new_summary = existing_summary
        new_summary_embedding = []

        # 2. If we have messages, we should update the summary to include them
        # "modify new summury of node ... = previous node summury + current convo sumurry"
        if messages_data:
            lc_messages = []
            for m in messages_data:
                if m['role'] == 'user':
                    lc_messages.append(HumanMessage(content=m['text']))
                elif m['role'] in ['assistant', 'ai']:
                    lc_messages.append(AIMessage(content=m['text']))
            
            # Generate new summary using LLM
            # existing_summary serves as the "summary" input, lc_messages as "conversation"
            try:
                llm = get_groq_llm()
                prompt = get_formated_summury_prompt(lc_messages, existing_summary)
                summary_response = llm.invoke(prompt)
                new_summary = summary_response.content
                
                # Generate embedding for the new summary
                new_summary_embedding = get_embedding(new_summary) or []
            except Exception as e:
                logger.error(f"Failed to generate summary during fork: {e}")
                # Fallback to existing summary if LLM fails
                new_summary = existing_summary

        success = mongo_store.fork_thread(
            user_id=fork_request.user_id,
            source_thread_id=fork_request.source_thread_id,
            new_thread_id=fork_request.new_thread_id,
            fork_at_message_id=fork_request.fork_at_message_id,
            summary=new_summary,
            summary_embedding=new_summary_embedding
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
