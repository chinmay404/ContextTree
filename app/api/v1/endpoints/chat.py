"""
Items CRUD endpoints.
"""
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from app.schemas.item import ChatMessage

router = APIRouter()

@router.post("/")
async def get_response(chat_message: ChatMessage):
    """
    Get AI response for a chat message
    """
    return {
        "conversation_id": chat_message.conversation_id or str(uuid4()),
        "user_message": chat_message.message,
        "ai_response": f"Response to: {chat_message.message}"
    }