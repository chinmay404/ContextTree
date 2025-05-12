"""
Items CRUD endpoints.
"""
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from app.schemas.item import ChatMessage
from app.Agent.main import getGraphResponse


router = APIRouter()
graph = getGraphResponse()

@router.post("/")
async def get_response(chat_message: ChatMessage):
    config = {
        "model": chat_message.model_name,
        "temperature": chat_message.temperature,
        "thread_id": chat_message.conversation_id,
    }

    res = graph.get_response(
        query=chat_message.message,
        config=config,
        thread_id=chat_message.conversation_id,
        user_id=chat_message.user_id
    )
    return res