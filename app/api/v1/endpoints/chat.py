"""
Items CRUD endpoints.
"""
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from app.schemas.item import ChatMessage
from app.agent.main import getGraphResponse
from app.api.limiter import limiter
from fastapi import Request


from app.core.logger import logger


router = APIRouter()
try:
    graph = getGraphResponse()
    graph_init = True
    graph_error = None
except Exception as e:
    logger.error(f"Failed to initialize graph: {e}")
    graph = None
    graph_init = False
    graph_error = str(e)


@router.post("/")
@limiter.limit("5/minute")
async def get_response(chat_message: ChatMessage,
                       request: Request):
    if graph and graph_init:
        try:
            config = {
                "model": chat_message.model_name,
                "temperature": chat_message.temperature,
                "thread_id": chat_message.conversation_id,
            }
            logger.info(f"REQUEST {chat_message.conversation_id}: {config}")

            res = graph.get_response(
                query=chat_message.message,
                msg_id=chat_message.message_id,
                config=config,
                thread_id=chat_message.conversation_id,
                user_id=chat_message.user_id
            )
            if res:
                logger.info(f"RESPONSE {chat_message.conversation_id}: {res}")
                return res
            else:
                raise HTTPException(
                    status_code=500, detail="Faild To Generate Genarte Response From AI")
        except Exception as e:
            logger.error(f"Error : Get Response Endpoint : {e}")
            raise HTTPException(500, detail=str(e))
    else:
        logger.error(
            f"Error : Get Response Endpoint : Graph Not init : {graph_error}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unavailable: Graph failed to initialize - {graph_error}"
        )
