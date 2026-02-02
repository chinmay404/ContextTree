"""
Items CRUD endpoints.
"""
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.schemas.item import ChatMessage
from app.agent.main import getGraphResponse
from app.api.limiter import limiter
from fastapi import Request


from app.core.logger import logger
from app.core.config import settings
from app.agent.helpers.get_llm import get_groq_llm
from app.agent.prompts.prompt_formation import get_formated_summury_prompt
from langchain_core.messages import HumanMessage, AIMessage
from app.agent.utils.embeddings import get_embedding


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
            # Initialize forked thread if needed (lazy fork on first message)
            # Check if fork init is needed
            thread_exists = graph.mongo_store.thread_exists(chat_message.conversation_id)
            should_init_fork = False
            
            if not thread_exists:
                should_init_fork = True
            elif (
                chat_message.parent_thread_id
                and chat_message.fork_at_message_id
            ):
                # If thread exists but has no messages, it might be an empty shell created by optimistic UI
                try:
                    msg_count = graph.mongo_store.get_thread_message_count(chat_message.conversation_id)
                    # Allow init if empty OR has just the trigger message (count=1)
                    if msg_count <= 1:
                        should_init_fork = True
                except Exception as e:
                    logger.warning(f"Failed to check message count for fork init: {e}")

            if (
                chat_message.parent_thread_id
                and chat_message.fork_at_message_id
                and chat_message.conversation_id
                and should_init_fork
            ):
                existing_summary, messages_data = graph.mongo_store.get_messages_until(
                    chat_message.user_id,
                    chat_message.parent_thread_id,
                    chat_message.fork_at_message_id,
                )
                if not messages_data:
                    logger.warning(
                        "Fork init fallback: fork_at_message_id not found; using parent last-K messages."
                    )
                    fallback_k = getattr(settings, "KEEP_LAST_MESSAGES", 6)
                    messages_data = graph.mongo_store.get_thread_recent_messages(
                        chat_message.user_id,
                        chat_message.parent_thread_id,
                        fallback_k,
                    )

                new_summary = existing_summary
                new_summary_embedding = []
                buffer_size = getattr(settings, "FORK_BUFFER_MESSAGES", 2)

                if len(messages_data) <= buffer_size:
                    messages_to_summarize = []
                    buffer_messages = messages_data
                else:
                    messages_to_summarize = messages_data[:-buffer_size]
                    buffer_messages = messages_data[-buffer_size:]

                if messages_to_summarize:
                    lc_messages = []
                    for m in messages_to_summarize:
                        if m['role'] == 'user':
                            lc_messages.append(HumanMessage(content=m['text']))
                        elif m['role'] in ['assistant', 'ai']:
                            lc_messages.append(AIMessage(content=m['text']))

                    llm = get_groq_llm(name=chat_message.model_name)
                    if llm:
                        prompt = get_formated_summury_prompt(lc_messages, existing_summary)
                        summary_response = llm.invoke(prompt)
                        new_summary = summary_response.content
                        new_summary_embedding = get_embedding(new_summary) or []

                graph.mongo_store.fork_thread(
                    user_id=chat_message.user_id,
                    source_thread_id=chat_message.parent_thread_id,
                    new_thread_id=chat_message.conversation_id,
                    fork_at_message_id=chat_message.fork_at_message_id,
                    summary=new_summary,
                    summary_embedding=new_summary_embedding,
                    initial_messages=buffer_messages,
                )

            config = {"configurable": {
                "model": chat_message.model_name,
                "temperature": chat_message.temperature,
                "thread_id": chat_message.conversation_id,
            }}
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


@router.post("/stream")
@limiter.limit("5/minute")
async def stream_response(chat_message: ChatMessage, request: Request):
    if graph and graph_init:
        try:
            # Initialize forked thread if needed (lazy fork on first message)
            # Check if fork init is needed
            thread_exists = graph.mongo_store.thread_exists(chat_message.conversation_id)
            should_init_fork = False
            
            if not thread_exists:
                should_init_fork = True
            elif (
                chat_message.parent_thread_id
                and chat_message.fork_at_message_id
            ):
                # If thread exists but has no messages, it might be an empty shell created by optimistic UI
                try:
                    msg_count = graph.mongo_store.get_thread_message_count(chat_message.conversation_id)
                    # Allow init if empty OR has just the trigger message (count=1)
                    if msg_count <= 1:
                        should_init_fork = True
                except Exception as e:
                    logger.warning(f"Failed to check message count for fork init: {e}")

            if (
                chat_message.parent_thread_id
                and chat_message.fork_at_message_id
                and chat_message.conversation_id
                and should_init_fork
            ):
                existing_summary, messages_data = graph.mongo_store.get_messages_until(
                    chat_message.user_id,
                    chat_message.parent_thread_id,
                    chat_message.fork_at_message_id,
                )
                if not messages_data:
                    logger.warning(
                        "Fork init fallback: fork_at_message_id not found; using parent last-K messages."
                    )
                    fallback_k = getattr(settings, "KEEP_LAST_MESSAGES", 6)
                    messages_data = graph.mongo_store.get_thread_recent_messages(
                        chat_message.user_id,
                        chat_message.parent_thread_id,
                        fallback_k,
                    )

                new_summary = existing_summary
                new_summary_embedding = []
                buffer_size = getattr(settings, "FORK_BUFFER_MESSAGES", 2)

                if len(messages_data) <= buffer_size:
                    messages_to_summarize = []
                    buffer_messages = messages_data
                else:
                    messages_to_summarize = messages_data[:-buffer_size]
                    buffer_messages = messages_data[-buffer_size:]

                if messages_to_summarize:
                    lc_messages = []
                    for m in messages_to_summarize:
                        if m['role'] == 'user':
                            lc_messages.append(HumanMessage(content=m['text']))
                        elif m['role'] in ['assistant', 'ai']:
                            lc_messages.append(AIMessage(content=m['text']))

                    llm = get_groq_llm(name=chat_message.model_name)
                    if llm:
                        prompt = get_formated_summury_prompt(lc_messages, existing_summary)
                        summary_response = llm.invoke(prompt)
                        new_summary = summary_response.content
                        new_summary_embedding = get_embedding(new_summary) or []

                graph.mongo_store.fork_thread(
                    user_id=chat_message.user_id,
                    source_thread_id=chat_message.parent_thread_id,
                    new_thread_id=chat_message.conversation_id,
                    fork_at_message_id=chat_message.fork_at_message_id,
                    summary=new_summary,
                    summary_embedding=new_summary_embedding,
                    initial_messages=buffer_messages,
                )

            config = {"configurable": {
                "model": chat_message.model_name,
                "temperature": chat_message.temperature,
                "thread_id": chat_message.conversation_id,
            }}
            logger.info(f"STREAM REQUEST {chat_message.conversation_id}: {config}")

            return StreamingResponse(
                graph.get_stream_response(
                    query=chat_message.message,
                    msg_id=chat_message.message_id,
                    config=config,
                    thread_id=chat_message.conversation_id,
                    user_id=chat_message.user_id
                ),
                media_type="text/event-stream"
            )
        except Exception as e:
            logger.error(f"Error : Stream Response Endpoint : {e}")
            raise HTTPException(500, detail=str(e))
    else:
        logger.error(f"Error : Stream Response Endpoint : Graph Not init : {graph_error}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unavailable: Graph failed to initialize - {graph_error}"
        )
