"""
Chat endpoints.

POST /       — synchronous (returns full response JSON)
POST /stream — streaming (Server-Sent Events)

Rate limit: 60 messages/minute per authenticated user_id.
If user_id is not present in the body (shouldn't happen in normal use),
falls back to remote IP to avoid crashing.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.helpers.get_llm import get_llm, validate_model_access
from app.agent.main import getGraphResponse
from app.agent.prompts.prompt_formation import get_formated_summury_prompt
from app.agent.utils.embeddings import get_embedding
from app.api.limiter import limiter
from app.core.config import settings
from app.core.logger import logger
from app.core.observability import build_langsmith_trace_config
from app.schemas.item import ChatMessage

router = APIRouter()

# ── Graph singleton ────────────────────────────────────────────────────────────
graph = None
graph_init = False
graph_error = None


def _build_chat_run_config(chat_message: ChatMessage, transport: str) -> dict:
    return {
        **build_langsmith_trace_config(
            run_name=f"contexttree_chat_{transport}",
            conversation_id=chat_message.conversation_id,
            user_id=chat_message.user_id,
            message_id=chat_message.message_id,
            model_name=chat_message.model_name,
            transport=transport,
            parent_thread_id=chat_message.parent_thread_id,
            fork_at_message_id=chat_message.fork_at_message_id,
            tags=["chat"],
        ),
        "configurable": {
            "model": chat_message.model_name,
            "temperature": chat_message.temperature,
            "thread_id": chat_message.conversation_id,
        },
    }


def _initialise_graph(force: bool = False) -> bool:
    """
    Build the LangGraph singleton.

    The backend used to try this only once at import time. If Supabase or the
    checkpointer was briefly unavailable during startup, the process stayed in a
    broken 503 state until it was restarted. We keep the singleton model, but
    allow a safe retry on demand.
    """
    global graph, graph_init, graph_error

    if graph_init and graph is not None and not force:
        return True

    try:
        graph = getGraphResponse()
        graph_init = True
        graph_error = None
        logger.info("Graph initialised successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialise graph: {e}")
        graph = None
        graph_init = False
        graph_error = str(e)
        return False


def _require_graph():
    if _initialise_graph(force=not graph_init) and graph is not None:
        return graph

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"Graph failed to initialise: {graph_error}",
    )


_initialise_graph()


# ── Rate-limit key: prefer user_id from body, fall back to IP ─────────────────
def _user_key(request: Request) -> str:
    """
    SlowAPI key function.  We want per-user limits, not per-IP, so that users
    behind the same NAT/proxy don't block each other.
    The user_id is injected into the request body by the Next.js proxy before
    forwarding, so we can read it here.  Body parsing is done lazily via state
    set by the dependency injection route function.
    """
    uid = getattr(request.state, "user_id", None)
    if uid:
        return uid
    # Fallback — shouldn't happen in production
    return request.client.host if request.client else "unknown"


# ── Fork initialisation (shared between sync and stream routes) ───────────────

def _init_fork_if_needed(chat_message: ChatMessage, active_graph, transport: str) -> None:
    """
    On the first message of a branched node, seed the new thread with a
    compressed summary + a small verbatim buffer from the parent.

    This preserves the Context Tree guarantee: the branch starts with an
    accurate, scoped summary of the parent lineage — not a blank slate.
    """
    if not (chat_message.parent_thread_id and chat_message.fork_at_message_id):
        return

    thread_exists = active_graph.mongo_store.thread_exists(chat_message.conversation_id)
    should_init = not thread_exists

    if thread_exists and not should_init:
        try:
            count = active_graph.mongo_store.get_thread_message_count(chat_message.conversation_id)
            if count <= 1:
                should_init = True
        except Exception as e:
            logger.warning(f"Could not check message count for fork init: {e}")

    if not should_init:
        return

    # ── Gather parent history up to the fork point ────────────────────────────
    existing_summary, messages_data = active_graph.mongo_store.get_messages_until(
        chat_message.user_id,
        chat_message.parent_thread_id,
        chat_message.fork_at_message_id,
    )
    if messages_data:
        logger.info(
            "Fork source scope resolved: parent=%s fork_at=%s returned_messages=%s last_message=%s",
            chat_message.parent_thread_id,
            chat_message.fork_at_message_id,
            len(messages_data),
            messages_data[-1]["message_id"],
        )
    if not messages_data:
        fallback_k = getattr(settings, "KEEP_LAST_MESSAGES", 6)
        logger.warning("fork_at_message_id not found; falling back to last-K messages")
        messages_data = active_graph.mongo_store.get_thread_recent_messages(
            chat_message.user_id,
            chat_message.parent_thread_id,
            fallback_k,
        )

    # ── Split into "summarise" portion and verbatim buffer ────────────────────
    buffer_size = getattr(settings, "FORK_BUFFER_MESSAGES", 2)
    if len(messages_data) <= buffer_size:
        messages_to_summarize = []
        buffer_messages = messages_data
    else:
        messages_to_summarize = messages_data[:-buffer_size]
        buffer_messages = messages_data[-buffer_size:]

    new_summary = existing_summary
    new_summary_embedding = []

    if messages_to_summarize:
        lc_messages = []
        for m in messages_to_summarize:
            if m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["text"]))
            elif m["role"] in ("assistant", "ai"):
                lc_messages.append(AIMessage(content=m["text"]))

        llm = get_llm(chat_message.model_name, user_id=chat_message.user_id)
        if llm:
            try:
                prompt = get_formated_summury_prompt(lc_messages, existing_summary)
                summary_response = llm.with_config(
                    build_langsmith_trace_config(
                        run_name="contexttree_fork_initial_summary",
                        conversation_id=chat_message.conversation_id,
                        user_id=chat_message.user_id,
                        message_id=chat_message.message_id,
                        model_name=chat_message.model_name,
                        transport=transport,
                        node_name="fork-init",
                        parent_thread_id=chat_message.parent_thread_id,
                        fork_at_message_id=chat_message.fork_at_message_id,
                        tags=["summary"],
                    )
                ).invoke(prompt)
                new_summary = summary_response.content
                new_summary_embedding = get_embedding(new_summary) or []
            except Exception as e:
                logger.error(f"Fork summarisation failed: {e}")
    logger.info(
        "Fork initialisation payload: new_thread=%s parent=%s buffer_messages=%s summarized_messages=%s summary_chars=%s",
        chat_message.conversation_id,
        chat_message.parent_thread_id,
        len(buffer_messages),
        len(messages_to_summarize),
        len(new_summary or ""),
    )

    active_graph.mongo_store.fork_thread(
        user_id=chat_message.user_id,
        source_thread_id=chat_message.parent_thread_id,
        new_thread_id=chat_message.conversation_id,
        fork_at_message_id=chat_message.fork_at_message_id,
        summary=new_summary,
        summary_embedding=new_summary_embedding,
        initial_messages=buffer_messages,
    )
    logger.info(
        f"Fork initialised: {chat_message.parent_thread_id} → {chat_message.conversation_id} "
        f"at message {chat_message.fork_at_message_id}"
    )


# ── Sync endpoint ──────────────────────────────────────────────────────────────

@router.post("/")
@limiter.limit("60/minute", key_func=_user_key)
async def get_response(chat_message: ChatMessage, request: Request):
    active_graph = _require_graph()

    if not chat_message.conversation_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="nodeId (conversation_id) is required",
        )

    # Store user_id on request.state so _user_key can read it
    request.state.user_id = chat_message.user_id or request.client.host

    try:
        access_error = validate_model_access(chat_message.model_name, chat_message.user_id)
        if access_error:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=access_error)

        _init_fork_if_needed(chat_message, active_graph, "http")

        config = _build_chat_run_config(chat_message, "http")
        logger.info(f"Chat request: node={chat_message.conversation_id} model={chat_message.model_name}")

        res = active_graph.get_response(
            query=chat_message.message,
            msg_id=chat_message.message_id,
            config=config,
            thread_id=chat_message.conversation_id,
            user_id=chat_message.user_id,
        )
        if not res:
            raise HTTPException(status_code=500, detail="Failed to generate AI response")

        summary = active_graph.mongo_store.get_thread_summary(
            chat_message.user_id, chat_message.conversation_id
        )
        return {"message": res, "summary": summary or ""}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Stream endpoint ────────────────────────────────────────────────────────────

@router.post("/stream")
@limiter.limit("60/minute", key_func=_user_key)
async def stream_response(chat_message: ChatMessage, request: Request):
    active_graph = _require_graph()

    if not chat_message.conversation_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="nodeId (conversation_id) is required",
        )

    request.state.user_id = chat_message.user_id or request.client.host

    try:
        access_error = validate_model_access(chat_message.model_name, chat_message.user_id)
        if access_error:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=access_error)

        _init_fork_if_needed(chat_message, active_graph, "sse")

        config = _build_chat_run_config(chat_message, "sse")
        logger.info(f"Stream request: node={chat_message.conversation_id} model={chat_message.model_name}")

        return StreamingResponse(
            active_graph.get_stream_response(
                query=chat_message.message,
                msg_id=chat_message.message_id,
                config=config,
                thread_id=chat_message.conversation_id,
                user_id=chat_message.user_id,
            ),
            media_type="text/event-stream",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stream endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
