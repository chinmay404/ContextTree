"""
Chat endpoints.

POST /       — synchronous (returns full response JSON)
POST /stream — streaming (Server-Sent Events)

Identity comes from the verified service JWT (app/api/auth.py), never from
the request body. Rate limit: 60 messages/minute per verified user id.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from app.api.auth import get_current_user_id

from app.agent.helpers.get_llm import get_llm, validate_model_access
from app.agent.helpers.memory_model import get_summary_model_name
from app.agent.main import getGraphResponse
from app.agent.memory import extract_memory_facts, sanitize_summary_text
from app.agent.prompts.prompt_formation import get_formated_summury_prompt
from app.agent.utils.embeddings import get_embedding
from app.agent.reads import _top_k as card_top_k, resolve_reads
from app.agent.store.board_store import get_card_row, store as board_store
from app.api.errors import ApiError
from app.api.limiter import limiter, user_key
from app.core.config import settings
from app.core.logger import logger
from app.core.observability import build_langsmith_trace_config
from app.schemas.item import ChatMessage

router = APIRouter()

# ── Graph singleton ────────────────────────────────────────────────────────────
graph = None
graph_init = False
graph_error = None


def _build_chat_run_config(chat_message: ChatMessage, transport: str, user_id: str) -> dict:
    return {
        **build_langsmith_trace_config(
            run_name=f"contexttree_chat_{transport}",
            conversation_id=chat_message.conversation_id,
            user_id=user_id,
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
            "max_output_tokens": chat_message.max_output_tokens,
            "last_k_messages": chat_message.last_k_messages,
            "external_context_top_k": chat_message.external_context_top_k,
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


# ── Rate-limit key: the VERIFIED user id (set by get_current_user_id) ────────
_user_key = user_key  # canonical per-user key lives in app.api.limiter


# ── Daily quota (durable, Postgres-backed — survives restarts/workers) ────────

def _enforce_daily_quota(active_graph, user_id: str) -> None:
    try:
        count = active_graph.mongo_store.increment_daily_quota(user_id)
    except Exception as e:
        # Quota accounting must never take chat down; log and allow.
        logger.error(f"Quota accounting failed (allowing request): {e}")
        return
    if count > settings.DAILY_MESSAGE_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Daily message limit reached ({settings.DAILY_MESSAGE_LIMIT}/day). Resets at midnight UTC.",
        )


# ── Ownership gate (tenancy: the launch-blocker check) ────────────────────────

def _assert_thread_access(active_graph, user_id: str, thread_id: Optional[str]) -> None:
    """
    403 if the thread exists and belongs to someone else. A nonexistent
    thread is allowed — first message on a new node creates it, owned by
    the caller. Defense-in-depth: every store read is ALSO user-scoped,
    but this gate fails loudly instead of returning empty data.
    """
    if not thread_id:
        return
    owner = active_graph.mongo_store.get_thread_owner(thread_id)
    if owner is None:
        return
    resolved = active_graph.mongo_store.resolve_user_email(user_id)
    if owner != resolved:
        logger.warning(f"Cross-tenant access denied: user={user_id} thread={thread_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this conversation",
        )


# ── Fork initialisation (shared between sync and stream routes) ───────────────

WHOLE_CARRY_MAX_MESSAGES = 40
WHOLE_CARRY_MAX_TOKENS = 12_000
_WHOLE_CARRY_MAX_CHARS = WHOLE_CARRY_MAX_TOKENS * 4  # token estimate: ceil(chars / 4)


def _whole_carry_split(messages: list) -> tuple[list, list]:
    """(older, buffer): buffer is the longest tail within the message/char caps."""
    buffer_start = len(messages)
    total_chars = 0
    for m in reversed(messages):
        total_chars += len(m.get("text") or "")
        if (len(messages) - buffer_start) >= WHOLE_CARRY_MAX_MESSAGES or total_chars > _WHOLE_CARRY_MAX_CHARS:
            break
        buffer_start -= 1
    return messages[:buffer_start], messages[buffer_start:]


def _init_fork_if_needed(
    chat_message: ChatMessage, active_graph, transport: str, user_id: str, carry: str = "summary"
) -> None:
    """
    On the first message of a branched node, seed the new thread from the parent.

    carry modes (spec 5.4):
      summary (default) - compressed summary of the lineage + the last
                          FORK_BUFFER_MESSAGES verbatim; unknown values map here.
      whole             - the parent's messages up to the fork point copied as
                          inherited rows (last 40 / 48 000 chars); anything older is
                          summarised, or the parent's stored summary is kept verbatim.
      fresh             - nothing inherited; lineage columns and is_initialized only.

    For summary this preserves the Context Tree guarantee: the branch starts with an
    accurate, scoped summary of the parent lineage — not a blank slate.
    """
    if carry not in ("summary", "whole", "fresh"):
        carry = "summary"
    parent_thread_id = chat_message.parent_thread_id
    fork_at_message_id = chat_message.fork_at_message_id
    node_metadata = None

    if chat_message.conversation_id:
        node_metadata = active_graph.mongo_store.get_thread_fork_metadata(
            chat_message.conversation_id
        )
        if node_metadata:
            parent_thread_id = parent_thread_id or node_metadata.get("parent_node_id")
            fork_at_message_id = fork_at_message_id or node_metadata.get("forked_from_message_id")

            # Keep downstream tracing/config consistent even when the proxy did
            # not include fork metadata in the request body.
            chat_message.parent_thread_id = parent_thread_id
            chat_message.fork_at_message_id = fork_at_message_id

    # Parent is mandatory; a missing fork-point message id is NOT — the
    # inheritance payload falls back to latest-parent-state in that case
    # (racing canvas saves have nulled forked_from_message_id in prod).
    if not parent_thread_id:
        # No parent in the request body, the node row, or the edges table.
        # This node will be created as a ROOT — if it was meant to be a fork,
        # that lineage is lost. Log loudly: this path was invisible in prod.
        logger.warning(
            "Fork init skipped for %s: no parent resolvable (body=%s, node_row=%s)",
            chat_message.conversation_id,
            chat_message.parent_thread_id,
            bool(node_metadata),
        )
        return
    fork_at_message_id = fork_at_message_id or ""

    thread_exists = active_graph.mongo_store.thread_exists(chat_message.conversation_id)
    is_pending_fork = bool(
        node_metadata
        and parent_thread_id
        and (
            not bool(node_metadata.get("is_initialized"))
            or not list(node_metadata.get("ancestor_ids") or [])
        )
    )
    should_init = (not thread_exists) or is_pending_fork

    own_message_count = 0
    if thread_exists:
        try:
            own_message_count = active_graph.mongo_store.get_thread_message_count(
                chat_message.conversation_id
            )
        except Exception as e:
            logger.warning(f"Could not check message count for fork init: {e}")

    if carry == "summary" and thread_exists and not should_init and own_message_count <= 1:
        should_init = True

    # Self-heal: branches stamped initialized while their parent had nothing
    # to give (e.g. the parent's replies weren't persisted at fork time) have
    # no summary. Re-run inheritance so they finally learn their lineage.
    if carry == "summary" and thread_exists and not should_init:
        try:
            own_summary = sanitize_summary_text(
                active_graph.mongo_store.get_thread_summary(
                    user_id, chat_message.conversation_id
                )
            )
            if not own_summary:
                logger.info(
                    "Fork %s has no inherited summary — re-running inheritance",
                    chat_message.conversation_id,
                )
                should_init = True
        except Exception as e:
            logger.warning(f"Could not check summary for fork re-init: {e}")

    if not should_init:
        return

    if carry == "fresh":
        # Lineage columns + is_initialized only; nothing inherited, no LLM.
        ok = active_graph.mongo_store.fork_thread(
            user_id=user_id,
            source_thread_id=parent_thread_id,
            new_thread_id=chat_message.conversation_id,
            fork_at_message_id=fork_at_message_id,
            summary=None,
            summary_embedding=None,
            initial_messages=[],
            memory_facts={},
        )
        if ok:
            logger.info(
                "Fork initialised (fresh): %s → %s", parent_thread_id, chat_message.conversation_id
            )
        return

    fork_source = active_graph.mongo_store.get_fork_inheritance_payload(
        user_id=user_id,
        thread_id=parent_thread_id,
        message_id=fork_at_message_id,
        recent_limit=max(2, int(getattr(settings, "KEEP_LAST_MESSAGES", 6))),
    )
    existing_summary = sanitize_summary_text(fork_source.get("summary"))
    parent_memory_facts = fork_source.get("memory_facts") or {}
    messages_data = fork_source.get("messages") or []
    if messages_data and carry != "whole":
        logger.info(
            "Fork source scope resolved: parent=%s fork_at=%s mode=%s returned_messages=%s last_message=%s",
            parent_thread_id,
            fork_at_message_id,
            fork_source.get("mode"),
            len(messages_data),
            messages_data[-1]["message_id"],
        )
    if carry == "whole":
        if fork_source.get("mode") != "scoped-fork-point":
            messages_data = active_graph.mongo_store.get_thread_messages(user_id, parent_thread_id)
        logger.info(
            "Fork source (whole): parent=%s fork_at=%s mode=%s returned_messages=%s",
            parent_thread_id,
            fork_at_message_id,
            fork_source.get("mode"),
            len(messages_data),
        )
    elif not messages_data:
        fallback_k = getattr(settings, "KEEP_LAST_MESSAGES", 6)
        logger.warning("fork_at_message_id not found; falling back to last-K messages")
        messages_data = active_graph.mongo_store.get_thread_recent_messages(
            user_id,
            parent_thread_id,
            fallback_k,
        )

    # ── Split into "summarise" portion and verbatim buffer ────────────────────
    buffer_size = getattr(settings, "FORK_BUFFER_MESSAGES", 2)
    if carry == "whole":
        messages_to_summarize, buffer_messages = _whole_carry_split(messages_data)
    elif len(messages_data) <= buffer_size:
        messages_to_summarize = []
        buffer_messages = messages_data
    else:
        messages_to_summarize = messages_data[:-buffer_size]
        buffer_messages = messages_data[-buffer_size:]

    # Re-initialising a branch that already has its own conversation: buffer
    # messages can't be inserted (they'd get present-day timestamps and
    # scramble order), so the summary must carry EVERYTHING — summarize the
    # full parent scope, not just the pre-buffer portion, and do it even in
    # skip_resummarize fallback modes (the whole point of the re-init is that
    # this branch has no summary yet).
    force_summarize = own_message_count > 1 and bool(messages_data)
    if force_summarize:
        messages_to_summarize = messages_data

    new_summary = existing_summary
    new_memory_facts = parent_memory_facts if fork_source.get("skip_resummarize") else extract_memory_facts({}, messages_data)
    new_summary_embedding = []

    if messages_to_summarize and (force_summarize or carry == "whole" or not fork_source.get("skip_resummarize")):
        lc_messages = []
        for m in messages_to_summarize:
            if m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["text"]))
            elif m["role"] in ("assistant", "ai"):
                lc_messages.append(AIMessage(content=m["text"]))

        summary_model = get_summary_model_name(chat_message.model_name)
        llm = get_llm(summary_model, user_id=user_id)
        if llm:
            try:
                prompt = get_formated_summury_prompt(lc_messages, existing_summary)
                summary_response = llm.with_config(
                    build_langsmith_trace_config(
                        run_name="contexttree_fork_initial_summary",
                        conversation_id=chat_message.conversation_id,
                        user_id=user_id,
                        message_id=chat_message.message_id,
                        model_name=summary_model,
                        transport=transport,
                        node_name="fork-init",
                        parent_thread_id=chat_message.parent_thread_id,
                        fork_at_message_id=chat_message.fork_at_message_id,
                        tags=["summary"],
                    )
                ).invoke(prompt)
                new_summary = sanitize_summary_text(summary_response.content)
                new_summary_embedding = get_embedding(new_summary) or []
            except Exception as e:
                logger.error(f"Fork summarisation failed: {e}")
    logger.info(
        "Fork initialisation payload: new_thread=%s parent=%s mode=%s buffer_messages=%s summarized_messages=%s summary_chars=%s",
        chat_message.conversation_id,
        parent_thread_id,
        fork_source.get("mode"),
        len(buffer_messages),
        len(messages_to_summarize),
        len(new_summary or ""),
    )

    # If the parent had NOTHING to inherit (no messages, no summary — e.g. the
    # parent's own replies failed to persist at the time), do NOT run the fork
    # write: fork_thread would stamp is_initialized=true and this branch would
    # be permanently frozen as a blank slate. Leaving it uninitialized lets the
    # NEXT message retry inheritance once the parent actually has content.
    if not buffer_messages and not (new_summary or "").strip():
        logger.warning(
            "Fork inheritance EMPTY for %s (parent=%s): nothing to seed — "
            "leaving fork uninitialized so a later message can retry",
            chat_message.conversation_id,
            parent_thread_id,
        )
        return

    # When re-initializing a branch that already has its own conversation,
    # seed the summary/facts only — inserting buffer messages now would give
    # them present-day timestamps and scramble message order.
    seed_messages = buffer_messages if own_message_count <= 1 else []

    # Single atomic fork: node row, buffer messages, and memory facts are
    # written under a transaction-scoped advisory lock keyed on the new
    # thread_id. Concurrent first-message requests for the same fork serialize
    # at this point and the loser observes is_initialized=true and returns.
    active_graph.mongo_store.fork_thread(
        user_id=user_id,
        source_thread_id=parent_thread_id,
        new_thread_id=chat_message.conversation_id,
        fork_at_message_id=fork_at_message_id,
        summary=new_summary,
        summary_embedding=new_summary_embedding,
        initial_messages=seed_messages,
        memory_facts=new_memory_facts,
    )
    logger.info(
        f"Fork initialised: {parent_thread_id} → {chat_message.conversation_id} "
        f"at message {fork_at_message_id} "
        f"(buffer={len(buffer_messages)}, summary_chars={len(new_summary or '')})"
    )


# ── V3 card gate ──────────────────────────────────────────────────────────────

def _resolve_v3_card(user_id: str, chat_message: ChatMessage) -> Optional[dict]:
    """
    None for V2 turns; the live chat-card row for V3 turns (row["_user_email"] holds
    the resolved owner email so callers do not resolve it again).

    A turn is V3 when the body carries boardId or the node row has kind NOT NULL.
    Missing / deleted / foreign / other-board / non-V3-board -> 404; not a chat -> 400.
    Never creates a row.
    """
    node_id = chat_message.conversation_id
    board_id = chat_message.board_id
    with board_store.tx() as cur:
        row = get_card_row(cur, node_id)
        if not board_id and not (row and row.get("kind") is not None):
            return None
        email = board_store.resolve_user_email(user_id)
        if (not row or row.get("kind") is None or row.get("deleted_at") is not None
                or row.get("user_email") != email
                or (board_id and row.get("canvas_id") != board_id)):
            raise ApiError(404, "not_found")
        cur.execute(
            "SELECT 1 FROM canvases WHERE id = %s AND user_email = %s "
            "AND board_version = 3 AND deleted_at IS NULL",
            (row["canvas_id"], email),
        )
        if not cur.fetchone():
            raise ApiError(404, "not_found")
    if row["kind"] != "chat":
        raise ApiError(400, "invalid", "only chat cards can be chatted with")
    return {**row, "_user_email": email}


def _v3_gate(user_id: str, chat_message: ChatMessage) -> tuple[Optional[dict], str]:
    """(card, carry). For V3 the card row wins over the body for lineage."""
    card = _resolve_v3_card(user_id, chat_message)
    if card is None:
        return None, "summary"
    chat_message.parent_thread_id = card.get("parent_node_id")
    chat_message.fork_at_message_id = card.get("forked_from_message_id")
    carry = (card.get("data") or {}).get("carry")
    return card, carry if isinstance(carry, str) else "summary"


def _graph_kwargs(chat_message: ChatMessage, card: Optional[dict], carry: str) -> dict:
    """Per-turn context kwargs: V2 passes the body through; V3 uses the server read set."""
    if card is None:
        return dict(
            context_node_ids=chat_message.context_node_ids,
            board_reads=None,
            last_k_messages=chat_message.last_k_messages,
            external_context_top_k=chat_message.external_context_top_k,
        )
    data = card.get("data") or {}
    adv = data.get("advanced") if isinstance(data.get("advanced"), dict) else {}
    budget = adv.get("contextBudgetTokens")
    last_k = chat_message.last_k_messages
    if carry == "whole" and last_k is None:
        last_k = 50
    return dict(
        context_node_ids=None,
        board_reads=resolve_reads(board_store, card["_user_email"], card["canvas_id"], card["id"]),
        budget_tokens=budget if isinstance(budget, int) and not isinstance(budget, bool) else None,
        last_k_messages=last_k,
        external_context_top_k=card_top_k(data),
        user_email=card["_user_email"],
    )


# ── Sync endpoint ──────────────────────────────────────────────────────────────

@router.post("/")
@limiter.limit("60/minute", key_func=_user_key)
async def get_response(
    chat_message: ChatMessage,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    active_graph = _require_graph()

    if not chat_message.conversation_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="nodeId (conversation_id) is required",
        )

    try:
        card, carry = _v3_gate(user_id, chat_message)
        if card is None:
            _assert_thread_access(active_graph, user_id, chat_message.conversation_id)
            _assert_thread_access(active_graph, user_id, chat_message.parent_thread_id)
        _enforce_daily_quota(active_graph, user_id)

        access_error = validate_model_access(chat_message.model_name, user_id)
        if access_error:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=access_error)

        _init_fork_if_needed(chat_message, active_graph, "http", user_id, carry=carry)
        graph_kw = _graph_kwargs(chat_message, card, carry)

        config = _build_chat_run_config(chat_message, "http", user_id)
        logger.info(f"Chat request: node={chat_message.conversation_id} model={chat_message.model_name}")

        res = active_graph.get_response(
            query=chat_message.message,
            msg_id=chat_message.message_id,
            config=config,
            thread_id=chat_message.conversation_id,
            user_id=user_id,
            system_prompt=chat_message.system_prompt,
            regenerate_last_user=chat_message.regenerate_last_user,
            web_search=chat_message.web_search,
            **graph_kw,
        )
        if not res:
            raise HTTPException(status_code=500, detail="Failed to generate AI response")

        summary = active_graph.mongo_store.get_thread_summary(
            user_id, chat_message.conversation_id
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
async def stream_response(
    chat_message: ChatMessage,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    active_graph = _require_graph()

    if not chat_message.conversation_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="nodeId (conversation_id) is required",
        )

    try:
        card, carry = _v3_gate(user_id, chat_message)
        if card is None:
            _assert_thread_access(active_graph, user_id, chat_message.conversation_id)
            _assert_thread_access(active_graph, user_id, chat_message.parent_thread_id)
        _enforce_daily_quota(active_graph, user_id)

        access_error = validate_model_access(chat_message.model_name, user_id)
        if access_error:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=access_error)

        _init_fork_if_needed(chat_message, active_graph, "sse", user_id, carry=carry)
        graph_kw = _graph_kwargs(chat_message, card, carry)

        config = _build_chat_run_config(chat_message, "sse", user_id)
        logger.info(f"Stream request: node={chat_message.conversation_id} model={chat_message.model_name}")

        return StreamingResponse(
            active_graph.get_stream_response(
                query=chat_message.message,
                msg_id=chat_message.message_id,
                config=config,
                thread_id=chat_message.conversation_id,
                user_id=user_id,
                system_prompt=chat_message.system_prompt,
                regenerate_last_user=chat_message.regenerate_last_user,
                web_search=chat_message.web_search,
                **graph_kw,
            ),
            media_type="text/event-stream",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stream endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
