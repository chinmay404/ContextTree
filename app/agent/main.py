
from langgraph.graph import StateGraph, START, END
from datetime import datetime
from typing import Optional
from app.agent.nodes.assistant_node import AgentNodes
from app.agent.state import State
from app.agent.prompts.prompt_formation import get_formated_prompt
from langchain_core.messages import AIMessage, HumanMessage
import json
from app.agent.memory import extract_memory_facts
from app.agent.utils.saver import postgres_saver, async_postgres_saver
from app.agent.store.PostgresStore import PostgresConversationStore
from app.agent.store.board_store import store as board_store
from app.agent.board_context import (
    build_board_context, has_meaningful_embedding, number_passages, trim_chunks_to_budget,
)
from app.agent.citations import CITE_INSTRUCTION, extract_citations
from app.agent.reads import Reads, _top_k as _default_top_k
from app.agent.utils.embeddings import get_embedding
from app.agent.utils.text import clip_text
from app.agent.helpers.web_search import search_web, format_web_snippets
from app.agent.helpers.web_fetch import extract_urls, fetch_pages, format_page_snippets
from app.agent.helpers.v3_llm import KeyRejectedError, key_rejected_frame
from app.core.config import settings
from app.core.logger import logger

from fastapi import HTTPException


SIMILAR_CONTEXT_LIMIT = 3
FILE_CONTEXT_LIMIT = 3
WEB_CONTEXT_LIMIT = 3
MAX_WEB_SNIPPET_CHARS = 600
MAX_SUMMARY_CHARS = 2400
MAX_RECENT_MESSAGE_CHARS = 1400
MAX_SIMILAR_SNIPPET_CHARS = 320
MAX_FILE_SNIPPET_CHARS = 900
MAX_FILE_METADATA_CHARS = 220
SIMILARITY_MIN_SCORE = max(0.0, float(getattr(settings, "SIMILARITY_MIN_SCORE", 0.2)))


def _split_reasoning(chunk) -> tuple[str, str]:
    """Separate a model chunk into (reasoning, answer) text.

    Providers deliver "thinking" three ways: inline <think> tags in content
    (already handled by the UI), a separate additional_kwargs field
    (NIM/Groq/OpenAI: reasoning_content or reasoning), or thought-flagged
    parts in a content list (Gemini). The stream re-frames the out-of-band
    kinds as inline <think> tags so the UI's live Thinking block works for
    every provider.
    """
    ak = getattr(chunk, "additional_kwargs", None) or {}
    reasoning = str(ak.get("reasoning_content") or ak.get("reasoning") or "")

    content = getattr(chunk, "content", None)
    if isinstance(content, list):
        answer_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                answer_parts.append(part)
            elif isinstance(part, dict):
                text = str(part.get("text") or part.get("thinking") or "")
                if part.get("thought") or part.get("type") in ("thinking", "reasoning"):
                    reasoning += text
                else:
                    answer_parts.append(text)
        return reasoning, "".join(answer_parts)
    return reasoning, _content_to_text(content)


def _content_to_text(content) -> str:
    """LangChain message content is a str OR a list of parts (Gemini sends
    [{'type':'text','text':...}, ...]). Normalize to plain text — string
    concatenation with a list part crashed async streaming to sync fallback."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict):
                out.append(str(part.get("text") or ""))
        return "".join(out)
    return str(content or "")


_clip_text = clip_text  # alias kept for existing call sites


def _normalized_message_id(message_id: str | None) -> str:
    raw = str(message_id or "").strip()
    for suffix in ("_assistant", "-assistant", "_ai", "_a", "-a", "_user", "-user", "_u", "-u"):
        if raw.endswith(suffix):
            return raw[: -len(suffix)]
    return raw


def _is_duplicate_hydrated_user_message(last_msg, msg_id: str | None) -> bool:
    """
    Only de-dupe exact replays of the same user message.

    Older logic also matched by content / prefix, which could incorrectly treat
    a legitimate new prompt as a retry. That was especially confusing around
    branches where the hydrated buffer already contains recent parent history.
    """
    if not isinstance(last_msg, HumanMessage):
        return False

    last_id = _normalized_message_id(getattr(last_msg, "id", None))
    incoming_id = _normalized_message_id(msg_id)
    return bool(last_id and incoming_id and last_id == incoming_id)


_has_meaningful_embedding = has_meaningful_embedding


def _history_chars(messages) -> int:
    return sum(len(_content_to_text(getattr(m, "content", ""))) for m in messages or [])


def _budget_history(current_state, initial_messages) -> list:
    """Messages that will reach the model: the hydrated buffer, or (when hydration was
    skipped) whatever the checkpointer already holds for the thread."""
    if initial_messages:
        return initial_messages
    values = getattr(current_state, "values", None) or {}
    return list(values.get("messages") or [])


def _build_memory_saver():
    """Best-effort in-memory fallback for local/dev when PostgresSaver is unavailable."""
    try:
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()
    except Exception as memory_err:
        try:
            from langgraph.checkpoint.memory import InMemorySaver
            return InMemorySaver()
        except Exception as in_memory_err:
            logger.error(
                "Failed to initialize in-memory LangGraph saver: %s / %s",
                memory_err,
                in_memory_err,
            )
            return None


class getGraphResponse():
    def __init__(self):
        self.memory = postgres_saver()
        if self.memory is None:
            # Boot can race the deploy-overlap window when the outgoing
            # instance still holds session-pooler slots (EMAXCONNSESSION).
            # Retry before accepting the in-memory fallback — that fallback
            # silently loses conversation state and never recovers.
            import time as _time

            for attempt in range(1, 4):
                _time.sleep(5)
                self.memory = postgres_saver()
                if self.memory is not None:
                    logger.info("PostgresSaver recovered on retry %d", attempt)
                    break
        if self.memory is None:
            logger.warning(
                "PostgresSaver unavailable; falling back to in-memory checkpointing."
            )
            self.memory = _build_memory_saver()
        if self.memory is None:
            raise MemoryError(
                "Failed to initialize any LangGraph checkpointer")
        self.mongo_store = PostgresConversationStore()
        Nodes = AgentNodes(mongo_store=self.mongo_store)
        self.nodes = Nodes.get_nodes()
        self.graph = self.build_graph()
        if self.graph is None:
            raise RuntimeError(
                "Failed to initialize graph: build_graph returned None")
        # Build async graph for streaming support
        self.async_graph = self._build_async_graph()

    def _make_builder(self):
        builder = StateGraph(State)
        builder.add_node("assistant", self.nodes["assistant"])
        builder.add_node("summurize", self.nodes["summurize"])
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant", self.nodes["summury_decision"],
            {True: "summurize", False: END}
        )
        builder.add_edge("summurize", END)
        return builder

    def build_graph(self):
        try:
            builder = self._make_builder()
            graph = builder.compile(checkpointer=self.memory)
            return graph
        except Exception as e:
            logger.error(f"getGraphResponse - graph builder - {e}")
            return None

    def _build_async_graph(self):
        """Build a separate graph with AsyncPostgresSaver for streaming."""
        try:
            async_memory = async_postgres_saver()
            if async_memory is None:
                logger.warning("AsyncPostgresSaver not available, streaming will use sync fallback")
                return None
            builder = self._make_builder()
            return builder.compile(checkpointer=async_memory)
        except Exception as e:
            logger.warning(f"Failed to build async graph: {e}")
            return None

    def get_response(
        self,
        query: str,
        config: dict,
        user_id: str,
        thread_id: str,
        msg_id: str,
        context_node_ids: Optional[list] = None,
        system_prompt: Optional[str] = None,
        last_k_messages: Optional[int] = None,
        external_context_top_k: Optional[int] = None,
        regenerate_last_user: bool = False,
        web_search: bool = False,
        board_reads: Optional[Reads] = None,
        budget_tokens: Optional[int] = None,
        user_email: Optional[str] = None,
        byok_only: bool = False,
    ):
        try:
            # Check for existing state in LangGraph
            current_state = self.graph.get_state(config)
            initial_messages = []

            summary_state = self.mongo_store.get_thread_summary_state(user_id, thread_id)
            existing_summary = summary_state["summary"]
            watermark = summary_state["summarized_up_to_position"]

            # If checkpointer has no live state (new thread or forked thread),
            # rebuild a buffer from the messages table — but only the tail past
            # the summary watermark. Already-summarized turns stay in the DB
            # for the UI; we don't replay them through the LLM.
            if not current_state.values or not current_state.values.get("messages"):
                logger.info(
                    "Hydrating recent state from DB for thread %s (watermark=%s)",
                    thread_id,
                    watermark,
                )
                initial_messages = self._hydrate_recent_messages(
                    user_id, thread_id, watermark, recent_limit_override=last_k_messages
                )

            existing_memory_facts = self.mongo_store.get_thread_memory_facts(user_id, thread_id)
            prompt_summary = _clip_text(existing_summary, MAX_SUMMARY_CHARS)
            passages: Optional[dict] = {} if board_reads is not None else None
            context_snippets, external_context_str, _web_hits, linked_answers = self._build_retrieved_context(
                query, user_id, thread_id, context_node_ids=context_node_ids,
                external_context_top_k=external_context_top_k,
                web_search=web_search,
                board_reads=board_reads,
                budget_tokens=budget_tokens,
                user_email=user_email,
                fixed_chars=len(query) + len(prompt_summary or "") + _history_chars(_budget_history(current_state, initial_messages)),
                passages=passages,
            )

            prompt = get_formated_prompt(query, user_id)
            timestamp = datetime.utcnow().isoformat()

            if regenerate_last_user:
                # Forking from a user message: that message already lives at the
                # tail of `initial_messages` (copied into the new branch by the
                # fork-init buffer). Don't append a second copy — the model is
                # being asked to answer the *existing* tail message.
                if not initial_messages or not isinstance(initial_messages[-1], HumanMessage):
                    logger.warning(
                        "regenerate_last_user requested but buffer tail is not a user message; "
                        "falling back to normal append for thread %s",
                        thread_id,
                    )
                    user_msg = HumanMessage(
                        content=prompt,
                        id=msg_id,
                        metadata={"user_id": user_id, "thread_id": thread_id, "timestamp": timestamp},
                    )
                    messages_input = initial_messages + [user_msg]
                    is_duplicate = False
                else:
                    logger.info(
                        "regenerate_last_user: replaying buffer tail as the prompt for thread %s",
                        thread_id,
                    )
                    messages_input = initial_messages
                    # Treat as duplicate so the post-LLM code skips the user-msg
                    # DB insert (it's already there from fork-init).
                    is_duplicate = True
            else:
                user_msg = HumanMessage(
                    content=prompt,
                    id=msg_id,
                    metadata={"user_id": user_id,
                              "thread_id": thread_id,
                              "timestamp": timestamp}
                )

                # Combine hydrated history with new message
                messages_input = initial_messages + [user_msg]

                # Deduplicate logic: If we hydrated from DB and the last message matches the current user message, don't append it again.
                is_duplicate = False
                if initial_messages:
                    last_msg = initial_messages[-1]

                    if _is_duplicate_hydrated_user_message(last_msg, msg_id):
                        is_duplicate = True

                    if is_duplicate:
                        logger.info(f"Duplicate message detected in hydration: {msg_id}. Using DB version.")
                        messages_input = initial_messages

            try:
                result = self.graph.invoke(
                    {
                        "user_id": user_id,
                        "thread_id": thread_id,
                        "messages": messages_input,
                        "summary": prompt_summary,
                        "summarized_up_to_position": watermark,
                        "memory_facts": existing_memory_facts,
                        "context": context_snippets,
                        "external_context": external_context_str,
                        "linked_answers": linked_answers,
                        "citation_note": CITE_INSTRUCTION if passages else None,
                        "context_node_ids": context_node_ids,
                        "model": config.get("configurable", {}).get("model"),
                        "temperature": config.get("configurable", {}).get("temperature"),
                        "max_output_tokens": config.get("configurable", {}).get("max_output_tokens"),
                        "last_k_messages": config.get("configurable", {}).get("last_k_messages"),
                        "system_prompt": system_prompt,
                        "byok_only": byok_only,
                        "user_email": user_email,
                    },
                    config
                )
                if result:
                    ai_messages = result.get("messages", [])
                    final_message = None
                    AI_RESPONSE_ = [
                        msg for msg in ai_messages if isinstance(msg, AIMessage)]
                    if AI_RESPONSE_:
                        final_message = AI_RESPONSE_[-1].content
                    else:
                        logger.error(f"No AI Response Found In State")
                        raise HTTPException(
                            status_code=500, detail="No AI Message")
                    
                    # Get updated summary from result
                    updated_summary = result.get("summary", existing_summary)
                else:
                    logger.error(f"No AI Response Found In State")
                    raise HTTPException(
                        status_code=500, detail="Invoke Messgae Failed")

            except HTTPException as http_exc:
                raise http_exc
            except KeyRejectedError:
                raise  # V3: the route answers key_rejected; nothing is stored
            except Exception as e:
                logger.error(f"getGraphResponse Graph invoke : {e}")
                return False

            if final_message is None:
                raise RuntimeError("Agent produced no output")

            # Persistence decision matrix:
            #   regenerate_last_user → user msg already in DB from fork-init;
            #                          save only the AI response.
            #   is_duplicate (retry) → both already in DB; save nothing.
            #   normal               → save both.
            should_save_user = (not is_duplicate) and (not regenerate_last_user)
            should_save_assistant = regenerate_last_user or (not is_duplicate)

            if should_save_user or should_save_assistant:
                try:
                    updated_memory_facts = self._update_thread_memory_facts(
                        user_id=user_id,
                        thread_id=thread_id,
                        previous_facts=existing_memory_facts,
                        new_messages=[
                            {"role": "user", "text": query},
                            {"role": "assistant", "text": final_message},
                        ],
                    )

                    if should_save_user:
                        self.mongo_store.add_message(
                            user_id=user_id,
                            thread_id=thread_id,
                            role="user",
                            text=query,
                            message_id=msg_id,
                            embedding=get_embedding(query) or [],
                            summary=updated_summary,
                            summarize_fn="None",
                            embed_summary_fn=get_embedding,
                            context_fn=[]
                        )

                    if should_save_assistant:
                        answer_text = str(final_message) if not isinstance(final_message, str) else final_message
                        self.mongo_store.add_message(
                            user_id=user_id,
                            thread_id=thread_id,
                            role="assistant",
                            message_id=f"{msg_id}_ai",
                            text=answer_text,
                            embedding=get_embedding(str(final_message)) or [],
                            summary=updated_summary,
                            summarize_fn="None",
                            embed_summary_fn=get_embedding,
                            context_fn=[],
                            citations=(extract_citations(answer_text, passages)
                                       if passages is not None else None),
                        )
                except Exception as e:
                    logger.error(f"getGraphResponse Convo save : {e}")
                    return False

            return final_message
        except KeyRejectedError:
            raise
        except Exception as e:
            logger.error(f"get_graph_res : {e}")
            return False

    def _prepare_stream_input(
        self,
        query,
        config,
        user_id,
        thread_id,
        msg_id,
        context_node_ids=None,
        system_prompt=None,
        last_k_messages=None,
        external_context_top_k=None,
        regenerate_last_user: bool = False,
        web_search: bool = False,
        board_reads: Optional[Reads] = None,
        budget_tokens: Optional[int] = None,
        user_email: Optional[str] = None,
        byok_only: bool = False,
    ):
        """Shared preparation logic for both sync and async streaming."""
        # Check for existing state in LangGraph
        current_state = self.graph.get_state(config)
        initial_messages = []

        summary_state = self.mongo_store.get_thread_summary_state(user_id, thread_id)
        existing_summary = summary_state["summary"]
        watermark = summary_state["summarized_up_to_position"]

        # Hydrate state from DB if checkpointer has no messages for this thread.
        # Filter to messages past the summary watermark so already-summarized
        # turns aren't replayed through the LLM (they stay in the DB for UI).
        if not current_state.values or not current_state.values.get("messages"):
            initial_messages = self._hydrate_recent_messages(
                user_id, thread_id, watermark, recent_limit_override=last_k_messages
            )

        existing_memory_facts = self.mongo_store.get_thread_memory_facts(user_id, thread_id)
        prompt_summary = _clip_text(existing_summary, MAX_SUMMARY_CHARS)

        # Built after hydration so the token budget can count the history.
        # V3 turns collect the numbered passages they send (n -> Citation); V2 turns pass None.
        passages: Optional[dict] = {} if board_reads is not None else None
        context_snippets, external_context_str, web_hits, linked_answers = self._build_retrieved_context(
            query, user_id, thread_id, context_node_ids=context_node_ids,
            external_context_top_k=external_context_top_k,
            web_search=web_search,
            board_reads=board_reads,
            budget_tokens=budget_tokens,
            user_email=user_email,
            fixed_chars=len(query) + len(prompt_summary or "") + _history_chars(_budget_history(current_state, initial_messages)),
            passages=passages,
        )

        prompt = get_formated_prompt(query, user_id)
        timestamp = datetime.utcnow().isoformat()

        if regenerate_last_user:
            # Forking from a user message: the buffer already ends with that
            # user message. Don't append a duplicate — replay the buffer tail
            # as the prompt and persist only the assistant's reply.
            if not initial_messages or not isinstance(initial_messages[-1], HumanMessage):
                logger.warning(
                    "regenerate_last_user requested but stream buffer tail is not a user "
                    "message; falling back to normal append for thread %s",
                    thread_id,
                )
                user_msg = HumanMessage(
                    content=prompt,
                    id=msg_id,
                    metadata={"user_id": user_id, "thread_id": thread_id, "timestamp": timestamp},
                )
                messages_input = initial_messages + [user_msg]
                is_duplicate = False
            else:
                logger.info(
                    "regenerate_last_user (stream): replaying buffer tail for thread %s",
                    thread_id,
                )
                messages_input = initial_messages
                is_duplicate = True
        else:
            user_msg = HumanMessage(
                content=prompt,
                id=msg_id,
                metadata={"user_id": user_id, "thread_id": thread_id, "timestamp": timestamp}
            )

            messages_input = initial_messages + [user_msg]

            # Deduplicate
            is_duplicate = False
            if initial_messages:
                last_msg = initial_messages[-1]
                if _is_duplicate_hydrated_user_message(last_msg, msg_id):
                    is_duplicate = True
                if is_duplicate:
                    logger.info(f"Duplicate message detected in stream hydration: {msg_id}.")
                    messages_input = initial_messages

        graph_input = {
            "user_id": user_id,
            "thread_id": thread_id,
            "messages": messages_input,
            "summary": prompt_summary,
            "summarized_up_to_position": watermark,
            "memory_facts": existing_memory_facts,
            "context": context_snippets,
            "external_context": external_context_str,
            "linked_answers": linked_answers,
            # Always set, so a checkpointed value from an earlier turn never carries over.
            "citation_note": CITE_INSTRUCTION if passages else None,
            "context_node_ids": context_node_ids,
            "model": config.get("configurable", {}).get("model"),
            "temperature": config.get("configurable", {}).get("temperature"),
            "max_output_tokens": config.get("configurable", {}).get("max_output_tokens"),
            "last_k_messages": config.get("configurable", {}).get("last_k_messages"),
            "system_prompt": system_prompt,
            # Always set, so a V3 flag never carries over from a checkpoint.
            "byok_only": byok_only,
            "user_email": user_email,
        }

        return graph_input, existing_summary, existing_memory_facts, is_duplicate, query, web_hits, passages

    def _hydrate_recent_messages(
        self,
        user_id: str,
        thread_id: str,
        watermark: int = 0,
        recent_limit_override: Optional[int] = None,
    ):
        """
        Pull recent messages out of the DB to seed LangGraph state.

        Only messages newer than `watermark` are returned: anything at or below
        is already represented in the rolling summary, and replaying it would
        double-count tokens. The DB rows themselves are never deleted — the UI
        still loads full history through other endpoints.
        """
        if recent_limit_override is None:
            recent_limit = max(2, int(getattr(settings, "KEEP_LAST_MESSAGES", 6)))
        else:
            recent_limit = max(0, min(50, int(recent_limit_override)))
        if recent_limit == 0:
            return []
        mongo_msgs = self.mongo_store.get_thread_messages_after(
            user_id, thread_id, after_position=int(watermark or 0), limit=recent_limit,
        )
        initial_messages = []
        for m in mongo_msgs:
            clipped_text = _clip_text(m["text"], MAX_RECENT_MESSAGE_CHARS)
            if m["role"] == "user":
                initial_messages.append(HumanMessage(content=clipped_text, id=m["message_id"]))
            elif m["role"] in ["assistant", "ai"]:
                initial_messages.append(AIMessage(content=clipped_text, id=m["message_id"]))
        return initial_messages

    def _build_retrieved_context(
        self,
        query: str,
        user_id: str,
        thread_id: str,
        context_node_ids: Optional[list] = None,
        external_context_top_k: Optional[int] = None,
        web_search: bool = False,
        board_reads: Optional[Reads] = None,
        budget_tokens: Optional[int] = None,
        fixed_chars: int = 0,
        user_email: Optional[str] = None,
        passages: Optional[dict] = None,
    ):
        """`passages` (V3 only) is filled with n -> Citation for every numbered passage sent."""
        context_snippets: list[str] = []
        external_context_snippets: list[str] = []
        web_hits: list[dict] = []

        # Web search is embedding-independent: it must work even when the
        # embedding provider is down, and it never blocks the turn.
        if web_search and getattr(settings, "WEB_SEARCH_ENABLED", True):
            web_hits = search_web(query, max_results=WEB_CONTEXT_LIMIT)
            if web_hits:
                logger.info(
                    "Web search injected %s snippets for thread %s", len(web_hits), thread_id
                )
            external_context_snippets.extend(
                format_web_snippets(web_hits, MAX_WEB_SNIPPET_CHARS)
            )

        # Webpage parser: URLs the user pasted are fetched directly (SSRF-
        # guarded, size- and char-capped in web_fetch) regardless of the web
        # search toggle — pasting a link means "read this". Pages join
        # web_hits so the UI source chips show them.
        if getattr(settings, "WEB_FETCH_ENABLED", True):
            pasted_urls = extract_urls(query)
            if pasted_urls:
                pages = fetch_pages(pasted_urls)
                if pages:
                    logger.info(
                        "URL fetch injected %s pages for thread %s", len(pages), thread_id
                    )
                    external_context_snippets.extend(format_page_snippets(pages))
                    web_hits = list(web_hits) + [
                        {"title": p.get("title"), "url": p.get("url"), "snippet": ""}
                        for p in pages
                    ]

        query_embedding = get_embedding(query) or []

        # V3: the server read set decides what is read (contextNodeIds is ignored).
        # Notes and linked answers need no embedding, so they go in before the early return.
        board_ctx = None
        linked_answers: Optional[str] = None
        if board_reads is not None:
            board_top_k = _default_top_k({})  # same default as the read-set token meter
            if external_context_top_k is not None:
                board_top_k = max(0, min(10, int(external_context_top_k)))
            try:
                board_ctx = build_board_context(
                    board_store, board_reads, user_email or board_store.resolve_user_email(user_id),
                    query_embedding, board_top_k,
                )
            except Exception as e:
                logger.exception("Board context build failed for chat %s (thread %s)",
                                 board_reads.chat_id, thread_id)

        def add_board_blocks(kept) -> Optional[str]:
            """Numbered notes and kept chunks join EXTERNAL_CONTEXT; returns the LINKED_ANSWERS block.
            Numbers run in prompt order over what is actually sent (citations plan E2)."""
            numbered = number_passages(board_ctx, kept)
            external_context_snippets.extend(numbered.notes + numbered.sources)
            if passages is not None:
                passages.update(numbered.passages)
            return numbered.linked_answers

        if not _has_meaningful_embedding(query_embedding):
            if board_ctx is not None:
                linked_answers = add_board_blocks([])
            joined = "\n\n".join(external_context_snippets) if external_context_snippets else None
            return [], joined, web_hits, linked_answers

        try:
            thread_scope = self.mongo_store.get_thread_ancestry_scopes(thread_id)
            sims = self.mongo_store.find_similar_by_message_id(
                user_id=user_id,
                thread_queries=thread_scope,
                query_embeddings=query_embedding,
                top_k=SIMILAR_CONTEXT_LIMIT,
                min_score=SIMILARITY_MIN_SCORE,
            ) or []
            for idx, s in enumerate(sims):
                score = s.get("score")
                role = s.get("role")
                text = _clip_text(s.get("text", ""), MAX_SIMILAR_SNIPPET_CHARS)
                msg_id_sim = s.get("message_id")
                score_str = f"{score:.3f}" if score is not None else "n/a"
                context_snippets.append(
                    f"[sim {idx + 1} | role={role} | score={score_str}] id={msg_id_sim}: {text}"
                )
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")

        if board_reads is not None:
            if board_ctx is not None:
                kept = []
                if board_ctx.chunks:
                    fixed = (fixed_chars + sum(len(x) for x in external_context_snippets)
                             + sum(len(x) for x in board_ctx.notes)
                             + sum(len(x) for x in context_snippets) + len(board_ctx.linked_answers or ""))
                    kept = trim_chunks_to_budget(board_ctx.chunks, fixed, budget_tokens)
                linked_answers = add_board_blocks(kept)
        else:
            try:
                file_context_limit = FILE_CONTEXT_LIMIT
                if external_context_top_k is not None:
                    file_context_limit = max(0, min(10, int(external_context_top_k)))
                file_chunks = self.mongo_store.get_related_file_context(
                    node_id=thread_id,
                    query_embedding=query_embedding,
                    limit=file_context_limit,
                    context_node_ids=context_node_ids,
                ) or []
                for idx, c in enumerate(file_chunks):
                    meta = c.get("metadata")
                    try:
                        meta = json.loads(meta) if isinstance(meta, str) else (meta or {})
                    except Exception:
                        meta = meta or {}
                    external_context_snippets.append(
                        f"[file {idx + 1} | {c.get('file_name')} | type={c.get('file_type')} | chunk={c.get('chunk_index')}] "
                        f"{_clip_text(c.get('chunk_text', ''), MAX_FILE_SNIPPET_CHARS)}\n"
                        f"meta: {_clip_text(meta, MAX_FILE_METADATA_CHARS)}"
                    )
            except Exception as e:
                logger.error(f"File context search failed: {e}")

        external_context_str = "\n\n".join(external_context_snippets) if external_context_snippets else None
        return context_snippets, external_context_str, web_hits, linked_answers

    async def get_stream_response(
        self,
        query: str,
        config: dict,
        user_id: str,
        thread_id: str,
        msg_id: str,
        context_node_ids: Optional[list] = None,
        system_prompt: Optional[str] = None,
        last_k_messages: Optional[int] = None,
        external_context_top_k: Optional[int] = None,
        regenerate_last_user: bool = False,
        web_search: bool = False,
        board_reads: Optional[Reads] = None,
        budget_tokens: Optional[int] = None,
        user_email: Optional[str] = None,
        byok_only: bool = False,
    ):
        try:
            (graph_input, existing_summary, existing_memory_facts, is_duplicate, raw_query, web_hits,
             passages) = self._prepare_stream_input(
                query,
                config,
                user_id,
                thread_id,
                msg_id,
                context_node_ids=context_node_ids,
                system_prompt=system_prompt,
                last_k_messages=last_k_messages,
                external_context_top_k=external_context_top_k,
                regenerate_last_user=regenerate_last_user,
                web_search=web_search,
                board_reads=board_reads,
                budget_tokens=budget_tokens,
                user_email=user_email,
                byok_only=byok_only,
            )

            full_response = ""
            updated_summary = existing_summary
            streamed = False

            # Preamble: tell the client what the web search found BEFORE the
            # tokens start, so the UI can render "Searched the web · N
            # sources" chips while the model is still thinking.
            if web_hits:
                yield f"data: {json.dumps({'web_search': {'results': [{'title': h.get('title'), 'url': h.get('url')} for h in web_hits]}})}\n\n"

            # Try async streaming with async graph first
            stream_graph = self.async_graph if self.async_graph else None
            if stream_graph:
                try:
                    from app.agent.utils.saver import ensure_async_pool_open
                    if not await ensure_async_pool_open():
                        raise RuntimeError("async checkpoint pool unavailable")
                    in_think = False
                    async for event in stream_graph.astream_events(graph_input, config, version="v1"):
                        kind = event["event"]
                        if kind == "on_chat_model_stream":
                            reasoning, content = _split_reasoning(event["data"]["chunk"])
                            out = ""
                            if reasoning:
                                if not in_think:
                                    out += "<think>"
                                    in_think = True
                                out += reasoning
                            if content:
                                if in_think:
                                    out += "</think>"
                                    in_think = False
                                out += content
                            if out:
                                full_response += out
                                yield f"data: {json.dumps({'message': out})}\n\n"
                    if in_think:
                        # model ended while "thinking" — close the tag so the
                        # UI collapses the block instead of spinning forever
                        full_response += "</think>"
                        yield f"data: {json.dumps({'message': '</think>'})}\n\n"
                    streamed = True
                except KeyRejectedError:
                    raise  # not a streaming failure: the sync fallback would ask the provider again
                except (NotImplementedError, Exception) as e:
                    logger.warning(f"Async streaming failed, using sync fallback: {e}")
                    streamed = False

            # Sync fallback
            if not streamed:
                result = self.graph.invoke(graph_input, config)
                ai_messages = result.get("messages", []) if result else []
                ai_responses = [msg for msg in ai_messages if isinstance(msg, AIMessage)]
                if ai_responses:
                    reasoning, answer = _split_reasoning(ai_responses[-1])
                    full_response = (
                        f"<think>{reasoning}</think>" if reasoning else ""
                    ) + answer
                    yield f"data: {json.dumps({'message': full_response})}\n\n"
                updated_summary = result.get("summary", existing_summary) if result else existing_summary

            if not full_response:
                logger.error("Agent produced no output in stream")
                return

            # V3: the passages the finished answer cited, right after the last delta. Always sent
            # for V3 ([] when none). A stream that errors or produces nothing never gets here, so
            # it sends no citations frame (the client sees the error frame instead).
            citations = None
            if passages is not None:
                citations = extract_citations(full_response, passages)
                yield f"data: {json.dumps({'citations': citations})}\n\n"

            # Persistence decision matrix (mirrors the sync get_response):
            #   regenerate_last_user → user msg already in DB from fork-init;
            #                          save only the AI response.
            #   is_duplicate (retry) → both already in DB; save nothing.
            #   normal               → save both.
            should_save_user = (not is_duplicate) and (not regenerate_last_user)
            should_save_assistant = regenerate_last_user or (not is_duplicate)
            try:
                if should_save_user or should_save_assistant:
                    updated_memory_facts = self._update_thread_memory_facts(
                        user_id=user_id,
                        thread_id=thread_id,
                        previous_facts=existing_memory_facts,
                        new_messages=[
                            {"role": "user", "text": raw_query},
                            {"role": "assistant", "text": full_response},
                        ],
                    )
                    if should_save_user:
                        self.mongo_store.add_message(
                            user_id=user_id, thread_id=thread_id,
                            role="user", text=raw_query, message_id=msg_id,
                            embedding=get_embedding(raw_query) or [],
                            summary=updated_summary, summarize_fn="None",
                            embed_summary_fn=get_embedding, context_fn=[]
                        )
                    if should_save_assistant:
                        self.mongo_store.add_message(
                            user_id=user_id, thread_id=thread_id,
                            role="assistant", message_id=f"{msg_id}_ai",
                            text=full_response,
                            embedding=get_embedding(full_response) or [],
                            summary=updated_summary, summarize_fn="None",
                            embed_summary_fn=get_embedding, context_fn=[],
                            citations=citations,
                        )
            except Exception as e:
                logger.error(f"getStreamResponse Convo save : {e}")

            # Include updated summary in the done signal so frontend can update
            done_data = {"summary": self.mongo_store.get_thread_summary(user_id, thread_id)}
            yield f"data: {json.dumps(done_data)}\n\n"
            yield "data: [DONE]\n\n"

        except KeyRejectedError as e:
            # V3: the provider refused the user's key. One coded frame, no [DONE], nothing stored
            # (neither the question nor an answer): the board re-asks once a key is saved.
            logger.warning("V3 turn on thread %s: %s refused the user's key", thread_id, e.provider)
            yield f"data: {json.dumps(key_rejected_frame(e.provider))}\n\n"
        except Exception as e:
            logger.error(f"get_stream_response : {repr(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    def _update_thread_memory_facts(
        self,
        *,
        user_id: str,
        thread_id: str,
        previous_facts: dict | None,
        new_messages: list[dict],
    ) -> dict:
        updated_memory_facts = extract_memory_facts(previous_facts, new_messages)
        self.mongo_store.update_thread_memory_facts(user_id, thread_id, updated_memory_facts)
        return updated_memory_facts
