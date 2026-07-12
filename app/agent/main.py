
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
from app.agent.utils.embeddings import get_embedding
from app.core.config import settings
from app.core.logger import logger

from fastapi import HTTPException


SIMILAR_CONTEXT_LIMIT = 3
FILE_CONTEXT_LIMIT = 3
MAX_SUMMARY_CHARS = 2400
MAX_RECENT_MESSAGE_CHARS = 1400
MAX_SIMILAR_SNIPPET_CHARS = 320
MAX_FILE_SNIPPET_CHARS = 900
MAX_FILE_METADATA_CHARS = 220
SIMILARITY_MIN_SCORE = max(0.0, float(getattr(settings, "SIMILARITY_MIN_SCORE", 0.2)))


def _clip_text(value, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 1].rstrip()}…"


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


def _has_meaningful_embedding(embedding) -> bool:
    return bool(embedding) and any(abs(float(x)) > 1e-8 for x in embedding)


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
            context_snippets, external_context_str = self._build_retrieved_context(
                query, user_id, thread_id, context_node_ids=context_node_ids,
                external_context_top_k=external_context_top_k,
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
                        "context_node_ids": context_node_ids,
                        "model": config.get("configurable", {}).get("model"),
                        "temperature": config.get("configurable", {}).get("temperature"),
                        "max_output_tokens": config.get("configurable", {}).get("max_output_tokens"),
                        "last_k_messages": config.get("configurable", {}).get("last_k_messages"),
                        "system_prompt": system_prompt,
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
                        self.mongo_store.add_message(
                            user_id=user_id,
                            thread_id=thread_id,
                            role="assistant",
                            message_id=f"{msg_id}_ai",
                            text=str(final_message) if not isinstance(final_message, str) else final_message,
                            embedding=get_embedding(str(final_message)) or [],
                            summary=updated_summary,
                            summarize_fn="None",
                            embed_summary_fn=get_embedding,
                            context_fn=[]
                        )
                except Exception as e:
                    logger.error(f"getGraphResponse Convo save : {e}")
                    return False

            return final_message
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
    ):
        """Shared preparation logic for both sync and async streaming."""
        # Check for existing state in LangGraph
        current_state = self.graph.get_state(config)
        initial_messages = []
        context_snippets, external_context_str = self._build_retrieved_context(
            query, user_id, thread_id, context_node_ids=context_node_ids,
            external_context_top_k=external_context_top_k,
        )

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
            "context_node_ids": context_node_ids,
            "model": config.get("configurable", {}).get("model"),
            "temperature": config.get("configurable", {}).get("temperature"),
            "max_output_tokens": config.get("configurable", {}).get("max_output_tokens"),
            "last_k_messages": config.get("configurable", {}).get("last_k_messages"),
            "system_prompt": system_prompt,
        }

        return graph_input, existing_summary, existing_memory_facts, is_duplicate, query

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
    ):
        query_embedding = get_embedding(query) or []
        if not _has_meaningful_embedding(query_embedding):
            return [], None

        context_snippets: list[str] = []
        external_context_snippets: list[str] = []

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
        return context_snippets, external_context_str

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
    ):
        try:
            graph_input, existing_summary, existing_memory_facts, is_duplicate, raw_query = self._prepare_stream_input(
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
            )

            full_response = ""
            updated_summary = existing_summary
            streamed = False

            # Try async streaming with async graph first
            stream_graph = self.async_graph if self.async_graph else None
            if stream_graph:
                try:
                    from app.agent.utils.saver import ensure_async_pool_open
                    if not await ensure_async_pool_open():
                        raise RuntimeError("async checkpoint pool unavailable")
                    async for event in stream_graph.astream_events(graph_input, config, version="v1"):
                        kind = event["event"]
                        if kind == "on_chat_model_stream":
                            content = event["data"]["chunk"].content
                            if content:
                                full_response += content
                                yield f"data: {json.dumps({'message': content})}\n\n"
                    streamed = True
                except (NotImplementedError, Exception) as e:
                    logger.warning(f"Async streaming failed, using sync fallback: {e}")
                    streamed = False

            # Sync fallback
            if not streamed:
                result = self.graph.invoke(graph_input, config)
                ai_messages = result.get("messages", []) if result else []
                ai_responses = [msg for msg in ai_messages if isinstance(msg, AIMessage)]
                if ai_responses:
                    full_response = str(ai_responses[-1].content)
                    yield f"data: {json.dumps({'message': full_response})}\n\n"
                updated_summary = result.get("summary", existing_summary) if result else existing_summary

            if not full_response:
                logger.error("Agent produced no output in stream")
                return

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
                            embed_summary_fn=get_embedding, context_fn=[]
                        )
            except Exception as e:
                logger.error(f"getStreamResponse Convo save : {e}")

            # Include updated summary in the done signal so frontend can update
            done_data = {"summary": self.mongo_store.get_thread_summary(user_id, thread_id)}
            yield f"data: {json.dumps(done_data)}\n\n"
            yield "data: [DONE]\n\n"

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
