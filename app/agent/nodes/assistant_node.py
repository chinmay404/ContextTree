from langgraph.graph import StateGraph, START, END

from app.agent.state import State
from app.agent.helpers.load_prompt import load_prompt_from_yaml
from app.agent.helpers.get_llm import get_llm
from app.agent.helpers.memory_model import get_summary_model_name
from app.agent.memory import (
    build_memory_context_block,
    sanitize_chat_response_text,
    sanitize_summary_text,
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.agent.prompts.prompt_formation import get_formated_summury_prompt
from langchain_core.messages import RemoveMessage
from langchain_core.prompts import PromptTemplate
from datetime import datetime

from app.core.logger import logger
from app.core.observability import build_langsmith_trace_config


def _message_content_to_text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue

            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    continue

                nested_text = item.get("text", {}).get("value") if isinstance(item.get("text"), dict) else None
                if isinstance(nested_text, str):
                    parts.append(nested_text)
                    continue

                value = item.get("value") or item.get("content")
                if isinstance(value, str):
                    parts.append(value)

        return "\n".join(part.strip() for part in parts if isinstance(part, str) and part.strip()).strip()

    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        value = content.get("value") or content.get("content")
        if isinstance(value, str):
            return value

    return str(content)


class AgentNodes:
    def __init__(self, mongo_store=None):
        self.mongo_store = mongo_store

    def _resolve_state_user_id(self, state: State):
        user_id = state.get("user_id")
        if user_id:
            return user_id

        for msg in reversed(state.get("messages", [])):
            metadata = getattr(msg, "metadata", None) or {}
            if isinstance(metadata, dict) and metadata.get("user_id"):
                return metadata.get("user_id")

        return None

    def assistant(self, state: State) -> State:
        try:
            current_count = state.get("count", 0)
            thread_id = state.get("thread_id", None)
            context = state.get("context", [])
            external_context = state.get("external_context", None)
            summary = state.get("summary", None)
            memory_facts = state.get("memory_facts", {})
            model = state.get("model", None)
            temperature = state.get("temperature", 0.9)
            custom_system_prompt = str(state.get("system_prompt") or "").strip()
            history = state.get("messages", [])
            user_id = self._resolve_state_user_id(state)

            template = load_prompt_from_yaml("SYSTEM_PROMPT")
            current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

            user_query = None
            for msg in reversed(history):
                if isinstance(msg, HumanMessage):
                    user_query = msg.content
                    break

            prompt_template = PromptTemplate.from_template(template)
            system_content = prompt_template.format(
                current_time=current_time,
                user_query=user_query or "",
            )
            memory_context = build_memory_context_block(
                summary=sanitize_summary_text(summary),
                memory_facts=memory_facts,
                context=context,
                external_context=external_context,
            )

            messages = [
                SystemMessage(content=system_content),
                SystemMessage(content=memory_context),
            ]
            if custom_system_prompt:
                messages.append(
                    SystemMessage(
                        content=(
                            "Node-specific custom system prompt. Apply these instructions "
                            f"for this node and its branch:\n{custom_system_prompt}"
                        )
                    )
                )
            messages += history

            llm = get_llm(model, user_id=user_id)
            if llm is None:
                raise RuntimeError("LLM initialization failed for model: " + str(model))

            ai_response: AIMessage = llm.with_config(
                build_langsmith_trace_config(
                    run_name="contexttree_assistant_llm",
                    conversation_id=thread_id,
                    user_id=user_id,
                    model_name=model,
                    node_name="assistant",
                    tags=["chat-model"],
                )
            ).invoke(input=messages)
            normalized_content = sanitize_chat_response_text(_message_content_to_text(ai_response.content))
            if normalized_content != ai_response.content:
                ai_response = AIMessage(
                    content=normalized_content,
                    additional_kwargs=ai_response.additional_kwargs,
                    response_metadata=ai_response.response_metadata,
                    id=ai_response.id,
                    tool_calls=getattr(ai_response, "tool_calls", []),
                    invalid_tool_calls=getattr(ai_response, "invalid_tool_calls", []),
                )

            return {
                "messages": [ai_response],
                "count": current_count + 1,
                "user_id": user_id,
                "thread_id": thread_id,
                "context": context,
                "external_context": external_context,
                "summary": sanitize_summary_text(summary),
                "memory_facts": memory_facts,
                "model": model,
                "temperature": temperature,
                "system_prompt": custom_system_prompt,
            }
        except Exception as e:
            logger.error("Assistant node error: %s", e)
            return {"messages": state.get("messages", [])}

    def summury_decision(self, state: State) -> bool:
        try:
            from app.core.config import settings
            count = int(state.get("count", 0))
            limit = getattr(settings, "MAX_MESSAGES_BEFORE_SUMMARY", 10)
            return count > limit
        except (KeyError, TypeError, ValueError) as e:
            logger.error("Error in summary decision: %s", e)
            return False

    def summurize(self, state: State):
        try:
            from app.core.config import settings
            prompt = get_formated_summury_prompt(state["messages"], state["summary"])
            model = get_summary_model_name(state.get("model"))
            user_id = self._resolve_state_user_id(state)
            llm = get_llm(model, user_id=user_id)
            if not llm:
                logger.error("Summarization skipped: LLM init failed")
                return {"messages": [], "summary": state.get("summary")}

            summary_msg = llm.with_config(
                build_langsmith_trace_config(
                    run_name="contexttree_summary_llm",
                    conversation_id=state.get("thread_id"),
                    user_id=user_id,
                    model_name=model,
                    node_name="summary",
                    tags=["summary", "chat-model"],
                )
            ).invoke(prompt)
            new_summary = sanitize_summary_text(_message_content_to_text(summary_msg.content))

            keep = getattr(settings, "KEEP_LAST_MESSAGES", 6)
            messages_being_summarized = state["messages"][:-keep] if keep > 0 else state["messages"]

            # Compute the watermark: the highest DB position of any message
            # being folded into the summary. We look it up by message id, not
            # by index, because LangGraph state may not preserve every DB row.
            new_watermark = state.get("summarized_up_to_position") or 0
            if self.mongo_store and messages_being_summarized:
                thread_id = state.get("thread_id")
                if thread_id:
                    try:
                        ids = [getattr(m, "id", None) for m in messages_being_summarized]
                        ids = [i for i in ids if i]
                        if ids:
                            new_watermark = max(
                                new_watermark,
                                self.mongo_store.get_max_position_for_messages(thread_id, ids),
                            )
                    except Exception as e:
                        logger.error(f"Failed to compute summary watermark: {e}")

            if self.mongo_store:
                thread_id = state.get("thread_id")
                user_id = self._resolve_state_user_id(state)
                if user_id and thread_id:
                    try:
                        # Single UPDATE writes the new summary and advances the
                        # watermark atomically. Watermark only ever moves forward.
                        self.mongo_store.update_thread_summary(
                            user_id=user_id,
                            thread_id=thread_id,
                            summary=new_summary,
                            summarized_up_to_position=new_watermark,
                        )
                    except Exception as e:
                        logger.error(f"Failed to save summary for thread {thread_id}: {e}")

            # Drop summarized messages from LangGraph state. The DB rows stay
            # — the UI still needs them to render full history.
            messages_to_delete = [RemoveMessage(id=m.id) for m in messages_being_summarized]
            return {
                "messages": messages_to_delete,
                "summary": new_summary,
                "summarized_up_to_position": new_watermark,
            }
        except Exception as e:
            logger.error(f"Summarization node error: {e}")
            return {"messages": [], "summary": state.get("summary")}

    def get_nodes(self) -> dict:
        return {
            "assistant": self.assistant,
            "summurize": self.summurize,
            "summury_decision": self.summury_decision,
        }
