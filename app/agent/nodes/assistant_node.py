from langgraph.graph import StateGraph, START, END

from app.agent.state import State
from app.agent.helpers.load_prompt import load_prompt_from_yaml
from app.agent.helpers.get_llm import get_llm
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.agent.prompts.prompt_formation import get_formated_summury_prompt
from langchain_core.messages import RemoveMessage
from langchain_core.prompts import PromptTemplate
from datetime import datetime

from app.core.logger import logger


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

    def assistant(self, state: State) -> State:
        try:
            current_count = state.get("count", 0)
            thread_id = state.get("thread_id", None)
            context = state.get("context", [])
            external_context = state.get("external_context", None)
            summary = state.get("summary", None)
            model = state.get("model", None)
            temperature = state.get("temperature", 0.9)
            history = state.get("messages", [])

            template = load_prompt_from_yaml("SYSTEM_PROMPT")
            current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

            user_query = None
            for msg in reversed(history):
                if isinstance(msg, HumanMessage):
                    user_query = msg.content
                    break

            prompt_template = PromptTemplate.from_template(template)
            system_content = prompt_template.format(
                summary=summary if summary else "None",
                current_time=current_time,
                user_query=user_query or "",
            )

            if context:
                context_str = "\n".join(context) if isinstance(context, list) else str(context)
                system_content += f"\n\n<RELEVANT_CONTEXT>\n{context_str}\n</RELEVANT_CONTEXT>"

            if external_context:
                system_content += f"\n\n<EXTERNAL_CONTEXT>\n{external_context}\n</EXTERNAL_CONTEXT>"

            messages = [SystemMessage(content=system_content)] + history

            llm = get_llm(model)
            if llm is None:
                raise RuntimeError("LLM initialization failed for model: " + str(model))

            ai_response: AIMessage = llm.invoke(input=messages)
            normalized_content = _message_content_to_text(ai_response.content)
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
                "thread_id": thread_id,
                "context": context,
                "external_context": external_context,
                "summary": summary,
                "model": model,
                "temperature": temperature,
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
            model = state.get("model")
            llm = get_llm(model)
            if not llm:
                logger.error("Summarization skipped: LLM init failed")
                return {"messages": [], "summary": state.get("summary")}

            summary_msg = llm.invoke(prompt)
            new_summary = _message_content_to_text(summary_msg.content)

            if self.mongo_store:
                thread_id = state.get("thread_id")
                messages = state.get("messages", [])
                user_id = None
                for msg in reversed(messages):
                    if hasattr(msg, "metadata") and msg.metadata:
                        user_id = msg.metadata.get("user_id")
                        if user_id:
                            break

                if user_id and thread_id:
                    try:
                        self.mongo_store.update_thread_summary(
                            user_id=user_id,
                            thread_id=thread_id,
                            summary=new_summary,
                        )
                    except Exception as e:
                        logger.error(f"Failed to save summary for thread {thread_id}: {e}")

            # Keep the last N messages as configured (not hardcoded to 3)
            keep = getattr(settings, "KEEP_LAST_MESSAGES", 6)
            messages_to_delete = [RemoveMessage(id=m.id) for m in state["messages"][:-keep]]
            return {"messages": messages_to_delete, "summary": new_summary}
        except Exception as e:
            logger.error(f"Summarization node error: {e}")
            return {"messages": [], "summary": state.get("summary")}

    def get_nodes(self) -> dict:
        return {
            "assistant": self.assistant,
            "summurize": self.summurize,
            "summury_decision": self.summury_decision,
        }
