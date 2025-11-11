
from langgraph.graph import StateGraph, START, END
import re
import json

from app.agent.state import State
from app.agent.helpers.load_prompt import load_prompt_from_yaml
from app.agent.helpers.get_llm import get_groq_llm
from langchain_core.messages import HumanMessage, AIMessage
from app.agent.prompts.prompt_formation import get_formated_prompt
from app.agent.prompts.prompt_formation import get_formated_summury_prompt
from langchain_core.messages import RemoveMessage

from app.core.logger import logger


class AgentNodes:
    def __init__(self, mongo_store=None):
        self.state = State()
        self.groq_llm = get_groq_llm()
        self.sys_msg = load_prompt_from_yaml("REACT_LANGGRAPH_PROMPT")
        self.summury_prompt = load_prompt_from_yaml("SUMMARY_PROMPT")
        self.mongo_store = mongo_store

    # def assistant(self, state: State):
    #     try:
    #         llm = get_groq_llm()
    #         res: AIMessage = llm.invoke(input=[self.sys_msg] + state.messages)
    #     except Exception as e:
    #         print("Error in assistant node:", e)
    #         return {"messages": state.messages}
    #     updated = state.messages + [res]
    #     return {"messages": updated}

    def assistant(self, state: State) -> State:
        try:
            current_count = state.get("count", 0)
            thread_id = state.get("thread_id", None)
            context = state.get("context", [])
            external_context = state.get("external_context", None)
            summary = state.get("summary", None)
            model = state.get("model", "default-model-name")
            temperature = state.get("temperature", 0.9)
            history = state.get("messages", [])
            print("Message History:", history)
            ai_response: AIMessage = self.groq_llm.invoke(input=history)

            print("Message Current Count:", current_count)
            return {
                "messages": history + [ai_response],
                "count": current_count + 1,
                "thread_id": thread_id,
                "context": context,
                "external_context": external_context,
                "summary": summary,
                "model": model,
                "temperature": temperature,
            }
        except Exception as e:
            print("Assistant Node error:", e)
            logger.error("Assistant Node error:", e)

    def summury_decision(self, state: State) -> bool:
        try:
            count = int(state.get("count", 0))
            return count > 5
        except (KeyError, TypeError, ValueError) as e:
            print("Error in summary decision:", e)
            logger.error("Error in summary decision:", e)
            return False

    def summurize(self, state: State):
        try:
            prompt = get_formated_summury_prompt(
                state["messages"], state["summary"])
            summary_msg = self.groq_llm.invoke(prompt)
            new_summary = summary_msg.content
            
            # Save summary to MongoDB if mongo_store is available
            if self.mongo_store:
                thread_id = state.get("thread_id")
                # Extract user_id from the last message metadata
                messages = state.get("messages", [])
                user_id = None
                for msg in reversed(messages):
                    if hasattr(msg, 'metadata') and msg.metadata:
                        user_id = msg.metadata.get('user_id')
                        if user_id:
                            break
                
                if user_id and thread_id:
                    try:
                        self.mongo_store.update_thread_summary(
                            user_id=user_id,
                            thread_id=thread_id,
                            summary=new_summary
                        )
                        logger.info(f"Summary saved to MongoDB for thread {thread_id}")
                    except Exception as e:
                        logger.error(f"Failed to save summary to MongoDB: {e}")
            
            messages_to_delete = [RemoveMessage(
                id=m.id) for m in state["messages"][:-3]]
            return {"messages": messages_to_delete, "summary": new_summary}
        except Exception as e:
            print("Error in summarization node:", e)
            logger.error(f"Error in summarization node: {e}")
            return {"messages": state["messages"], "summary": None}

    def get_nodes(self) -> dict:
        return {
            "assistant": self.assistant,
            "summurize": self.summurize,
            "summury_decision": self.summury_decision,
        }
