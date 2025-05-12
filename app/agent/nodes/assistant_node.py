
from langgraph.graph import StateGraph, START, END
import re
import json

from app.Agent.state import State
from app.Agent.helpers.load_prompt import load_prompt_from_yaml
from app.Agent.helpers.get_llm import get_groq_llm
from langchain_core.messages import HumanMessage, AIMessage
from app.Agent.prompts.prompt_formation import get_formated_prompt


class AgentNodes:
    def __init__(self):
        self.state = State()
        self.groq_llm = get_groq_llm()
        self.sys_msg = load_prompt_from_yaml("REACT_LANGGRAPH_PROMPT")

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
            current_count        = state.get("count", 0)
            thread_id            = state.get("thread_id", None)
            context              = state.get("context", [])
            external_context     = state.get("external_context", None)
            summary              = state.get("summary", None)
            model                = state.get("model", "default-model-name")
            temperature          = state.get("temperature", 1.0)
            history              = state.get("messages", [])
            ai_response: AIMessage = self.groq_llm.invoke(input=history)

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

    def summurize(self, state: State):
        return {"messages": [self.llm.invoke([self.sys_msg] + state["messages"])]}

    def get_nodes(self) -> dict:
        return {
            "assistant": self.assistant,
            "summurize": self.summurize,
        }
