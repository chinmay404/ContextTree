from typing import List, Union
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.agent.state import State
from app.agent.helpers.load_prompt import load_prompt_from_yaml
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from datetime import datetime


def get_formated_prompt(user_query: str, user_id: str):
    return user_query


def get_formated_summury_prompt(messages, summury):
    template = load_prompt_from_yaml("SUMMARY_PROMPT")
    prompt_template = PromptTemplate(
        input_variables=["summary", "conversation"],
        template=template
    )
    # Format LangChain message objects into readable conversation text
    conversation_lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            conversation_lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            conversation_lines.append(f"Assistant: {msg.content}")
        elif isinstance(msg, SystemMessage):
            conversation_lines.append(f"System: {msg.content}")
        elif isinstance(msg, dict):
            role = msg.get("role", "unknown")
            text = msg.get("text", "") or msg.get("content", "")
            conversation_lines.append(f"{role.capitalize()}: {text}")
        else:
            conversation_lines.append(str(msg))
    conversation_text = "\n".join(conversation_lines)

    final_prompt = prompt_template.format(
        conversation=conversation_text,
        summary=summury if summury else "None",
    )
    return final_prompt
