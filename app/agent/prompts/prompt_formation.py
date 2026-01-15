from typing import List, Union
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.agent.state import State
from app.agent.helpers.load_prompt import load_prompt_from_yaml
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from datetime import datetime


def get_formated_prompt(user_query: str, user_id: str):
    template = load_prompt_from_yaml("SYSTEM_PROMPT")
    current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    prompt_template = PromptTemplate(
        input_variables=["user_query", "summary", "current_time"],
        template=template
    )
    final_prompt = prompt_template.format(
        user_query=user_query,
        summary="None its a new query",
        current_time=current_time
    )
    return final_prompt


def get_formated_summury_prompt(messages, summury):
    template = load_prompt_from_yaml("SUMMARY_PROMPT")
    prompt_template = PromptTemplate(
        input_variables=["summary", "conversation"],
        template=template
    )
    final_prompt = prompt_template.format(
        conversation=messages,
        summary=summury,
    )
    return final_prompt
