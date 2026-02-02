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
    final_prompt = prompt_template.format(
        conversation=messages,
        summary=summury,
    )
    return final_prompt
