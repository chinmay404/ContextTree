from typing import List, Union
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.Agent.state import State
from app.Agent.helpers.load_prompt import load_prompt_from_yaml
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain import PromptTemplate





def get_formated_prompt(user_query: str , user_id: str) :
    template = load_prompt_from_yaml("SYSTEM_PROMPT")
    prompt_template = PromptTemplate(
        input_variables=["user_query", "summary"],
        template=template
    )
    final_prompt = prompt_template.format(
        user_query=user_query,  
        summary="None its a new query"  
    )
    return final_prompt

    
