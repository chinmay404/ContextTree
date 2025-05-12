from .load_prompt import load_prompt_from_yaml
from langchain_core.prompts import ChatPromptTemplate



def get_sys_prompt():
    """Get system prompt from YAML file"""
    sys_prompt = load_prompt_from_yaml("REACT_LANGGRAPH_PROMPT")
    prompt_template = ChatPromptTemplate.from_messages([
    ("system", sys_prompt),
    ("placeholder", "{state}" )
    ])
    return prompt_template