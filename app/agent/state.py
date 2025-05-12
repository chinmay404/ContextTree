from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    count: int
    thread_id: Optional[str]
    context: List[str]
    external_context: Optional[str]
    summary: Optional[str]
    model: Optional[str]
    temperature: Optional[float]
    messages: Annotated[List[AnyMessage], add_messages]
