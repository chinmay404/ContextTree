from typing import TypedDict, Annotated, List, Optional, Any
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    user_id: Optional[str]
    count: int
    thread_id: Optional[str]
    context: List[str]
    external_context: Optional[str]
    # IDs of external-context nodes currently attached to this chat node.
    # None  → backend falls back to persisted edges (legacy behavior)
    # []    → external context explicitly disabled for this turn
    # [...] → exact set of context nodes to RAG over
    context_node_ids: Optional[List[str]]
    summary: Optional[str]
    # Highest `messages.position` value already folded into `summary`.
    # Used to skip re-hydrating already-summarized turns into LangGraph state
    # without deleting them from the messages table (the UI still displays them).
    summarized_up_to_position: int
    memory_facts: Optional[dict[str, Any]]
    model: Optional[str]
    temperature: Optional[float]
    messages: Annotated[List[AnyMessage], add_messages]
