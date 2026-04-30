"""
Pydantic schemas for the Item model.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class ItemBase(BaseModel):
    """Base Item schema with common attributes."""
    name: str = Field(..., description="Name of the item",
                      example="Example Item")
    description: Optional[str] = Field(
        None, description="Optional description of the item", example="This is an example item")
    price: Optional[float] = Field(
        None, description="Price of the item", example=19.99)
    is_active: bool = Field(True, description="Whether the item is active")


class ItemCreate(ItemBase):
    """Schema for creating a new item."""
    pass


class ItemUpdate(BaseModel):
    """Schema for updating an item with optional fields."""
    name: Optional[str] = Field(None, description="Name of the item")
    description: Optional[str] = Field(
        None, description="Description of the item")
    price: Optional[float] = Field(None, description="Price of the item")
    is_active: Optional[bool] = Field(
        None, description="Whether the item is active")


class ChatMessage(BaseModel):
    """Schema for chat messages."""
    model_config = ConfigDict(populate_by_name=True)
    message: str = Field(..., description="The user's message")
    message_id: str = Field(description="Unique identifier for the message")
    conversation_id: Optional[str] = Field(
        None, description="Conversation identifier", alias="nodeId")
    model_name: Optional[str] = Field(
        None, description="Name of the model used for the response", alias="model")
    system_prompt: Optional[str] = Field(
        None, description="Node-specific system prompt", alias="systemPrompt")
    parent_thread_id: Optional[str] = Field(
        None, description="Parent thread identifier", alias="parentNodeId")
    fork_at_message_id: Optional[str] = Field(
        None, description="Fork point message identifier", alias="forkedFromMessageId")
    temperature: Optional[float] = Field(
        None, description="Temperature setting for the model")
    max_output_tokens: Optional[int] = Field(
        None, description="Maximum output tokens", alias="maxOutputTokens")
    last_k_messages: Optional[int] = Field(
        None, description="Override recent history window", alias="lastKMessages")
    external_context_top_k: Optional[int] = Field(
        None, description="External context chunk count", alias="externalContextTopK")
    context: Optional[list] = Field(
        None, description="Context for the conversation")
    context_node_ids: Optional[list[str]] = Field(
        default=None,
        description=(
            "External-context node IDs the user has attached to this chat node "
            "right now. When provided this is the source of truth for RAG; "
            "passing [] disables external context for this turn. When None "
            "the backend falls back to the persisted canvas edges."
        ),
        alias="contextNodeIds",
    )
    regenerate_last_user: bool = Field(
        default=False,
        description=(
            "Set to true when forking from a user message and immediately "
            "asking the model to answer it again in the new branch. The fork "
            "buffer already ends with the user message; the backend skips "
            "appending a new HumanMessage and skips the user-message DB write, "
            "and only persists the assistant's fresh reply."
        ),
        alias="regenerateLastUser",
    )
    user_id: Optional[str] = Field(None, description="User identifier")
