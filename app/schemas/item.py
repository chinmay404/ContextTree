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
    parent_thread_id: Optional[str] = Field(
        None, description="Parent thread identifier", alias="parentNodeId")
    fork_at_message_id: Optional[str] = Field(
        None, description="Fork point message identifier", alias="forkedFromMessageId")
    temperature: Optional[float] = Field(
        None, description="Temperature setting for the model")
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
    user_id: Optional[str] = Field(None, description="User identifier")
