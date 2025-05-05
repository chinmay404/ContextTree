"""
Pydantic schemas for the Item model.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ItemBase(BaseModel):
    """Base Item schema with common attributes."""
    name: str = Field(..., description="Name of the item", example="Example Item")
    description: Optional[str] = Field(None, description="Optional description of the item", example="This is an example item")
    price: Optional[float] = Field(None, description="Price of the item", example=19.99)
    is_active: bool = Field(True, description="Whether the item is active")


class ItemCreate(ItemBase):
    """Schema for creating a new item."""
    pass


class ItemUpdate(BaseModel):
    """Schema for updating an item with optional fields."""
    name: Optional[str] = Field(None, description="Name of the item")
    description: Optional[str] = Field(None, description="Description of the item")
    price: Optional[float] = Field(None, description="Price of the item")
    is_active: Optional[bool] = Field(None, description="Whether the item is active")


class ChatMessage(BaseModel):
    """Schema for chat messages."""
    message: str = Field(..., description="The user's message")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")