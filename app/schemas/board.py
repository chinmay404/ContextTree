"""Pydantic v2 models for the V3 board API (camelCase on the wire)."""
from copy import deepcopy
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

ID_PATTERN = r"^[A-Za-z0-9_-]{1,100}$"
SOURCE_TYPES = ("note", "pdf", "doc", "web", "video", "audio", "csv", "image", "pile")
STATUSES = ("processing", "ready", "failed", "soon")
CARRY = ("summary", "whole", "fresh")
MAX_CARDS_PER_BOARD = 50
MAX_POSITIONS = 200
DEFAULT_MODEL = "openai/gpt-oss-120b"

DEFAULT_ADVANCED = {
    "temperature": None,
    "maxOutputTokens": None,
    "historyMode": "auto",
    "lastKMessages": 6,
    "externalContextTopK": 5,
    "contextBudgetTokens": None,
}


def default_chat_data(name: str) -> dict:
    return {
        "name": name,
        "model": DEFAULT_MODEL,
        "persona": "none",
        "systemPrompt": "",
        "inheritDesk": True,
        "carry": "summary",
        "advanced": deepcopy(DEFAULT_ADVANCED),
        "hide": [],
    }


class _Model(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel, allow_inf_nan=False)


class BoardCreate(_Model):
    id: str = Field(pattern=ID_PATTERN)
    title: str = Field(min_length=1, max_length=200)


class Viewport(_Model):
    x: float
    y: float
    zoom: float


class BoardPatch(_Model):
    title: Optional[str] = Field(default=None, min_length=1, max_length=200)
    viewport: Optional[Viewport] = None


class CardCreate(_Model):
    id: str = Field(pattern=ID_PATTERN)
    kind: Literal["chat", "source"]
    source_type: Optional[Literal[SOURCE_TYPES]] = None
    x: float
    y: float
    desk_id: Optional[str] = Field(default=None, pattern=ID_PATTERN)
    pile_id: Optional[str] = Field(default=None, pattern=ID_PATTERN)
    parent_id: Optional[str] = Field(default=None, pattern=ID_PATTERN)
    forked_from_message_id: Optional[str] = Field(default=None, max_length=200)
    status: Optional[Literal[STATUSES]] = None
    data: dict[str, Any] = Field(default_factory=dict)


class CardPatch(_Model):
    """Use `model_fields_set` to tell an absent field from an explicit null."""
    data: Optional[dict[str, Any]] = None
    desk_id: Optional[str] = Field(default=None, pattern=ID_PATTERN)
    pile_id: Optional[str] = Field(default=None, pattern=ID_PATTERN)
    status: Optional[Literal[STATUSES]] = None


class Position(_Model):
    id: str = Field(pattern=ID_PATTERN)
    x: float
    y: float


class PositionsPatch(_Model):
    positions: list[Position] = Field(max_length=MAX_POSITIONS)


class EdgeCreate(_Model):
    id: str = Field(pattern=ID_PATTERN)
    kind: Literal["wire", "answer"]
    from_: str = Field(alias="from", pattern=ID_PATTERN)
    to: str = Field(pattern=ID_PATTERN)


class LinkCreate(_Model):
    id: str = Field(pattern=ID_PATTERN)
    url: str = Field(min_length=1, max_length=2048)
    x: float
    y: float
    desk_id: Optional[str] = Field(default=None, pattern=ID_PATTERN)
    desc: Optional[str] = Field(default=None, max_length=2000)
