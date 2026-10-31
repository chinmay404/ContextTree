"""
What went wrong on a turn, in words a person can read.

Every failure here used to end in a log line nobody read while the answer carried on as if nothing
had happened. Now each one logs `ENGINE_ALERT code=...`, which is searchable in the Vercel logs, and
is stored on the turn's trace. If it changes what the user got, the board shows them the message.
"""
from __future__ import annotations

from typing import Optional

from app.core.logger import logger

# code: (does the user need to know?, what they are told)
ISSUES: dict[str, tuple[bool, str]] = {
    "sources_unavailable": (True, "Your sources couldn't be read, so this answer doesn't use them. Try again."),
    "not_saved": (True, "This answer wasn't saved and may be gone after a refresh. Copy it if you need it."),
    "history_unavailable": (True, "This branch's earlier history couldn't be read, so this answer may miss it."),
    "history_partial": (True, "Part of this branch's history couldn't be summarised, so this answer may miss it."),
    "web_search_empty": (True, "Web search found nothing or failed, so this answer doesn't use the web."),
    # Embeddings are told to the user through the older `context` frame the board already reads.
    "embedding_down": (False, "Meaning search is down; sources were found by keyword only."),
    "board_map_failed": (False, "The board map couldn't be built for this turn."),
    "past_search_failed": (False, "Earlier messages couldn't be searched for this turn."),
    "jev_failed": (False, "Jev didn't answer, so passages stayed in search order."),
    "agent_failed": (False, "The agent couldn't go deeper on this model."),
    "trace_not_saved": (False, "This turn's trace wasn't saved."),
}


def alert(code: str, where: str, exc: Optional[BaseException] = None) -> None:
    """The exception's type only: provider errors can quote parts of a key."""
    logger.error("ENGINE_ALERT code=%s where=%s error=%s", code, where, type(exc).__name__ if exc else "-")


def note(trace: Optional[dict], code: str, where: str, exc: Optional[BaseException] = None) -> None:
    alert(code, where, exc)
    if trace is None:
        return
    told, message = ISSUES[code]
    issues = trace.setdefault("issues", [])
    if not any(i["code"] == code for i in issues):
        issues.append({"code": code, "message": message, "shown": told})


def shown_to_user(trace: Optional[dict], already: set[str] | None = None) -> list[dict]:
    """The frame's payload: what the user needs to hear and hasn't yet."""
    return [{"code": i["code"], "message": i["message"]} for i in (trace or {}).get("issues", [])
            if i["shown"] and i["code"] not in (already or set())]
