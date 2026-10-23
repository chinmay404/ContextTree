"""
The web search query for a V3 turn (board polish 2, S1).

A follow-up like "what it can do for me" means nothing to a search engine. Before searching,
the chat's own model rewrites the message into one standalone query from the chat's name, its
last few turns and the titles of the sources it reads. The call:

- runs on the turn's TurnKey (the user's key, already looked up; no second lookup);
- is small: at most 32 output tokens, temperature 0, a 4 s limit;
- is never stored and never counts as a daily message (it is part of the same request).

Any failure (timeout, provider error, a refused key, an empty or oversized reply, fake mode,
no turn key) falls back to "<chat name> <message>". A refused key is not reported here: the
answer's own call reports key_rejected as usual.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.helpers.get_llm import _is_reasoning_model
from app.agent.helpers.v3_llm import get_v3_llm, turn_key_from
from app.core.config import settings
from app.core.logger import logger

REWRITE_MAX_TOKENS = 32
REWRITE_TIMEOUT_SECONDS = 4
MAX_QUERY_CHARS = 200
RECENT_MESSAGES = 6          # the last three turns
TURN_CHARS = 300
MAX_TITLES = 8
TITLE_CHARS = 80
MESSAGE_CHARS = 500
FALLBACK_MESSAGE_CHARS = 300

# The rewrite prompt's fixed opening (tests recognise the call by it).
REWRITE_MARKER = "Rewrite the user's last message as one standalone web search query."

# A rewrite that outlives its 4 s is abandoned, not waited for.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="web-query")


def _clip(text, n: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def fallback_search_query(chat_name: Optional[str], message: str) -> str:
    name = " ".join(str(chat_name or "").split())
    msg = _clip(message, FALLBACK_MESSAGE_CHARS)
    return f"{name} {msg}".strip() if name else msg


def _prompt(message, chat_name, history, source_titles) -> str:
    turns = []
    for m in list(history or [])[-RECENT_MESSAGES:]:
        if isinstance(m, HumanMessage):
            turns.append(f"User: {_clip(m.content, TURN_CHARS)}")
        elif isinstance(m, AIMessage):
            turns.append(f"Assistant: {_clip(m.content, TURN_CHARS)}")
    titles = [_clip(t, TITLE_CHARS) for t in (source_titles or []) if str(t or "").strip()][:MAX_TITLES]
    return (
        f"{REWRITE_MARKER} Use the chat topic, the sources and the recent turns to resolve words "
        "like \"it\" or \"this\". Reply with the query only: no quotes, at most 12 words.\n\n"
        f"Chat: {_clip(chat_name, TITLE_CHARS) or '(untitled)'}\n"
        f"Sources: {'; '.join(titles) or '(none)'}\n"
        f"Recent turns:\n{chr(10).join(turns) or '(none)'}\n\n"
        f"Last message: {_clip(message, MESSAGE_CHARS)}\n"
        "Query:"
    )


def _clean(reply) -> str:
    content = getattr(reply, "content", reply)
    if isinstance(content, list):
        content = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    lines = [ln.strip() for ln in str(content or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    line = lines[0]
    if line.lower().startswith("query:"):
        line = line[len("query:"):].strip()
    return " ".join(line.strip("\"'`“”‘’ ").split())


def rewrite_search_query(
    message: str,
    config: Optional[dict],
    *,
    chat_name: Optional[str],
    history=None,
    source_titles=None,
) -> str:
    """The standalone query for this V3 turn's web search, or the "<chat name> <message>" fallback."""
    fallback = fallback_search_query(chat_name, message)
    turn_key = turn_key_from(config)
    model_id = ((config or {}).get("configurable") or {}).get("model")
    if turn_key is None or not model_id or settings.ENGINE_FAKE_LLM:
        return fallback
    failure: Optional[str] = None
    try:
        llm = get_v3_llm(model_id, None, temperature=0, max_output_tokens=REWRITE_MAX_TOKENS,
                         resolved=turn_key.resolved())
        if turn_key.provider == "openai" and _is_reasoning_model(turn_key.route.wire_model):
            llm = llm.bind(reasoning_effort="minimal")  # 32 tokens leave no room to think at length
        prompt = [HumanMessage(content=_prompt(message, chat_name, history, source_titles))]
        reply = _executor.submit(llm.invoke, prompt).result(timeout=REWRITE_TIMEOUT_SECONDS)
        query = _clean(reply)
    except FutureTimeout:
        failure = "timeout"
    except Exception as exc:  # noqa: BLE001 - any failure means the plain fallback
        failure = type(exc).__name__
    if failure is not None:
        logger.info("Web query rewrite fell back (%s)", failure)
        return fallback
    if not query or len(query) > MAX_QUERY_CHARS:
        logger.info("Web query rewrite fell back (empty or oversized reply)")
        return fallback
    return query
