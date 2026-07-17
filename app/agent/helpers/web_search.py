"""
Free web search via DuckDuckGo (the `ddgs` package) — no API key needed.

Results are injected into the <EXTERNAL_CONTEXT> block by
`_build_retrieved_context` when the request opts in (ChatMessage.web_search),
using the same labeled-snippet convention as file chunks so the context
inspector stays honest about what the model saw.

Failure policy: web search NEVER blocks a chat turn. Any error (network,
rate-limit, import) logs a warning and returns an empty list.
"""

from typing import List

from app.core.logger import logger


def search_web(query: str, max_results: int = 3, timeout: int = 6) -> List[dict]:
    """Return [{title, url, snippet}] from DuckDuckGo, or [] on any failure."""
    text = str(query or "").strip()
    if not text:
        return []

    try:
        from ddgs import DDGS
    except ImportError:
        logger.warning("ddgs not installed — web search unavailable")
        return []

    try:
        with DDGS(timeout=timeout) as client:
            raw = client.text(text, max_results=max(1, int(max_results)))
    except Exception as e:
        logger.warning(f"Web search failed (non-blocking): {e}")
        return []

    results: List[dict] = []
    for item in raw or []:
        title = str(item.get("title") or "").strip()
        url = str(item.get("href") or item.get("url") or "").strip()
        snippet = str(item.get("body") or item.get("snippet") or "").strip()
        if snippet or title:
            results.append({"title": title, "url": url, "snippet": snippet})
    return results


def format_web_snippets(results: List[dict], clip_chars: int) -> List[str]:
    """Label web hits like file chunks: [web N | title | url] snippet."""
    lines: List[str] = []
    for idx, r in enumerate(results):
        snippet = str(r.get("snippet") or "").strip()
        if len(snippet) > clip_chars:
            snippet = f"{snippet[: clip_chars - 1].rstrip()}…"
        lines.append(
            f"[web {idx + 1} | {r.get('title') or 'untitled'} | {r.get('url') or 'no-url'}] {snippet}"
        )
    return lines
