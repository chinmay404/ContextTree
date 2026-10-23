"""
Webpage parser: fetch user-pasted URLs into <EXTERNAL_CONTEXT>.

Security (SSRF guard): the fetch goes through `safe_fetch` — http/https on
the default ports only, every hop of every redirect re-validated, every
resolved address public and globally routable (no loopback, private,
link-local incl. 169.254.169.254, CGNAT 100.64.0.0/10, reserved, multicast,
unspecified; IPv4, IPv6, v4-mapped), the connection pinned to the checked
address (no DNS rebinding), env proxies ignored. Limits: one wall-clock
deadline over DNS and every hop, `Accept-Encoding: identity` with only
gzip/deflate decoded here and the byte cap applied to decoded bytes, and
only text-ish content types are parsed.

Context-window safety: at most MAX_FETCH_URLS pages per turn, each page's
extracted text clipped to DEFAULT_MAX_CHARS — a pasted URL can never
flood the prompt.

Failure policy: like web search, fetching NEVER blocks a chat turn — any
error logs a warning and returns None/[].
"""

import ipaddress
import re
import socket
from html.parser import HTMLParser
from typing import List, Optional, Tuple

from app.agent.citations import neutralise_markers
from app.agent.helpers import safe_fetch
from app.core.logger import logger

MAX_FETCH_URLS = 2
FETCH_TIMEOUT_SECONDS = 8  # the whole fetch, every hop included (safe_fetch)
MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024  # 2 MB
DEFAULT_MAX_CHARS = 3500
_ALLOWED_CONTENT_TYPES = ("text/html", "application/xhtml", "text/plain")
_URL_RE = re.compile(r"https?://[^\s<>\"'\)\]\}]+", re.IGNORECASE)
# NUL and other control characters (Postgres rejects NUL in text; none of them is readable).
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "iframe", "head"}


class _TextExtractor(HTMLParser):
    """Readable text + <title> from HTML; scripts/styles/etc. skipped."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self.title_parts: List[str] = []
        self.text_parts: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title_parts.append(data)
        elif self._skip_depth == 0:
            self.text_parts.append(data)


def extract_urls(text: str, limit: int = MAX_FETCH_URLS) -> List[str]:
    """Unique http(s) URLs from a message, in order, capped at `limit`."""
    seen = set()
    out: List[str] = []
    for match in _URL_RE.finditer(str(text or "")):
        url = match.group(0).rstrip(".,;:!?")
        if url in seen:
            continue
        seen.add(url)
        out.append(url)
        if len(out) >= limit:
            break
    return out


_ip_is_public = safe_fetch.ip_is_public  # one definition of "public", shared with safe_fetch


def host_is_public(host: str) -> bool:
    """True only when EVERY address the host resolves to is public unicast."""
    name = str(host or "").strip().lower().rstrip(".")
    if not name or name == "localhost" or name.endswith(".localhost"):
        return False
    try:
        infos = socket.getaddrinfo(name, None)
    except (socket.gaierror, UnicodeError, ValueError):
        return False
    if not infos:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if not _ip_is_public(ip):
            return False
    return True


def _extract_text(content_type: str, body: str) -> Tuple[str, str]:
    """(title, collapsed text) from a fetched body."""
    if "html" in content_type or "xhtml" in content_type:
        parser = _TextExtractor()
        try:
            parser.feed(body)
        except Exception:  # malformed HTML — keep whatever was captured
            pass
        title = " ".join("".join(parser.title_parts).split())
        text = " ".join(" ".join(parser.text_parts).split())
        return title, text
    return "", " ".join(body.split())


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 1].rstrip()}…"


def fetch_url_content(
    url: str, max_chars: int = DEFAULT_MAX_CHARS
) -> Optional[dict]:
    """Fetch one URL → {url, title, text} (clipped), or None on any refusal/error.

    The whole fetch (DNS, every redirect hop, the body) ends within FETCH_TIMEOUT_SECONDS.
    """
    try:
        got = safe_fetch.fetch(url, max_bytes=MAX_DOWNLOAD_BYTES, timeout=FETCH_TIMEOUT_SECONDS,
                               accept_types=_ALLOWED_CONTENT_TYPES)
    except safe_fetch.FetchBlocked:
        logger.warning("URL fetch refused (scheme, port or non-public host)")
        return None
    except safe_fetch.FetchFailed:
        logger.warning("URL fetch failed (unreachable, refused encoding or too slow)")
        return None
    except Exception as e:  # noqa: BLE001 - fetching never blocks a turn or an ingest
        logger.warning("URL fetch failed (non-blocking): %s", type(e).__name__)
        return None
    return _page(got, max_chars)


def _page(got: "safe_fetch.Fetched", max_chars: int) -> Optional[dict]:
    """{url, title, text} from a fetched body, or None when it has no readable text."""
    if not got.matched:
        logger.warning("URL fetch refused (content-type %s)", got.content_type)
        return None
    title, text = _extract_text(got.content_type, _CONTROL.sub(" ", got.text()))
    text = _clip(text, max_chars)
    if not text:
        return None
    return {"url": got.url, "title": title or got.host, "text": text}


def fetch_pages(urls: List[str], max_chars: int = DEFAULT_MAX_CHARS) -> List[dict]:
    """Fetch up to MAX_FETCH_URLS urls side by side under one FETCH_TIMEOUT_SECONDS deadline;
    failures are silently dropped."""
    try:
        results = safe_fetch.fetch_many(urls[:MAX_FETCH_URLS], max_bytes=MAX_DOWNLOAD_BYTES,
                                        timeout=FETCH_TIMEOUT_SECONDS, accept_types=_ALLOWED_CONTENT_TYPES)
    except Exception as e:  # noqa: BLE001 - fetching never blocks a turn
        logger.warning("URL fetch failed (non-blocking): %s", type(e).__name__)
        return []
    pages: List[dict] = []
    for got in results:
        if isinstance(got, safe_fetch.Fetched):
            page = _page(got, max_chars)
            if page:
                pages.append(page)
    return pages


def format_page_snippets(pages: List[dict]) -> List[str]:
    """Label fetched pages like other context: [page N | title | url] text.

    Page text arrives already clipped (fetch_pages). Bracketed numbers in the title and text
    become (n), so a page cannot pose as one of this turn's passage numbers; the label stays.
    """
    return [
        f"[page {idx + 1} | {neutralise_markers(p.get('title') or 'untitled')} | "
        f"{p.get('url') or 'no-url'}] {neutralise_markers(p.get('text', ''))}"
        for idx, p in enumerate(pages)
    ]
