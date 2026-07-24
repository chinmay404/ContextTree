"""
Webpage parser: fetch user-pasted URLs into <EXTERNAL_CONTEXT>.

Security (SSRF guard): only http/https; every hop of every redirect is
re-validated — the host must resolve exclusively to public unicast IPs.
Loopback, RFC1918 private ranges, link-local (incl. the 169.254.169.254
cloud metadata endpoint), reserved, multicast and unspecified addresses
are all refused, for IPv4, IPv6 and v4-mapped-v6. Responses are streamed
with a hard byte cap and only text-ish content types are parsed.
(Known limitation: the classic DNS-rebinding TOCTOU between our resolve
and the client's — mitigated by re-checking every redirect hop and by the
short timeout; full IP pinning is a later hardening step.)

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
from urllib.parse import urljoin, urlparse

from app.core.logger import logger

MAX_FETCH_URLS = 2
MAX_REDIRECTS = 3
FETCH_TIMEOUT_SECONDS = 8
MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024  # 2 MB
DEFAULT_MAX_CHARS = 3500
_ALLOWED_CONTENT_TYPES = ("text/html", "application/xhtml", "text/plain")
_URL_RE = re.compile(r"https?://[^\s<>\"'\)\]\}]+", re.IGNORECASE)
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


def _ip_is_public(ip: ipaddress._BaseAddress) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


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
    """Fetch one URL → {url, title, text} (clipped), or None on any refusal/error."""
    try:
        import httpx
    except ImportError:
        logger.warning("httpx not installed — URL fetch unavailable")
        return None

    current = str(url or "").strip()
    try:
        for _ in range(MAX_REDIRECTS + 1):
            parsed = urlparse(current)
            if parsed.scheme not in ("http", "https"):
                logger.warning("URL fetch refused (scheme): %s", current)
                return None
            if not host_is_public(parsed.hostname or ""):
                logger.warning("URL fetch refused (non-public host): %s", current)
                return None

            with httpx.Client(
                timeout=FETCH_TIMEOUT_SECONDS,
                follow_redirects=False,
                headers={"User-Agent": "ContextTree/1.0 (+https://contexttree.tech)"},
            ) as client:
                with client.stream("GET", current) as resp:
                    if resp.status_code in (301, 302, 303, 307, 308):
                        location = resp.headers.get("location")
                        if not location:
                            return None
                        current = urljoin(current, location)
                        continue  # next hop re-validated at loop top
                    if resp.status_code != 200:
                        logger.warning(
                            "URL fetch got %s for %s", resp.status_code, current
                        )
                        return None
                    ctype = (resp.headers.get("content-type") or "").lower()
                    if not any(t in ctype for t in _ALLOWED_CONTENT_TYPES):
                        logger.warning(
                            "URL fetch refused (content-type %s): %s", ctype, current
                        )
                        return None
                    raw = b""
                    for chunk in resp.iter_bytes():
                        raw += chunk
                        if len(raw) >= MAX_DOWNLOAD_BYTES:
                            break
                    encoding = resp.encoding or "utf-8"

            body = raw[:MAX_DOWNLOAD_BYTES].decode(encoding, errors="replace")
            title, text = _extract_text(ctype, body)
            text = _clip(text, max_chars)
            if not text:
                return None
            return {"url": current, "title": title or (parsed.hostname or ""), "text": text}

        logger.warning("URL fetch: too many redirects for %s", url)
        return None
    except Exception as e:
        logger.warning(f"URL fetch failed (non-blocking) for {url}: {e}")
        return None


def fetch_pages(urls: List[str], max_chars: int = DEFAULT_MAX_CHARS) -> List[dict]:
    """Fetch up to MAX_FETCH_URLS urls; failures are silently dropped."""
    pages: List[dict] = []
    for u in urls[:MAX_FETCH_URLS]:
        page = fetch_url_content(u, max_chars=max_chars)
        if page:
            pages.append(page)
    return pages


def format_page_snippets(pages: List[dict]) -> List[str]:
    """Label fetched pages like other context: [page N | title | url] text."""
    return [
        f"[page {idx + 1} | {p.get('title') or 'untitled'} | {p.get('url') or 'no-url'}] "
        f"{p.get('text', '')}"
        for idx, p in enumerate(pages)
    ]
