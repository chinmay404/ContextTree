"""
Link preview for the board's paste box: title, site, description and image of a pasted URL
(contract amendment 2026-10-23, `GET /links/preview`). Nothing is stored.

The fetch goes through `safe_fetch` (SSRF guard, pinned connection, default ports only, a hard
4 s wall-clock deadline over DNS and every hop, identity encoding with a 256 KB cap on decoded
bytes). A non-HTML page returns only its site. Errors never carry upstream text.
"""
from __future__ import annotations

import asyncio
import ipaddress
import re
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import urljoin

import httpx

from app.agent.helpers import safe_fetch

PREVIEW_TIMEOUT_SECONDS = 4
PREVIEW_MAX_BYTES = 256 * 1024
TITLE_CHARS = 300
DESCRIPTION_CHARS = 500
SITE_CHARS = 100
IMAGE_URL_CHARS = 2048
_HTML_TYPES = ("text/html", "application/xhtml")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_PRIVATE_SUFFIXES = (".localhost", ".local", ".internal", ".intranet", ".lan", ".home.arpa", ".corp")

LinkBlocked = safe_fetch.FetchBlocked
LinkUnreachable = safe_fetch.FetchFailed


def _clean(value, limit: int) -> Optional[str]:
    text = " ".join(_CONTROL.sub(" ", str(value or "")).split())
    if not text:
        return None
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class _MetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self._in_title = False
        self.title_parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        elif tag == "meta":
            a = {k.lower(): (v or "") for k, v in attrs if k}
            key = (a.get("property") or a.get("name") or "").strip().lower()
            if key and "content" in a and key not in self.meta:
                self.meta[key] = a["content"]

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title_parts.append(data)


def _site(host: str) -> str:
    host = (host or "").lower().rstrip(".")
    return host[4:] if host.startswith("www.") else host


def _image(page_url: str, value: str) -> Optional[str]:
    """An http(s) og:image on a public-looking host name; IP literals and private names are dropped
    (the browser would load it, so it must not point inside a network)."""
    if not value:
        return None
    try:
        url = httpx.URL(urljoin(page_url, value.strip()))
    except Exception:  # noqa: BLE001 - an unparseable image is just no image
        return None
    if url.scheme not in ("http", "https") or len(str(url)) > IMAGE_URL_CHARS or url.userinfo:
        return None
    host = url.host.lower().rstrip(".")
    try:
        ipaddress.ip_address(host)
        return None  # an IP literal
    except ValueError:
        pass
    if (not host or "." not in host or host == "localhost" or host.endswith(_PRIVATE_SUFFIXES)
            or not safe_fetch._port_allowed(url.scheme, url.port)):
        return None
    return str(url)


def _parse(page_url: str, host: str, html: str) -> dict:
    parser = _MetaParser()
    try:
        parser.feed(html)
    except Exception:  # noqa: BLE001 - malformed HTML: keep what was read
        pass
    meta = parser.meta
    return {
        "title": _clean(meta.get("og:title") or "".join(parser.title_parts), TITLE_CHARS),
        "site": _clean(meta.get("og:site_name"), SITE_CHARS) or _site(host),
        "description": _clean(meta.get("og:description") or meta.get("description"), DESCRIPTION_CHARS),
        "image": _image(page_url, meta.get("og:image", "")),
    }


async def afetch_link_preview(url: str) -> dict:
    """{title, site, description, image} for a public http(s) page.

    Raises LinkBlocked or LinkUnreachable; neither carries upstream text.
    """
    got = await safe_fetch.afetch(url, max_bytes=PREVIEW_MAX_BYTES, timeout=PREVIEW_TIMEOUT_SECONDS,
                                  accept_types=_HTML_TYPES)
    if not got.matched:
        return {"title": None, "site": _site(got.host), "description": None, "image": None}
    return _parse(got.url, got.host, got.text())


def fetch_link_preview(url: str) -> dict:
    """Blocking form of `afetch_link_preview` (scripts and tests)."""
    return asyncio.run(afetch_link_preview(url))
