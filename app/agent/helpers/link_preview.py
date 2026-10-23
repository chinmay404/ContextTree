"""
Link preview for the board's paste box: title, site, description and image of a pasted URL
(contract amendment 2026-10-23, `GET /links/preview`). Nothing is stored.

SSRF guard (the same rules as POST /sources/link and the chat's page fetch, web_fetch.py):
- http/https only, and no credentials in the URL;
- the host must resolve only to public unicast addresses (no loopback, private, link-local
  incl. 169.254.169.254, reserved, multicast or unspecified; IPv4, IPv6 and v4-mapped);
- every redirect hop is checked again, at most 3 redirects;
- the connection goes to the address that was checked (pinned: the HTTP client does no DNS
  lookup of its own, TLS still verifies the host name via SNI), so DNS rebinding between the
  check and the connect cannot reach an internal address;
- proxy settings from the environment are ignored.

Budget: 4 s in total across every hop, at most 256 KB of HTML read; a non-HTML page returns
only its site. Errors never carry the upstream body or exception text.
"""
from __future__ import annotations

import ipaddress
import re
import socket
import time
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import urljoin

import httpx

from app.agent.helpers.web_fetch import _ip_is_public
from app.core.logger import logger

PREVIEW_TIMEOUT_SECONDS = 4
PREVIEW_MAX_BYTES = 256 * 1024
PREVIEW_MAX_REDIRECTS = 3
TITLE_CHARS = 300
DESCRIPTION_CHARS = 500
SITE_CHARS = 100
IMAGE_URL_CHARS = 2048
_HTML_TYPES = ("text/html", "application/xhtml")
_REDIRECTS = (301, 302, 303, 307, 308)
_USER_AGENT = "ContextTree/1.0 link preview (+https://contexttree.tech)"
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

# Seams for tests: name resolution and the clock.
_getaddrinfo = socket.getaddrinfo
_now = time.monotonic


class LinkBlocked(Exception):
    """The URL may not be fetched (scheme, credentials, or a non-public address)."""


class LinkUnreachable(Exception):
    """The page could not be read in time (DNS, network, status, redirects, budget)."""


def _public_address(host: str) -> str:
    """The first address of `host`, once every address it resolves to is public."""
    name = str(host or "").strip().lower().rstrip(".")
    if not name or name == "localhost" or name.endswith(".localhost"):
        raise LinkBlocked()
    try:
        infos = _getaddrinfo(name, None)
    except (socket.gaierror, UnicodeError, ValueError, OSError):
        raise LinkUnreachable() from None
    if not infos:
        raise LinkUnreachable()
    addresses = []
    for info in infos:
        try:
            ip = ipaddress.ip_address(str(info[4][0]).split("%")[0])
        except ValueError:
            raise LinkBlocked() from None
        if not _ip_is_public(ip):
            raise LinkBlocked()
        addresses.append(ip)
    return str(addresses[0])


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
    host = (host or "").lower()
    return host[4:] if host.startswith("www.") else host


def _image(page_url: str, value: str) -> Optional[str]:
    if not value:
        return None
    try:
        url = httpx.URL(urljoin(page_url, value.strip()))
    except Exception:  # noqa: BLE001 - an unparseable image is just no image
        return None
    if url.scheme not in ("http", "https") or len(str(url)) > IMAGE_URL_CHARS:
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


def _target(url: str) -> httpx.URL:
    try:
        parsed = httpx.URL(url)
    except Exception:  # noqa: BLE001
        raise LinkBlocked() from None
    if parsed.scheme not in ("http", "https") or not parsed.host or parsed.userinfo:
        raise LinkBlocked()
    return parsed


def fetch_link_preview(url: str) -> dict:
    """{title, site, description, image} for a public http(s) page.

    Raises LinkBlocked or LinkUnreachable; neither carries upstream text.
    """
    deadline = _now() + PREVIEW_TIMEOUT_SECONDS
    current = str(url or "").strip()
    for _hop in range(PREVIEW_MAX_REDIRECTS + 1):
        target = _target(current)
        host_ascii = target.raw_host.decode("ascii")  # IDNA form: what DNS and TLS see
        address = _public_address(host_ascii)
        remaining = deadline - _now()
        if remaining <= 0:
            raise LinkUnreachable()
        pinned = target.copy_with(host=address)
        headers = {"Host": target.netloc.decode("ascii"), "User-Agent": _USER_AGENT,
                   "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1"}
        extensions = {"sni_hostname": host_ascii} if target.scheme == "https" else {}
        try:
            with httpx.Client(timeout=httpx.Timeout(remaining), follow_redirects=False,
                              trust_env=False) as client:
                request = client.build_request("GET", pinned, headers=headers, extensions=extensions)
                response = client.send(request, stream=True)
                try:
                    if response.status_code in _REDIRECTS:
                        location = response.headers.get("location")
                        if not location:
                            raise LinkUnreachable()
                        current = urljoin(str(target), location)
                        continue  # the next hop is checked again at the top
                    if response.status_code != 200:
                        raise LinkUnreachable()
                    ctype = (response.headers.get("content-type") or "").lower()
                    if not any(t in ctype for t in _HTML_TYPES):
                        return {"title": None, "site": _site(target.host), "description": None,
                                "image": None}
                    raw = bytearray()
                    for chunk in response.iter_bytes():
                        raw += chunk
                        if len(raw) >= PREVIEW_MAX_BYTES or _now() >= deadline:
                            break
                    encoding = response.encoding or "utf-8"
                finally:
                    response.close()
        except (LinkBlocked, LinkUnreachable):
            raise
        except Exception as exc:  # noqa: BLE001 - network errors, timeouts, bad responses
            logger.info("Link preview could not read %s (%s)", target.host, type(exc).__name__)
            raise LinkUnreachable() from None
        html = bytes(raw[:PREVIEW_MAX_BYTES]).decode(encoding, errors="replace")
        return _parse(str(target), target.host, html)
    raise LinkUnreachable()  # more than PREVIEW_MAX_REDIRECTS redirects
