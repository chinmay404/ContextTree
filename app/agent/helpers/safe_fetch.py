"""
One guarded HTTP GET for URLs that users supply: the paste-box link preview, link sources
(ingest) and the chat's page fetch all go through `afetch` / `fetch`.

SSRF guard:
- http/https only, the scheme's default port only (80 / 443), no credentials in the URL;
- every address the host resolves to must be public and globally routable
  (`ip_is_public`: no loopback, private, link-local incl. 169.254.169.254, shared/CGNAT
  100.64.0.0/10, reserved, multicast or unspecified; IPv4, IPv6 and v4-mapped);
- every redirect hop is checked again, at most 3 redirects;
- the connection goes to the address that was checked (pinned: the HTTP client does no DNS
  lookup of its own; Host and TLS SNI keep the name, so the certificate is still verified);
- proxy settings from the environment are ignored.

Resource limits:
- one hard wall-clock deadline over the whole fetch: DNS, connect, TLS, headers, body and
  every hop (`anyio.fail_after`; a hung DNS lookup is abandoned, not waited for);
- `Accept-Encoding: identity`, the body is read raw, and only gzip/deflate are decoded here,
  with the cap applied to DECODED bytes; any other Content-Encoding (br, zstd, ...) is refused,
  so a small compressed body cannot expand into gigabytes.

Errors are FetchBlocked (the URL may not be fetched) or FetchFailed (it could not be read in
time); neither carries upstream text.
"""
from __future__ import annotations

import asyncio
import codecs
import ipaddress
import socket
import zlib
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin

import anyio
import httpx

from app.core.logger import logger

MAX_REDIRECTS = 3
DEFAULT_PORTS = {"http": 80, "https": 443}
_REDIRECTS = (301, 302, 303, 307, 308)
_USER_AGENT = "ContextTree/1.0 (+https://contexttree.tech)"

# Seams for tests (a loopback server stands in for a public host only inside tests).
_getaddrinfo = socket.getaddrinfo


def _port_allowed(scheme: str, port: Optional[int]) -> bool:
    return port is None or port == DEFAULT_PORTS.get(scheme)


class FetchBlocked(Exception):
    """The URL may not be fetched (scheme, port, credentials or a non-public address)."""


class FetchFailed(Exception):
    """The page could not be read (DNS, network, status, redirects, encoding or the deadline)."""


def ip_is_public(ip) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not (
        ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
        or ip.is_multicast or ip.is_unspecified
    )


@dataclass
class Fetched:
    url: str              # the final URL (after redirects), with the original host name
    host: str             # its host name
    content_type: str     # lower-cased; "" when missing
    matched: bool         # the content type was one the caller accepts (else body is empty)
    body: bytes           # decoded body, at most max_bytes
    encoding: str         # a codec Python knows (the declared charset, or utf-8)

    def text(self) -> str:
        return self.body.decode(self.encoding, errors="replace")


def _target(url: str) -> httpx.URL:
    try:
        target = httpx.URL(str(url or "").strip())
    except Exception:  # noqa: BLE001
        raise FetchBlocked() from None
    if target.scheme not in DEFAULT_PORTS or not target.host or target.userinfo:
        raise FetchBlocked()
    if not _port_allowed(target.scheme, target.port):
        raise FetchBlocked()
    return target


async def _public_address(host: str) -> str:
    name = host.strip().lower().rstrip(".")
    if not name or name == "localhost" or name.endswith(".localhost"):
        raise FetchBlocked()
    try:
        infos = await anyio.to_thread.run_sync(_getaddrinfo, name, None, abandon_on_cancel=True)
    except (socket.gaierror, UnicodeError, ValueError, OSError):
        raise FetchFailed() from None
    if not infos:
        raise FetchFailed()
    addresses = []
    for info in infos:
        try:
            ip = ipaddress.ip_address(str(info[4][0]).split("%")[0])
        except ValueError:
            raise FetchBlocked() from None
        if not ip_is_public(ip):
            raise FetchBlocked()
        addresses.append(ip)
    return str(addresses[0])


def _codec(name: Optional[str]) -> str:
    try:
        return codecs.lookup(name or "utf-8").name
    except LookupError:
        return "utf-8"


async def _read_capped(response: httpx.Response, max_bytes: int) -> bytes:
    """The body, decoded here (identity, gzip or deflate only), never more than max_bytes."""
    encoding = (response.headers.get("content-encoding") or "identity").strip().lower()
    if encoding in ("", "identity"):
        decoder = None
    elif encoding in ("gzip", "x-gzip"):
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    elif encoding == "deflate":
        decoder = zlib.decompressobj(zlib.MAX_WBITS)
    else:
        raise FetchFailed()  # br, zstd, compress, stacked encodings: refused, never decoded
    out = bytearray()
    raw_read = 0
    async for chunk in response.aiter_raw():
        raw_read += len(chunk)
        room = max_bytes - len(out)
        if decoder is None:
            out += chunk[:room]
        else:
            try:
                out += decoder.decompress(chunk, room)  # never more than `room` bytes out
            except zlib.error:
                if not out:
                    raise FetchFailed() from None
                break
        if len(out) >= max_bytes or raw_read >= max_bytes:
            break
    return bytes(out[:max_bytes])


async def _fetch(url: str, max_bytes: int, accept_types: tuple) -> Fetched:
    current = url
    for _hop in range(MAX_REDIRECTS + 1):
        target = _target(current)
        host = target.raw_host.decode("ascii").rstrip(".")  # IDNA form, no trailing dot
        address = await _public_address(host)
        pinned = target.copy_with(host=address)
        headers = {
            "Host": host,  # default ports only, so no port in Host
            "User-Agent": _USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
            "Accept-Encoding": "identity",
        }
        extensions = {"sni_hostname": host} if target.scheme == "https" else {}
        async with httpx.AsyncClient(follow_redirects=False, trust_env=False, timeout=None) as client:
            request = client.build_request("GET", pinned, headers=headers, extensions=extensions)
            response = await client.send(request, stream=True)
            try:
                if response.status_code in _REDIRECTS:
                    location = response.headers.get("location")
                    if not location:
                        raise FetchFailed()
                    current = urljoin(str(target), location)
                    continue  # the next hop is checked again at the top
                if response.status_code != 200:
                    raise FetchFailed()
                ctype = (response.headers.get("content-type") or "").lower()
                charset = _codec(response.charset_encoding)
                if not any(t in ctype for t in accept_types):
                    return Fetched(str(target), host, ctype, False, b"", charset)
                body = await _read_capped(response, max_bytes)
                return Fetched(str(target), host, ctype, True, body, charset)
            finally:
                await response.aclose()
    raise FetchFailed()  # more than MAX_REDIRECTS redirects


async def afetch(url: str, *, max_bytes: int, timeout: float, accept_types: tuple) -> Fetched:
    """GET one user-supplied URL under the guard; the whole fetch ends within `timeout` s."""
    try:
        with anyio.fail_after(timeout):
            return await _fetch(url, max_bytes, accept_types)
    except (FetchBlocked, FetchFailed):
        raise
    except TimeoutError:
        raise FetchFailed() from None
    except Exception as exc:  # noqa: BLE001 - network errors, bad responses
        logger.info("Guarded fetch failed (%s)", type(exc).__name__)
        raise FetchFailed() from None


async def afetch_many(urls: list, *, max_bytes: int, timeout: float, accept_types: tuple) -> list:
    """`afetch` for several URLs side by side under ONE deadline; each slot is a Fetched or an
    exception (FetchBlocked / FetchFailed). A URL still running at the deadline is FetchFailed."""
    results: list = [FetchFailed() for _ in urls]

    async def one(i: int, url: str) -> None:
        try:
            results[i] = await _fetch(url, max_bytes, accept_types)
        except (FetchBlocked, FetchFailed) as exc:
            results[i] = exc
        except Exception as exc:  # noqa: BLE001 - network errors, bad responses
            logger.info("Guarded fetch failed (%s)", type(exc).__name__)
            results[i] = FetchFailed()

    with anyio.move_on_after(timeout):
        async with anyio.create_task_group() as tg:
            for i, url in enumerate(urls):
                tg.start_soon(one, i, url)
    return results


# Sync callers (ingest, the chat's page fetch) run the fetch on its own event loop in a worker.
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="safe-fetch")


def fetch(url: str, *, max_bytes: int, timeout: float, accept_types: tuple) -> Fetched:
    """Blocking `afetch`, usable from any thread (also one that runs an event loop)."""
    future = _executor.submit(
        asyncio.run, afetch(url, max_bytes=max_bytes, timeout=timeout, accept_types=accept_types))
    try:
        return future.result(timeout=timeout + 1)
    except FutureTimeout:
        raise FetchFailed() from None


def fetch_many(urls: list, *, max_bytes: int, timeout: float, accept_types: tuple) -> list:
    """Blocking `afetch_many`."""
    if not urls:
        return []
    future = _executor.submit(
        asyncio.run, afetch_many(list(urls), max_bytes=max_bytes, timeout=timeout,
                                 accept_types=accept_types))
    try:
        return future.result(timeout=timeout + 1)
    except FutureTimeout:
        return [FetchFailed() for _ in urls]
