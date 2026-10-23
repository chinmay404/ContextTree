"""Unit tests for the webpage parser (app/agent/helpers/web_fetch.py).

No network, no DB: SSRF refusals trigger before any request is made, and
the success/redirect paths use a fake httpx client.
"""

from types import SimpleNamespace

import httpx
import pytest

from app.agent.helpers.web_fetch import (
    DEFAULT_MAX_CHARS,
    MAX_FETCH_URLS,
    _clip,
    _extract_text,
    extract_urls,
    fetch_url_content,
    format_page_snippets,
    host_is_public,
)


# ── URL extraction ──────────────────────────────────────────────


def test_extract_urls_finds_and_dedupes():
    text = (
        "see https://a.example/x and http://b.example/y, "
        "again https://a.example/x trailing https://c.example/z."
    )
    urls = extract_urls(text, limit=10)
    assert urls == [
        "https://a.example/x",
        "http://b.example/y",
        "https://c.example/z",
    ]


def test_extract_urls_caps_at_limit():
    text = " ".join(f"https://site{i}.example/p" for i in range(5))
    assert len(extract_urls(text)) == MAX_FETCH_URLS


def test_extract_urls_strips_trailing_punctuation():
    assert extract_urls("look: https://a.example/page.") == ["https://a.example/page"]


def test_extract_urls_empty_input():
    assert extract_urls("") == []
    assert extract_urls(None) == []


# ── SSRF guard ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",       # loopback
        "10.0.0.5",        # RFC1918
        "172.16.3.4",      # RFC1918
        "192.168.1.1",     # RFC1918
        "169.254.169.254", # cloud metadata / link-local
        "::1",             # v6 loopback
        "::ffff:127.0.0.1",# v4-mapped loopback
        "0.0.0.0",         # unspecified
        "100.64.0.1",      # shared address space (CGNAT), not global
        "100.100.100.200", # a cloud metadata address inside 100.64.0.0/10
        "localhost",
        "foo.localhost",
        "",
    ],
)
def test_host_is_public_refuses_internal(host):
    assert host_is_public(host) is False


def test_host_is_public_allows_public_ip_literal():
    assert host_is_public("8.8.8.8") is True


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/file",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "http://127.0.0.1/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://localhost:8000/api",
        "http://[::1]/x",
    ],
)
def test_fetch_refuses_bad_scheme_and_internal_hosts(url):
    # Guard fires before any HTTP request — no network involved.
    assert fetch_url_content(url) is None


# ── Text extraction + clipping ──────────────────────────────────


def test_extract_text_html_strips_scripts_and_captures_title():
    html = (
        "<html><head><title>My Page</title><style>.x{}</style></head>"
        "<body><script>evil()</script><h1>Hello</h1><p>World  now</p>"
        "<noscript>nope</noscript></body></html>"
    )
    title, text = _extract_text("text/html; charset=utf-8", html)
    assert title == "My Page"
    assert "Hello" in text and "World now" in text
    assert "evil" not in text and ".x{}" not in text and "nope" not in text


def test_extract_text_plain_passthrough():
    title, text = _extract_text("text/plain", "  hello\n\nworld  ")
    assert title == ""
    assert text == "hello world"


def test_clip_caps_length():
    clipped = _clip("a" * 10_000, DEFAULT_MAX_CHARS)
    assert len(clipped) <= DEFAULT_MAX_CHARS
    assert clipped.endswith("…")


# ── Fetch flow over a fake transport (the guarded fetch: app/agent/helpers/safe_fetch.py) ──


@pytest.fixture
def fake_httpx(monkeypatch):
    """routes: "http://<host><path>" -> httpx.Response factory; sent: every request."""
    state = SimpleNamespace(routes={}, sent=[])

    async def handle(self, request):
        state.sent.append(request)
        make = state.routes.get(f"{request.url.scheme}://{request.headers['host']}{request.url.path}")
        if make is None:
            return httpx.Response(404, request=request)
        return make(request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle)
    return state


async def _stream(body: bytes):
    # A streamed body, as a real transport gives it (bytes content would be pre-read and decoded).
    yield body


def respond(status=200, headers=None, body=b""):
    return lambda req: httpx.Response(status, headers=headers or {}, content=_stream(body), request=req)


def test_fetch_success_html(fake_httpx):
    fake_httpx.routes["http://8.8.8.8/page"] = respond(
        headers={"content-type": "text/html"},
        body=b"<html><head><title>T</title></head><body><p>Body text</p></body></html>",
    )
    page = fetch_url_content("http://8.8.8.8/page")
    assert page is not None
    assert page["title"] == "T"
    assert "Body text" in page["text"]
    assert fake_httpx.sent[0].headers["accept-encoding"] == "identity"


def test_fetch_refuses_wrong_content_type(fake_httpx):
    fake_httpx.routes["http://8.8.8.8/data"] = respond(headers={"content-type": "application/json"}, body=b"{}")
    assert fetch_url_content("http://8.8.8.8/data") is None


def test_fetch_follows_public_redirect(fake_httpx):
    fake_httpx.routes["http://8.8.8.8/a"] = respond(302, headers={"location": "http://9.9.9.9/b"})
    fake_httpx.routes["http://9.9.9.9/b"] = respond(headers={"content-type": "text/plain"}, body=b"dest")
    page = fetch_url_content("http://8.8.8.8/a")
    assert page is not None and page["text"] == "dest"
    assert page["url"] == "http://9.9.9.9/b"


def test_fetch_refuses_redirect_to_internal(fake_httpx):
    fake_httpx.routes["http://8.8.8.8/evil"] = respond(302, headers={"location": "http://169.254.169.254/latest/"})
    assert fetch_url_content("http://8.8.8.8/evil") is None
    assert [r.url.path for r in fake_httpx.sent] == ["/evil"]


def test_fetch_caps_output_chars(fake_httpx):
    fake_httpx.routes["http://8.8.8.8/big"] = respond(headers={"content-type": "text/plain"}, body=b"x" * 50_000)
    page = fetch_url_content("http://8.8.8.8/big")
    assert page is not None
    assert len(page["text"]) <= DEFAULT_MAX_CHARS


@pytest.mark.parametrize("encoding", ["br", "zstd"])
def test_fetch_refuses_brotli_and_zstd(fake_httpx, encoding):
    fake_httpx.routes["http://8.8.8.8/e"] = respond(
        headers={"content-type": "text/html", "content-encoding": encoding}, body=b"....")
    assert fetch_url_content("http://8.8.8.8/e") is None


def test_fetch_caps_a_gzip_bomb_on_decoded_bytes(fake_httpx, monkeypatch):
    import gzip

    from app.agent.helpers import web_fetch
    plain = b"<title>bomb</title>" + b"a" * (30 * 1024 * 1024)
    fake_httpx.routes["http://8.8.8.8/bomb"] = respond(
        headers={"content-type": "text/html", "content-encoding": "gzip"}, body=gzip.compress(plain, 6))
    seen = []
    real = web_fetch._extract_text
    monkeypatch.setattr(web_fetch, "_extract_text", lambda ctype, body: seen.append(len(body)) or real(ctype, body))
    page = fetch_url_content("http://8.8.8.8/bomb")
    assert page is not None and page["title"] == "bomb"
    assert seen and seen[0] <= web_fetch.MAX_DOWNLOAD_BYTES


def test_fetch_has_one_deadline_including_dns(monkeypatch):
    import time

    from app.agent.helpers import safe_fetch, web_fetch

    def hung(*a, **k):
        time.sleep(5)
        raise OSError("late")

    monkeypatch.setattr(safe_fetch, "_getaddrinfo", hung)
    monkeypatch.setattr(web_fetch, "FETCH_TIMEOUT_SECONDS", 1)
    started = time.monotonic()
    assert fetch_url_content("http://example.com/") is None
    assert time.monotonic() - started < 2.5


@pytest.mark.parametrize("url", ["http://8.8.8.8:8080/", "http://100.64.0.1/", "http://100.100.100.200/"])
def test_fetch_refuses_other_ports_and_shared_address_space(fake_httpx, url):
    assert fetch_url_content(url) is None
    assert fake_httpx.sent == []


def test_format_page_snippets_labels():
    lines = format_page_snippets(
        [{"title": "T", "url": "http://u", "text": "body"}]
    )
    assert lines == ["[page 1 | T | http://u] body"]


def test_format_page_snippets_neutralise_citation_markers():
    """A fetched page cannot pose as one of this turn's passage numbers (app-shell plan Task 9)."""
    [line] = format_page_snippets(
        [{"title": "Report [2]", "url": "https://example.test/p", "text": "see [3] and [1, 2]"}])
    assert line.startswith("[page 1 | ")
    assert line == "[page 1 | Report (2) | https://example.test/p] see (3) and (1, 2)"
