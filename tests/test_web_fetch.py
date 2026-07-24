"""Unit tests for the webpage parser (app/agent/helpers/web_fetch.py).

No network, no DB: SSRF refusals trigger before any request is made, and
the success/redirect paths use a fake httpx client.
"""

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


# ── Fetch flow with a fake httpx client ─────────────────────────


class FakeResponse:
    def __init__(self, status_code=200, headers=None, body=b"", encoding="utf-8"):
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body
        self.encoding = encoding

    def iter_bytes(self):
        yield self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeClient:
    routes: dict = {}

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def stream(self, method, url):
        return FakeClient.routes[url]


@pytest.fixture
def fake_httpx(monkeypatch):
    FakeClient.routes = {}
    monkeypatch.setattr(httpx, "Client", FakeClient)
    return FakeClient


def test_fetch_success_html(fake_httpx):
    fake_httpx.routes["http://8.8.8.8/page"] = FakeResponse(
        headers={"content-type": "text/html"},
        body=b"<html><head><title>T</title></head><body><p>Body text</p></body></html>",
    )
    page = fetch_url_content("http://8.8.8.8/page")
    assert page is not None
    assert page["title"] == "T"
    assert "Body text" in page["text"]


def test_fetch_refuses_wrong_content_type(fake_httpx):
    fake_httpx.routes["http://8.8.8.8/data"] = FakeResponse(
        headers={"content-type": "application/json"}, body=b"{}"
    )
    assert fetch_url_content("http://8.8.8.8/data") is None


def test_fetch_follows_public_redirect(fake_httpx):
    fake_httpx.routes["http://8.8.8.8/a"] = FakeResponse(
        status_code=302, headers={"location": "http://9.9.9.9/b"}
    )
    fake_httpx.routes["http://9.9.9.9/b"] = FakeResponse(
        headers={"content-type": "text/plain"}, body=b"dest"
    )
    page = fetch_url_content("http://8.8.8.8/a")
    assert page is not None and page["text"] == "dest"
    assert page["url"] == "http://9.9.9.9/b"


def test_fetch_refuses_redirect_to_internal(fake_httpx):
    fake_httpx.routes["http://8.8.8.8/evil"] = FakeResponse(
        status_code=302, headers={"location": "http://169.254.169.254/latest/"}
    )
    assert fetch_url_content("http://8.8.8.8/evil") is None


def test_fetch_caps_output_chars(fake_httpx):
    fake_httpx.routes["http://8.8.8.8/big"] = FakeResponse(
        headers={"content-type": "text/plain"}, body=b"x" * 50_000
    )
    page = fetch_url_content("http://8.8.8.8/big")
    assert page is not None
    assert len(page["text"]) <= DEFAULT_MAX_CHARS


def test_format_page_snippets_labels():
    lines = format_page_snippets(
        [{"title": "T", "url": "http://u", "text": "body"}]
    )
    assert lines == ["[page 1 | T | http://u] body"]
