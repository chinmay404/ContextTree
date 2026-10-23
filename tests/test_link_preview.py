"""GET /links/preview (contract amendment 2026-10-23, polish-2 P11). No real network: DNS and the
HTTP transport are patched; every request that would leave the machine is recorded."""
import ipaddress
import socket
from types import SimpleNamespace

import httpx
import pytest

from app.agent.helpers import link_preview as lp
from app.api.v1.endpoints import sources
from tests.v3_support import api, db, requires_db, user  # noqa: F401

HTML = (b"<html><head><title>Plain title</title>"
        b'<meta property="og:title" content="OG &amp; Title">'
        b'<meta property="og:description" content="What the page is about.">'
        b'<meta property="og:image" content="/img/card.png">'
        b'<meta property="og:site_name" content="Example Site">'
        b"</head><body>hello</body></html>")


def _addr(ip: str):
    fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
    return (fam, socket.SOCK_STREAM, 6, "", (ip, 443))


@pytest.fixture
def net(monkeypatch):
    """dns: host -> [ips]; routes: (host, path) -> response factory; sent: every request."""
    state = SimpleNamespace(dns={"example.com": ["93.184.216.34"]}, routes={}, sent=[])

    def getaddrinfo(host, *a, **k):
        name = str(host).lower().rstrip(".")
        if name.isdigit():  # like the real resolver: "2130706433" is 127.0.0.1
            return [_addr(str(ipaddress.ip_address(int(name))))]
        try:
            ipaddress.ip_address(name.strip("[]"))
            return [_addr(name.strip("[]"))]
        except ValueError:
            pass
        if name not in state.dns:
            raise socket.gaierror("no such host")
        return [_addr(ip) for ip in state.dns[name]]

    monkeypatch.setattr(lp, "_getaddrinfo", getaddrinfo)

    def handle(self, request):
        state.sent.append(request)
        host = request.headers.get("host", "").split(":")[0]
        make = state.routes.get((host, request.url.path))
        if make is None:
            return httpx.Response(404, content=b"not here", request=request)
        return make(request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle)
    return state


def html_page(body=HTML, ctype="text/html; charset=utf-8"):
    return lambda req: httpx.Response(200, headers={"content-type": ctype}, content=body, request=req)


def redirect(to):
    return lambda req: httpx.Response(302, headers={"location": to}, request=req)


# ── the fetch layer ──────────────────────────────────────────────────────────

def test_parses_og_tags_and_resolves_the_image(net):
    net.routes[("example.com", "/post")] = html_page()
    out = lp.fetch_link_preview("https://example.com/post")
    assert out == {"title": "OG & Title", "site": "Example Site", "description": "What the page is about.",
                   "image": "https://example.com/img/card.png"}


def test_falls_back_to_title_tag_meta_description_and_host(net):
    body = (b"<html><head><title>  Just a   title </title>"
            b'<meta name="description" content="Plain description"></head></html>')
    net.dns["www.example.org"] = ["93.184.216.35"]
    net.routes[("www.example.org", "/")] = html_page(body)
    out = lp.fetch_link_preview("http://www.example.org/")
    assert out == {"title": "Just a title", "site": "example.org", "description": "Plain description",
                   "image": None}


def test_the_connection_is_pinned_to_the_checked_address(net):
    net.routes[("example.com", "/post")] = html_page()
    lp.fetch_link_preview("https://example.com/post")
    [req] = net.sent
    assert req.url.host == "93.184.216.34"  # no second DNS lookup by the HTTP client
    assert req.headers["host"] == "example.com"
    assert req.extensions.get("sni_hostname") == "example.com"  # TLS still verifies example.com


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/", "http://localhost:8000/", "http://10.0.0.5/", "http://192.168.1.1/",
    "http://172.16.0.1/", "http://169.254.169.254/latest/meta-data/", "http://[::1]/",
    "http://[fe80::1]/", "http://[::ffff:127.0.0.1]/", "http://0.0.0.0/", "http://2130706433/",
    "ftp://example.com/a", "file:///etc/passwd", "javascript:alert(1)", "http://user:pw@example.com/",
])
def test_blocked_addresses_never_leave_the_machine(net, url):
    with pytest.raises(lp.LinkBlocked):
        lp.fetch_link_preview(url)
    assert net.sent == []


def test_a_name_that_resolves_to_a_private_address_is_blocked(net):
    net.dns["internal.example.com"] = ["93.184.216.34", "10.1.2.3"]  # any private answer blocks
    with pytest.raises(lp.LinkBlocked):
        lp.fetch_link_preview("https://internal.example.com/")
    assert net.sent == []


@pytest.mark.parametrize("target", ["http://169.254.169.254/latest/meta-data/", "http://[::1]:8080/x",
                                    "http://internal.example.com/"])
def test_a_redirect_to_a_private_address_is_blocked(net, target):
    net.dns["internal.example.com"] = ["192.168.0.10"]
    net.routes[("example.com", "/go")] = redirect(target)
    with pytest.raises(lp.LinkBlocked):
        lp.fetch_link_preview("https://example.com/go")
    assert [r.url.path for r in net.sent] == ["/go"]  # the private hop is never requested


def test_three_redirects_are_followed_and_a_fourth_is_unreachable(net):
    for i in range(4):
        net.routes[("example.com", f"/r{i}")] = redirect(f"/r{i + 1}")
    net.routes[("example.com", "/r3")] = html_page()
    assert lp.fetch_link_preview("https://example.com/r0")["title"] == "OG & Title"
    net.routes[("example.com", "/r3")] = redirect("/r4")
    net.routes[("example.com", "/r4")] = html_page()
    net.sent.clear()
    with pytest.raises(lp.LinkUnreachable):
        lp.fetch_link_preview("https://example.com/r0")
    assert len(net.sent) == 4


def test_unknown_host_error_status_and_timeouts_are_unreachable(net):
    with pytest.raises(lp.LinkUnreachable):
        lp.fetch_link_preview("https://no-such-host.example/")
    net.routes[("example.com", "/gone")] = lambda req: httpx.Response(500, content=b"boom", request=req)
    with pytest.raises(lp.LinkUnreachable):
        lp.fetch_link_preview("https://example.com/gone")

    def slow(req):
        raise httpx.ReadTimeout("timed out", request=req)

    net.routes[("example.com", "/slow")] = slow
    with pytest.raises(lp.LinkUnreachable):
        lp.fetch_link_preview("https://example.com/slow")


def test_the_four_second_budget_covers_every_hop(net, monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(lp, "_now", lambda: clock["t"])

    def slow_redirect(req):
        clock["t"] += 4.5
        return httpx.Response(302, headers={"location": "/next"}, request=req)

    net.routes[("example.com", "/first")] = slow_redirect
    net.routes[("example.com", "/next")] = html_page()
    with pytest.raises(lp.LinkUnreachable):
        lp.fetch_link_preview("https://example.com/first")
    assert [r.url.path for r in net.sent] == ["/first"]
    assert lp.PREVIEW_TIMEOUT_SECONDS == 4


def test_reads_at_most_256_kb(net):
    pulled = []

    def chunks():
        yield HTML[: HTML.index(b"</head>")]
        for _ in range(100):  # 6.4 MB of padding the reader must not pull
            pulled.append(1)
            yield b"x" * 65536

    net.routes[("example.com", "/big")] = lambda req: httpx.Response(
        200, headers={"content-type": "text/html"}, content=chunks(), request=req)
    out = lp.fetch_link_preview("https://example.com/big")
    assert out["title"] == "OG & Title"
    assert len(pulled) * 65536 <= lp.PREVIEW_MAX_BYTES + 65536
    assert lp.PREVIEW_MAX_BYTES == 256 * 1024


def test_non_html_returns_the_site_without_reading_the_body(net):
    pulled = []

    def chunks():
        pulled.append(1)
        yield b"%PDF-1.4"

    net.routes[("example.com", "/doc.pdf")] = lambda req: httpx.Response(
        200, headers={"content-type": "application/pdf"}, content=chunks(), request=req)
    out = lp.fetch_link_preview("https://example.com/doc.pdf")
    assert out == {"title": None, "site": "example.com", "description": None, "image": None}
    assert pulled == []


def test_long_and_hostile_values_are_clipped_and_cleaned(net):
    body = ('<html><head><meta property="og:title" content="' + "T" * 1000 + '">'
            '<meta property="og:description" content="line\x00one\nline two">'
            '<meta property="og:image" content="javascript:alert(1)"></head></html>').encode()
    net.routes[("example.com", "/x")] = html_page(body)
    out = lp.fetch_link_preview("https://example.com/x")
    assert len(out["title"]) <= lp.TITLE_CHARS
    assert out["description"] == "line one line two" and out["image"] is None


def test_the_client_ignores_proxy_settings_from_the_environment(net, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://10.0.0.9:3128")
    monkeypatch.setenv("HTTP_PROXY", "http://10.0.0.9:3128")
    net.routes[("example.com", "/post")] = html_page()
    assert lp.fetch_link_preview("https://example.com/post")["title"] == "OG & Title"
    assert net.sent[0].url.host == "93.184.216.34"


# ── the route ────────────────────────────────────────────────────────────────

@requires_db
def test_route_returns_the_preview(api, net):
    net.routes[("example.com", "/post")] = html_page()
    r = api.get("/links/preview", params={"url": "https://example.com/post"})
    assert r.status_code == 200, r.text
    assert r.json() == {"url": "https://example.com/post", "kind": "web", "title": "OG & Title",
                        "site": "Example Site", "description": "What the page is about.",
                        "image": "https://example.com/img/card.png"}


@requires_db
def test_route_youtube_uses_oembed(api, net, monkeypatch):
    seen = []

    def oembed(url, timeout=5):
        seen.append((url, timeout))
        return {"title": "A talk", "thumbnail_url": "https://i.ytimg.com/vi/abcdefghijk/hqdefault.jpg"}

    monkeypatch.setattr(sources, "fetch_youtube_oembed", oembed)
    url = "https://www.youtube.com/watch?v=abcdefghijk"
    r = api.get("/links/preview", params={"url": url})
    assert r.status_code == 200
    assert r.json() == {"url": url, "kind": "video", "title": "A talk", "site": "YouTube",
                        "description": None, "image": "https://i.ytimg.com/vi/abcdefghijk/hqdefault.jpg"}
    assert seen == [(url, 4)] and net.sent == []


@requires_db
def test_route_errors_carry_no_upstream_text(api, net):
    r = api.get("/links/preview", params={"url": "http://169.254.169.254/latest/meta-data/"})
    assert r.status_code == 400 and r.json()["error"] == "invalid"
    net.routes[("example.com", "/gone")] = lambda req: httpx.Response(
        500, content=b"SECRET UPSTREAM BODY", request=req)
    r = api.get("/links/preview", params={"url": "https://example.com/gone"})
    assert r.status_code == 422 and r.json()["error"] == "unreachable"
    assert set(r.json()) == {"error", "detail"} and "SECRET" not in r.text and "500" not in r.text
    for bad in ("", "not a url", "x" * 3000):
        r = api.get("/links/preview", params={"url": bad})
        assert r.status_code == 400 and r.json()["error"] == "invalid", bad
    r = api.get("/links/preview")
    assert r.status_code == 400 and r.json()["error"] == "invalid"


@requires_db
def test_route_needs_a_signed_in_user(net):
    from fastapi.testclient import TestClient
    from app.api.server import app
    r = TestClient(app).get("/api/v1/links/preview", params={"url": "https://example.com/"})
    assert r.status_code == 401 and net.sent == []


def test_route_is_rate_limited_per_user():
    from app.api.limiter import limiter
    fn = sources.api_link_preview
    [limit] = limiter._route_limits[f"{fn.__module__}.{fn.__name__}"]
    assert str(limit.limit) == "60 per 1 minute"
    assert limit.key_func is sources.user_key


def test_route_runs_in_the_threadpool():
    import inspect
    assert not inspect.iscoroutinefunction(sources.api_link_preview)
