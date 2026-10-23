"""GET /links/preview (contract amendment 2026-10-23, polish-2 P11) and the guarded fetch under it.

No real network. Most tests patch DNS (`safe_fetch._getaddrinfo`) and the async HTTP transport and
record every request that would leave the machine. The deadline and decompression tests use a real
loopback server; only inside those tests is the guard relaxed to let 127.0.0.1 stand in for a
public host.
"""
import asyncio
import gzip
import inspect
import ipaddress
import socket
import threading
import time
import tracemalloc
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest

from app.agent.helpers import link_preview as lp
from app.agent.helpers import safe_fetch
from app.api.v1.endpoints import sources
from tests.v3_support import JWT_SECRET, api, db, make_board, new_id, requires_db, user  # noqa: F401

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

    monkeypatch.setattr(safe_fetch, "_getaddrinfo", getaddrinfo)

    async def handle(self, request):
        state.sent.append(request)
        make = state.routes.get((request.headers.get("host", ""), request.url.path))
        if make is None:
            return httpx.Response(404, content=b"not here", request=request)
        return make(request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle)
    return state


async def _stream(body: bytes):
    # A streamed body, as a real transport gives it (bytes content would be pre-read and decoded).
    yield body


def html_page(body=HTML, ctype="text/html; charset=utf-8", headers=None):
    return lambda req: httpx.Response(200, headers={"content-type": ctype, **(headers or {})},
                                      content=_stream(body), request=req)


def redirect(to):
    return lambda req: httpx.Response(302, headers={"location": to}, request=req)


# ── parsing ──────────────────────────────────────────────────────────────────

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


def test_long_and_hostile_values_are_clipped_and_cleaned(net):
    body = ('<html><head><meta property="og:title" content="' + "T" * 1000 + '">'
            '<meta property="og:description" content="line\x00one\nline \x1b[31mtwo">'
            '<meta property="og:image" content="javascript:alert(1)"></head></html>').encode()
    net.routes[("example.com", "/x")] = html_page(body)
    out = lp.fetch_link_preview("https://example.com/x")
    assert len(out["title"]) <= lp.TITLE_CHARS
    assert out["description"] == "line one line [31mtwo" and out["image"] is None


@pytest.mark.parametrize("image", [
    "http://169.254.169.254/x.png", "http://10.0.0.1/x.png", "http://[::1]/x.png", "http://8.8.8.8/x.png",
    "http://intranet/x.png", "https://cdn.local/x.png", "https://svc.internal/x.png",
    "https://localhost/x.png", "https://cdn.example.com:8443/x.png", "https://u:p@cdn.example.com/x.png",
])
def test_an_image_on_an_ip_or_private_name_is_dropped(net, image):
    net.routes[("example.com", "/i")] = html_page(f'<meta property="og:image" content="{image}">'.encode())
    assert lp.fetch_link_preview("https://example.com/i")["image"] is None


def test_a_bogus_charset_still_decodes(net):
    net.routes[("example.com", "/c")] = html_page(b"<title>t</title>", ctype="text/html; charset=x-bogus-1")
    assert lp.fetch_link_preview("https://example.com/c")["title"] == "t"


# ── the guard ────────────────────────────────────────────────────────────────

def test_the_connection_is_pinned_and_asks_for_no_encoding(net):
    net.routes[("example.com", "/post")] = html_page()
    lp.fetch_link_preview("https://example.com./post")  # a trailing dot is dropped for Host and SNI
    [req] = net.sent
    assert req.url.host == "93.184.216.34"  # no second DNS lookup by the HTTP client
    assert req.headers["host"] == "example.com"
    assert req.extensions.get("sni_hostname") == "example.com"  # TLS still verifies example.com
    assert req.headers["accept-encoding"] == "identity"


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/", "http://localhost/", "http://10.0.0.5/", "http://192.168.1.1/",
    "http://172.16.0.1/", "http://169.254.169.254/latest/meta-data/", "http://[::1]/",
    "http://[fe80::1]/", "http://[::ffff:127.0.0.1]/", "http://0.0.0.0/", "http://2130706433/",
    "http://100.64.0.1/", "http://100.100.100.200/latest/meta-data/",
    "ftp://example.com/a", "file:///etc/passwd", "javascript:alert(1)", "http://user:pw@example.com/",
    "http://example.com:8080/", "https://example.com:80/", "http://example.com:443/",
])
def test_blocked_urls_never_leave_the_machine(net, url):
    with pytest.raises(lp.LinkBlocked):
        lp.fetch_link_preview(url)
    assert net.sent == []


def test_the_default_port_written_out_is_allowed(net):
    net.routes[("example.com", "/post")] = html_page()
    assert lp.fetch_link_preview("https://example.com:443/post")["title"] == "OG & Title"


def test_a_name_that_resolves_to_a_private_address_is_blocked(net):
    net.dns["internal.example.com"] = ["93.184.216.34", "10.1.2.3"]  # any private answer blocks
    net.dns["cgnat.example.com"] = ["100.64.12.1"]
    for url in ("https://internal.example.com/", "https://cgnat.example.com/"):
        with pytest.raises(lp.LinkBlocked):
            lp.fetch_link_preview(url)
    assert net.sent == []


@pytest.mark.parametrize("target, error", [
    ("http://169.254.169.254/latest/meta-data/", lp.LinkBlocked), ("http://[::1]/x", lp.LinkBlocked),
    ("http://internal.example.com/", lp.LinkBlocked), ("http://example.com:8080/", lp.LinkBlocked),
    ("file:///etc/passwd", lp.LinkBlocked),
    # httpx itself rejects a javascript: Location while reading the 302, so it is "unreachable"
    ("javascript:alert(1)", lp.LinkUnreachable),
])
def test_a_redirect_to_a_blocked_target_is_never_requested(net, target, error):
    net.dns["internal.example.com"] = ["192.168.0.10"]
    net.routes[("example.com", "/go")] = redirect(target)
    with pytest.raises(error):
        lp.fetch_link_preview("https://example.com/go")
    assert [r.url.path for r in net.sent] == ["/go"]  # the blocked hop is never requested


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


def test_the_client_ignores_proxy_settings_from_the_environment(net, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://10.0.0.9:3128")
    monkeypatch.setenv("HTTP_PROXY", "http://10.0.0.9:3128")
    net.routes[("example.com", "/post")] = html_page()
    assert lp.fetch_link_preview("https://example.com/post")["title"] == "OG & Title"
    assert net.sent[0].url.host == "93.184.216.34"


# ── size: 256 KB of DECODED bytes, encodings ─────────────────────────────────

def test_reads_at_most_256_kb(net):
    pulled = []

    async def chunks():
        yield HTML[: HTML.index(b"</head>")]
        for _ in range(100):  # 6.4 MB of padding the reader must not pull
            pulled.append(1)
            yield b"x" * 65536

    net.routes[("example.com", "/big")] = lambda req: httpx.Response(
        200, headers={"content-type": "text/html"}, content=chunks(), request=req)
    assert lp.fetch_link_preview("https://example.com/big")["title"] == "OG & Title"
    assert len(pulled) * 65536 <= lp.PREVIEW_MAX_BYTES + 65536
    assert lp.PREVIEW_MAX_BYTES == 256 * 1024


@pytest.mark.parametrize("encoding", ["br", "zstd", "compress", "gzip, br", "identity, gzip"])
def test_unsupported_content_encodings_are_refused(net, encoding):
    net.routes[("example.com", "/e")] = html_page(b"\x28\xb5\x2f\xfd....", headers={"content-encoding": encoding})
    with pytest.raises(lp.LinkUnreachable):
        lp.fetch_link_preview("https://example.com/e")


@pytest.mark.parametrize("encoding", ["gzip", "deflate"])
def test_gzip_and_deflate_bombs_are_capped_on_decoded_bytes(net, encoding):
    plain = HTML + b"\0" * (40 * 1024 * 1024)
    body = gzip.compress(plain, 6) if encoding == "gzip" else zlib.compress(plain, 6)
    assert len(body) < 100 * 1024
    net.routes[("example.com", "/bomb")] = html_page(body, headers={"content-encoding": encoding})

    got = asyncio.run(safe_fetch.afetch("https://example.com/bomb", max_bytes=lp.PREVIEW_MAX_BYTES,
                                        timeout=4, accept_types=("text/html",)))
    assert len(got.body) == lp.PREVIEW_MAX_BYTES and got.body.startswith(b"<html>")
    assert lp.fetch_link_preview("https://example.com/bomb")["title"] == "OG & Title"


def test_non_html_returns_the_site_without_reading_the_body(net):
    pulled = []

    async def chunks():
        pulled.append(1)
        yield b"%PDF-1.4"

    net.routes[("example.com", "/doc.pdf")] = lambda req: httpx.Response(
        200, headers={"content-type": "application/pdf"}, content=chunks(), request=req)
    out = lp.fetch_link_preview("https://example.com/doc.pdf")
    assert out == {"title": None, "site": "example.com", "description": None, "image": None}
    assert pulled == []


# ── a real loopback server: the wall-clock deadline and real compression ─────

class _Slow(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    bodies: dict = {}

    def log_message(self, *a):
        pass

    def do_GET(self):
        try:
            if self.path == "/slowhead":  # headers trickle in over 15 s
                self.wfile.write(b"HTTP/1.1 200 OK\r\nX-A: ")
                for _ in range(10):
                    self.wfile.flush()
                    time.sleep(1.5)
                    self.wfile.write(b"a")
                self.wfile.write(b"\r\nContent-Type: text/html\r\nContent-Length: 20\r\n\r\n<title>late</title> ")
            elif self.path == "/slowbody":  # the body trickles in over 17 s
                self.wfile.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n"
                                 b"Content-Length: 100000\r\n\r\n<title>t")
                for _ in range(5):
                    self.wfile.flush()
                    time.sleep(3.5)
                    self.wfile.write(b"a")
            else:
                body, encoding = self.bodies[self.path]
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Encoding", encoding)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            self.wfile.flush()
        except OSError:
            pass


@pytest.fixture
def loopback(monkeypatch):
    """A local server standing in for a public host (the guard is relaxed only in this test)."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Slow)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    real_public = safe_fetch.ip_is_public

    def gai(name, *a, **k):
        if str(name).endswith("probe.test"):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))]
        raise socket.gaierror("offline test")

    monkeypatch.setattr(safe_fetch, "_getaddrinfo", gai)
    monkeypatch.setattr(safe_fetch, "ip_is_public", lambda ip: str(ip) == "127.0.0.1" or real_public(ip))
    monkeypatch.setattr(safe_fetch, "_port_allowed", lambda scheme, port: True)
    try:
        yield f"http://site.probe.test:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _timed(fn):
    started = time.monotonic()
    try:
        return fn(), time.monotonic() - started
    except Exception as exc:  # noqa: BLE001
        return exc, time.monotonic() - started


@pytest.mark.parametrize("path", ["/slowhead", "/slowbody"])
def test_a_trickling_server_is_cut_off_at_four_seconds(loopback, path):
    out, took = _timed(lambda: lp.fetch_link_preview(loopback + path))
    assert isinstance(out, lp.LinkUnreachable)
    assert took <= 4.6, took


def test_dns_is_inside_the_deadline(loopback, monkeypatch):
    def hung(*a, **k):
        time.sleep(8)
        raise socket.gaierror("late")

    monkeypatch.setattr(safe_fetch, "_getaddrinfo", hung)
    out, took = _timed(lambda: lp.fetch_link_preview(loopback + "/x"))
    assert isinstance(out, lp.LinkUnreachable) and took <= 4.6, took


def test_real_compression_bombs_stay_small(loopback):
    import brotli
    import zstandard

    plain = b"<html><head><title>x</title></head>" + bytes(60 * 1024 * 1024)
    _Slow.bodies = {"/gz": (gzip.compress(plain, 6), "gzip"),
                    "/br": (brotli.compress(plain, quality=5), "br"),
                    "/zst": (zstandard.ZstdCompressor(level=3).compress(plain), "zstd")}
    tracemalloc.start()
    try:
        gz, took = _timed(lambda: lp.fetch_link_preview(loopback + "/gz"))
        br, _ = _timed(lambda: lp.fetch_link_preview(loopback + "/br"))
        zs, _ = _timed(lambda: lp.fetch_link_preview(loopback + "/zst"))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert gz["title"] == "x" and took < 4.6
    assert isinstance(br, lp.LinkUnreachable) and isinstance(zs, lp.LinkUnreachable)
    assert peak < 16 * 1024 * 1024, peak  # decoded at most 256 KB at a time, never the 60 MB


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
def test_route_youtube_oembed_is_cut_off_at_four_seconds(api, net, monkeypatch):
    def hung(url, timeout=5):
        time.sleep(8)
        return {"title": "late"}

    monkeypatch.setattr(sources, "fetch_youtube_oembed", hung)
    started = time.monotonic()
    r = api.get("/links/preview", params={"url": "https://youtu.be/abcdefghijk"})
    assert r.status_code == 200 and r.json()["title"] is None
    assert time.monotonic() - started <= 4.6


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
def test_route_allows_two_previews_in_flight_per_user(api, net, monkeypatch):
    net.routes[("example.com", "/post")] = html_page()
    monkeypatch.setitem(sources._previews_in_flight, api.email, 2)
    r = api.get("/links/preview", params={"url": "https://example.com/post"})
    assert r.status_code == 429 and r.json()["error"] == "busy" and net.sent == []
    monkeypatch.setitem(sources._previews_in_flight, api.email, 1)
    assert api.get("/links/preview", params={"url": "https://example.com/post"}).status_code == 200
    assert sources._previews_in_flight[api.email] == 1  # its own slot was given back
    sources._previews_in_flight.pop(api.email)
    net.routes[("example.com", "/gone")] = lambda req: httpx.Response(500, request=req)
    assert api.get("/links/preview", params={"url": "https://example.com/gone"}).status_code == 422
    assert api.email not in sources._previews_in_flight  # released after an error too


@requires_db
def test_route_needs_a_signed_in_user(net, monkeypatch):
    from fastapi.testclient import TestClient

    from app.api.server import app
    from app.core.config import settings
    monkeypatch.setattr(settings, "BACKEND_JWT_SECRET", JWT_SECRET)
    r = TestClient(app).get("/api/v1/links/preview", params={"url": "https://example.com/"})
    assert r.status_code == 401 and net.sent == []


@requires_db
def test_link_sources_take_only_the_default_port(api, monkeypatch):
    monkeypatch.setattr(sources, "host_is_public", lambda h: True)
    b = make_board(api)["id"]
    for url in ("https://example.com:8443/x", "http://example.com:22/", "http://example.com:99999/"):
        r = api.post(f"/boards/{b}/sources/link", json={"id": new_id("c"), "url": url, "x": 0, "y": 0})
        assert r.status_code == 400 and r.json()["error"] == "invalid", url
    r = api.post(f"/boards/{b}/sources/link",
                 json={"id": new_id("c"), "url": "https://example.com:443/x", "x": 0, "y": 0})
    assert r.status_code == 202


def test_route_is_rate_limited_per_user():
    from app.api.limiter import limiter
    fn = sources.api_link_preview
    [limit] = limiter._route_limits[f"{fn.__module__}.{fn.__name__}"]
    assert str(limit.limit) == "60 per 1 minute"
    assert limit.key_func is sources.user_key


def test_route_is_async_so_a_slow_page_never_holds_a_worker_thread():
    assert inspect.iscoroutinefunction(sources.api_link_preview)
