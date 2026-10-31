"""Reading a whole site from one link: scope, robots, slices, limits. No network."""
import pytest

from app.agent import crawl, ingest
from app.agent.crawl import in_scope, links_in, scope_of
from app.agent.helpers import safe_fetch
from app.api.v1.endpoints import sources
from tests.v3_support import (  # noqa: F401
    FakeLLM, api, db, make_board, new_id, requires_db, seed_board, seed_card, unit_emb, user,
)


# ── which links are followed ─────────────────────────────────────


def test_scope_is_the_host_and_the_folder_of_the_link():
    assert scope_of("https://docs.example.com/guide/intro") == ("docs.example.com", "/guide/")
    assert scope_of("https://docs.example.com/guide/") == ("docs.example.com", "/guide/")
    assert scope_of("https://example.com") == ("example.com", "/")


@pytest.mark.parametrize("url,ok", [
    ("https://x.dev/guide/a", True),
    ("https://x.dev/guide/", True),
    ("https://x.dev/blog/a", False),
    ("https://other.dev/guide/a", False),
    ("https://sub.x.dev/guide/a", False),
    ("https://x.dev/guide/logo.png", False),
    ("https://x.dev/guide/paper.pdf", False),
    ("ftp://x.dev/guide/a", False),
])
def test_only_pages_in_scope_are_followed(url, ok):
    assert in_scope(url, "x.dev", "/guide/") is ok


def test_links_are_absolute_and_lose_their_fragment_and_query():
    html = ('<a href="/guide/a#top">a</a><a href="b?page=2">b</a><a href="mailto:x@x.dev">m</a>'
            '<a href="javascript:void(0)">j</a><a>none</a><a href="/guide/a">again</a>')
    assert links_in(html, "https://x.dev/guide/index") == ["https://x.dev/guide/a", "https://x.dev/guide/b"]


def test_malformed_html_gives_the_links_found_so_far():
    assert links_in('<a href="/one">x</a><a href="http://[bad">y</a>', "https://x.dev/") == ["https://x.dev/one"]


# ── a site, end to end ───────────────────────────────────────────

ROOT = "https://x.dev/guide/"


def page(n, links=()):
    anchors = "".join(f'<a href="{u}">go</a>' for u in links)
    body = "".join(f"<p>Page {n} line {i} talks about something long enough to count as real text here.</p>"
                   for i in range(8))
    return f"<html><head><title>Page {n}</title></head><body><article>{body}</article>{anchors}</body></html>"


class Site:
    """A stand-in for safe_fetch.fetch: {url: html}, plus an optional robots.txt."""

    def __init__(self, pages, robots=None):
        self.pages, self.robots, self.asked = pages, robots, []

    def __call__(self, url, *, max_bytes, timeout, accept_types):
        self.asked.append(url)
        if url.endswith("/robots.txt"):
            if self.robots is None:
                raise safe_fetch.FetchFailed()
            return safe_fetch.Fetched(url, "x.dev", "text/plain", True, self.robots.encode(), "utf-8")
        got = self.pages.get(url)
        if got is None:
            raise safe_fetch.FetchFailed()
        if isinstance(got, Exception):
            raise got
        if isinstance(got, safe_fetch.Fetched):
            return got
        return safe_fetch.Fetched(url, "x.dev", "text/html", True, got.encode(), "utf-8")


@pytest.fixture
def stubs(monkeypatch):
    monkeypatch.setattr(ingest, "get_embeddings", lambda texts: [unit_emb(i % 768) for i, _ in enumerate(texts)])
    monkeypatch.setattr(ingest, "get_embedding", lambda t: unit_emb(0))
    monkeypatch.setattr(ingest, "get_llm", lambda *a, **k: FakeLLM("An overview."))
    monkeypatch.setattr(crawl, "POLITE_DELAY_SECONDS", 0)


def _site(monkeypatch, pages, robots=None):
    site = Site(pages, robots)
    monkeypatch.setattr(safe_fetch, "fetch", site)
    return site


def _card(db, user, url=ROOT):
    board = seed_board(db, user)
    return seed_card(db, user, board, kind="source", source_type="web", status="processing",
                     data={"url": url, "title": "x.dev", "meta": "x.dev", "crawled": {"pages": 0, "done": False}})


def _row(db, card):
    with db.cursor() as cur:
        cur.execute("SELECT status, data FROM nodes WHERE id = %s", (card,))
        return cur.fetchone()


def _files(db, card):
    with db.cursor() as cur:
        cur.execute("SELECT file_name FROM external_files WHERE node_id = %s ORDER BY created_at, id", (card,))
        return [r[0] for r in cur.fetchall()]


def _locs(db, card):
    with db.cursor() as cur:
        cur.execute("SELECT fc.metadata->>'loc' FROM file_chunks fc JOIN external_files ef ON ef.id = fc.file_id "
                    "WHERE ef.node_id = %s", (card,))
        return [r[0] for r in cur.fetchall()]


@requires_db
def test_a_site_is_read_breadth_first_into_one_card(db, user, stubs, monkeypatch):
    card = _card(db, user)
    _site(monkeypatch, {
        ROOT: page(1, ["/guide/a", "/guide/b", "/blog/off"]),
        "https://x.dev/guide/a": page(2, ["/guide/c"]),
        "https://x.dev/guide/b": page(3),
        "https://x.dev/guide/c": page(4),
    })
    crawl.ingest_crawl_slice(card, user)
    status, data = _row(db, card)
    assert status == "ready"
    assert data["crawled"] == {"pages": 4, "done": True}
    assert data["meta"] == "x.dev · 4 pages" and data["title"] == "Page 1"
    assert data["summary"] == "An overview." and data["excerpts"]
    assert "crawl" not in data
    assert _files(db, card) == ["Page 1", "Page 2", "Page 3", "Page 4"]
    locs = _locs(db, card)
    assert len(locs) == len(set(locs)), "locations must be unique across pages"


@requires_db
def test_every_fetch_goes_through_the_guard(db, user, stubs, monkeypatch):
    """A link to a blocked address is refused by safe_fetch, and the crawl carries on."""
    card = _card(db, user)
    _site(monkeypatch, {ROOT: page(1, ["/guide/internal", "/guide/ok"]),
                        "https://x.dev/guide/internal": safe_fetch.FetchBlocked(),
                        "https://x.dev/guide/ok": page(2)})
    crawl.ingest_crawl_slice(card, user)
    status, data = _row(db, card)
    assert status == "ready" and data["crawled"]["pages"] == 2
    assert data["meta"] == "x.dev · 2 pages · 1 skipped"


@requires_db
def test_robots_txt_is_respected_for_followed_links(db, user, stubs, monkeypatch):
    card = _card(db, user)
    site = _site(monkeypatch, {ROOT: page(1, ["/guide/private", "/guide/open"]),
                               "https://x.dev/guide/private": page(2),
                               "https://x.dev/guide/open": page(3)},
                 robots="User-agent: *\nDisallow: /guide/private\n")
    crawl.ingest_crawl_slice(card, user)
    assert "https://x.dev/guide/private" not in site.asked
    assert _row(db, card)[1]["crawled"]["pages"] == 2


@requires_db
def test_depth_and_page_limits_hold(db, user, stubs, monkeypatch):
    card = _card(db, user)
    monkeypatch.setattr(crawl, "MAX_PAGES", 3)
    links = [f"/guide/p{i}" for i in range(10)]
    pages = {ROOT: page(0, links)}
    pages.update({f"https://x.dev/guide/p{i}": page(i + 1) for i in range(10)})
    site = _site(monkeypatch, pages)
    crawl.ingest_crawl_slice(card, user)
    assert _row(db, card)[1]["crawled"]["pages"] == 3
    assert len([u for u in site.asked if "robots" not in u]) == 3


@requires_db
def test_links_past_the_depth_limit_are_not_followed(db, user, stubs, monkeypatch):
    card = _card(db, user)
    site = _site(monkeypatch, {
        ROOT: page(1, ["/guide/d1"]),
        "https://x.dev/guide/d1": page(2, ["/guide/d2"]),
        "https://x.dev/guide/d2": page(3, ["/guide/d3"]),
        "https://x.dev/guide/d3": page(4),
    })
    crawl.ingest_crawl_slice(card, user)
    assert "https://x.dev/guide/d3" not in site.asked
    assert _row(db, card)[1]["crawled"]["pages"] == 3


@requires_db
def test_one_long_page_cannot_use_up_the_whole_budget(db, user, stubs, monkeypatch):
    card = _card(db, user)
    long_body = "".join(f"<p>{'word ' * 150}number {i}</p>" for i in range(200))
    long_page = f"<html><head><title>Long</title></head><body><article>{long_body}</article><a href='/guide/b'>b</a></body></html>"
    _site(monkeypatch, {ROOT: long_page, "https://x.dev/guide/b": page(2)})
    crawl.ingest_crawl_slice(card, user)
    locs = _locs(db, card)
    assert len(locs) <= crawl.CHUNKS_PER_PAGE + 10
    assert _row(db, card)[1]["crawled"]["pages"] == 2


@requires_db
def test_the_total_chunk_budget_ends_the_crawl_early(db, user, stubs, monkeypatch):
    card = _card(db, user)
    monkeypatch.setattr(crawl, "MAX_CHUNKS", 1)
    _site(monkeypatch, {ROOT: page(1, ["/guide/a"]), "https://x.dev/guide/a": page(2)})
    crawl.ingest_crawl_slice(card, user)
    assert _row(db, card)[1]["crawled"]["pages"] == 1


@requires_db
def test_copies_of_a_page_are_kept_once(db, user, stubs, monkeypatch):
    card = _card(db, user)
    _site(monkeypatch, {ROOT: page(1, ["/guide/copy"]), "https://x.dev/guide/copy": page(1)})
    crawl.ingest_crawl_slice(card, user)
    assert _row(db, card)[1]["crawled"]["pages"] == 1


@requires_db
def test_a_redirect_off_the_site_is_skipped(db, user, stubs, monkeypatch):
    card = _card(db, user)
    off = safe_fetch.Fetched("https://evil.example/x", "evil.example", "text/html", True, page(9).encode(), "utf-8")
    _site(monkeypatch, {ROOT: page(1, ["/guide/away"]), "https://x.dev/guide/away": off})
    crawl.ingest_crawl_slice(card, user)
    data = _row(db, card)[1]
    assert data["crawled"]["pages"] == 1 and data["meta"] == "x.dev · 1 page · 1 skipped"


@requires_db
def test_a_first_redirect_moves_the_scope_to_where_the_site_really_is(db, user, stubs, monkeypatch):
    card = _card(db, user, url="https://x.dev/")
    moved = safe_fetch.Fetched("https://www.x.dev/guide/", "www.x.dev", "text/html", True,
                               page(1, ["/guide/a", "/other"]).encode(), "utf-8")
    _site(monkeypatch, {"https://x.dev/": moved, "https://www.x.dev/guide/a": page(2)})
    crawl.ingest_crawl_slice(card, user)
    assert _row(db, card)[1]["crawled"]["pages"] == 2


@requires_db
def test_a_page_with_nothing_below_it_reads_its_neighbours(db, user, stubs, monkeypatch):
    """Docs sites end most pages in a slash, so /guide/intro/ looks like a folder of its own."""
    card = _card(db, user, url="https://x.dev/guide/intro/")
    _site(monkeypatch, {"https://x.dev/guide/intro/": page(1, ["/guide/setup/", "/blog/off"]),
                        "https://x.dev/guide/setup/": page(2)})
    crawl.ingest_crawl_slice(card, user)
    assert _row(db, card)[1]["crawled"]["pages"] == 2


@requires_db
def test_a_folder_with_pages_below_it_stays_inside_itself(db, user, stubs, monkeypatch):
    card = _card(db, user)
    site = _site(monkeypatch, {ROOT: page(1, ["/guide/a", "/other"]), "https://x.dev/guide/a": page(2),
                               "https://x.dev/other": page(3)})
    crawl.ingest_crawl_slice(card, user)
    assert "https://x.dev/other" not in site.asked


@requires_db
def test_a_crawl_of_a_link_that_is_a_pdf_reads_the_one_document(db, user, stubs, monkeypatch):
    card = _card(db, user, url="https://x.dev/paper.pdf")
    not_html = safe_fetch.Fetched("https://x.dev/paper.pdf", "x.dev", "application/pdf", False, b"", "utf-8")
    _site(monkeypatch, {"https://x.dev/paper.pdf": not_html})
    read = []
    monkeypatch.setattr(ingest, "ingest_web_source", lambda c, u: read.append((c, u)))
    crawl.ingest_crawl_slice(card, user)
    assert read == [(card, user)]
    assert "crawled" not in _row(db, card)[1]


@requires_db
def test_a_site_whose_first_page_cannot_be_read_fails(db, user, stubs, monkeypatch):
    card = _card(db, user)
    _site(monkeypatch, {})
    crawl.ingest_crawl_slice(card, user)
    status, data = _row(db, card)
    assert status == "failed" and data["error"] == "Couldn't read this page" and "crawl" not in data


@requires_db
def test_a_slice_that_runs_out_of_time_saves_its_place_and_the_next_one_goes_on(db, user, stubs, monkeypatch):
    card = _card(db, user)
    site = _site(monkeypatch, {ROOT: page(1, ["/guide/a", "/guide/b"]),
                               "https://x.dev/guide/a": page(2), "https://x.dev/guide/b": page(3)})
    monkeypatch.setattr(crawl, "SLICE_SECONDS", 0)
    crawl.ingest_crawl_slice(card, user)
    status, data = _row(db, card)
    assert status == "processing" and data["crawled"] == {"pages": 1, "done": False}
    assert data["crawl"]["queue"], "the place is saved"
    assert "ingestLease" not in data, "the next kick can claim it"

    monkeypatch.setattr(crawl, "SLICE_SECONDS", 100)
    crawl.ingest_crawl_slice(card, user)
    status, data = _row(db, card)
    assert status == "ready" and data["crawled"] == {"pages": 3, "done": True}
    assert len([u for u in site.asked if u == ROOT]) == 1, "a page is never fetched twice"


@requires_db
def test_too_many_slices_finish_with_what_was_read(db, user, stubs, monkeypatch):
    card = _card(db, user)
    _site(monkeypatch, {ROOT: page(1, ["/guide/a"]), "https://x.dev/guide/a": page(2)})
    monkeypatch.setattr(crawl, "SLICE_SECONDS", 0)
    monkeypatch.setattr(crawl, "MAX_SLICES", 1)
    crawl.ingest_crawl_slice(card, user)
    status, data = _row(db, card)
    assert status == "ready" and data["crawled"] == {"pages": 1, "done": True}


@requires_db
def test_a_daily_limit_stops_a_crawl_with_what_it_has(db, user, stubs, monkeypatch):
    card = _card(db, user)
    monkeypatch.setattr(crawl, "PAGES_PER_DAY", 1)
    _site(monkeypatch, {ROOT: page(1, ["/guide/a"]), "https://x.dev/guide/a": page(2)})
    crawl.ingest_crawl_slice(card, user)
    status, data = _row(db, card)
    assert status == "ready" and data["crawled"]["pages"] == 1
    assert "daily limit" in data["meta"]


@requires_db
def test_a_daily_limit_with_nothing_read_yet_fails_and_says_why(db, user, stubs, monkeypatch):
    first = _card(db, user)
    monkeypatch.setattr(crawl, "PAGES_PER_DAY", 1)
    _site(monkeypatch, {ROOT: page(1)})
    crawl.ingest_crawl_slice(first, user)
    second = _card(db, user)
    crawl.ingest_crawl_slice(second, user)
    status, data = _row(db, second)
    assert status == "failed" and "limit" in data["error"]


@requires_db
def test_embedding_failure_fails_the_card(db, user, stubs, monkeypatch):
    card = _card(db, user)
    monkeypatch.setattr(ingest, "get_embeddings", lambda texts: [[0.0] * 768 for _ in texts])
    _site(monkeypatch, {ROOT: page(1)})
    crawl.ingest_crawl_slice(card, user)
    status, data = _row(db, card)
    assert status == "failed" and data["error"] == "Failed to generate embeddings"


@requires_db
def test_a_card_deleted_mid_crawl_stops_it(db, user, stubs, monkeypatch):
    card = _card(db, user)

    class Deleting(Site):
        def __call__(self, url, **kw):
            if url.endswith("/guide/a"):
                with db.cursor() as cur:
                    cur.execute("UPDATE nodes SET deleted_at = now() WHERE id = %s", (card,))
            return super().__call__(url, **kw)

    site = Deleting({ROOT: page(1, ["/guide/a", "/guide/b"]), "https://x.dev/guide/a": page(2),
                     "https://x.dev/guide/b": page(3)})
    monkeypatch.setattr(safe_fetch, "fetch", site)
    crawl.ingest_crawl_slice(card, user)
    assert "https://x.dev/guide/b" not in site.asked


# ── the routes ───────────────────────────────────────────────────


@requires_db
def test_a_link_asks_for_a_crawl_and_the_ingest_route_runs_slices(api, db, stubs, monkeypatch):
    b = make_board(api)["id"]
    monkeypatch.setattr(sources, "host_is_public", lambda h: True)
    _site(monkeypatch, {ROOT: page(1, ["/guide/a"]), "https://x.dev/guide/a": page(2)})
    r = api.post(f"/boards/{b}/sources/link", json={"id": new_id("c"), "url": ROOT, "x": 0, "y": 0, "crawl": True})
    assert r.status_code == 202 and r.json()["data"]["crawled"] == {"pages": 0, "done": False}
    monkeypatch.setattr(crawl, "SLICE_SECONDS", 0)
    first = api.post(f"/boards/{b}/cards/{r.json()['id']}/ingest").json()
    assert first["status"] == "processing" and first["data"]["crawled"]["pages"] == 1
    assert "crawl" not in first["data"], "the crawl's working state stays in the engine"
    monkeypatch.setattr(crawl, "SLICE_SECONDS", 100)
    done = api.post(f"/boards/{b}/cards/{r.json()['id']}/ingest").json()
    assert done["status"] == "ready" and done["data"]["crawled"] == {"pages": 2, "done": True}
    passages = api.get(f"/files/{r.json()['id']}/passages").json()["passages"]
    assert {p["loc"].split(" § ")[0] for p in passages} == {"/guide/", "/guide/a"}


@requires_db
def test_a_plain_link_is_not_crawled(api, monkeypatch):
    b = make_board(api)["id"]
    monkeypatch.setattr(sources, "host_is_public", lambda h: True)
    r = api.post(f"/boards/{b}/sources/link", json={"id": new_id("c"), "url": ROOT, "x": 0, "y": 0})
    assert "crawled" not in r.json()["data"]


@requires_db
def test_a_video_link_ignores_crawl(api, monkeypatch):
    b = make_board(api)["id"]
    monkeypatch.setattr(sources, "fetch_youtube_oembed", lambda u: None)
    r = api.post(f"/boards/{b}/sources/link",
                 json={"id": new_id("c"), "url": "https://youtu.be/abcdefghijk", "x": 0, "y": 0, "crawl": True})
    assert "crawled" not in r.json()["data"]


@requires_db
def test_the_crawl_state_cannot_be_written_by_the_client(api, db):
    b = make_board(api)["id"]
    card = seed_card(db, api.email, b, kind="source", source_type="note", status="ready", data={"title": "T"})
    r = api.patch(f"/boards/{b}/cards/{card}", json={"data": {"crawl": {"queue": [["http://localhost/", 0]]}}})
    assert r.status_code == 200 and "crawl" not in r.json()["data"]
    assert "crawl" not in _row(db, card)[1]
