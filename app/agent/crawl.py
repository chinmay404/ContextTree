"""Read a whole site from one link: the pages it links to, into one source card.

Every request goes through `safe_fetch` (never a crawling library, which would open its own
connections past the guard). The crawl stays on the link's host and folder, goes two links deep
and reads at most 30 pages, one at a time, and honours robots.txt for the pages it follows; the
page the user pasted is read like any other pasted link.

A crawl can outlast one Vercel request, so it runs in slices: POST .../ingest reads for up to
SLICE_SECONDS, saves its place in the card's engine-only `crawl` data, gives up the lease and
returns the card still `processing`; the client calls again. `crawled` {pages, done} is the part
the client sees. Each page is a row in external_files, so chat search and /passages need nothing new.
"""
from __future__ import annotations

import hashlib
import json
import time
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.agent import ingest
from app.agent.helpers import safe_fetch
from app.agent.helpers.web_fetch import (
    _ALLOWED_CONTENT_TYPES, FETCH_TIMEOUT_SECONDS, MAX_DOWNLOAD_BYTES, _page,
)
from app.agent.store.board_store import store
from app.agent.utils.text import clip_text
from app.core.logger import logger

MAX_PAGES = 30
MAX_DEPTH = 2
MAX_CHUNKS = ingest.MAX_EMBED_CHUNKS  # the same bound as one source
CHUNKS_PER_PAGE = 20  # so the first long pages cannot use up the whole budget
PAGE_CHARS = 20_000
MAX_QUEUE = 200
PAGES_PER_DAY = 150  # per user: each page spends embedding quota on the engine's key
SLICE_SECONDS = 150  # well inside the 300 s a request may run
MAX_SLICES = 12  # a crawl always ends, even if the client keeps calling
POLITE_DELAY_SECONDS = 0.5
MAX_ROBOTS_DELAY_SECONDS = 5
ROBOTS_MAX_BYTES = 64 * 1024
ROBOTS_TIMEOUT_SECONDS = 4
_ROBOTS_AGENT = "ContextTree"
_NOT_PAGES = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".css", ".js", ".json", ".xml", ".pdf",
    ".zip", ".gz", ".tar", ".mp3", ".mp4", ".mov", ".webm", ".woff", ".woff2", ".ttf", ".eot",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".csv", ".rss", ".atom",
)
_LIMIT_ERROR = "You've reached today's limit for reading whole sites. Try again tomorrow."
_READ_ERROR = "Couldn't read this page"


def scope_of(url: str) -> tuple[str, str]:
    """(host, folder) a crawl from this URL stays inside."""
    u = urlparse(url)
    path = u.path or "/"
    return (u.hostname or "").lower(), path[: path.rfind("/") + 1] or "/"


def parent_folder(prefix: str) -> str:
    return prefix.rstrip("/").rpartition("/")[0] + "/"


def in_scope(url: str, host: str, prefix: str) -> bool:
    try:
        u = urlparse(url)
        name = (u.hostname or "").lower()
    except ValueError:
        return False
    path = u.path or "/"
    return (u.scheme in ("http", "https") and name == host
            and (path.startswith(prefix) or path == prefix.rstrip("/"))
            and not path.lower().endswith(_NOT_PAGES))


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def links_in(html: str, base: str) -> list[str]:
    """The page's links as absolute http(s) URLs, once each, without fragment or query (a query
    is how calendars and filters make endless pages)."""
    parser = _Links()
    try:
        parser.feed(html)
    except Exception:  # noqa: BLE001 - malformed HTML: keep what was found
        pass
    out: list[str] = []
    for href in parser.hrefs:
        try:
            url = urldefrag(urljoin(base, href)).url
            parsed = urlparse(url)
            url = parsed._replace(query="").geturl()
        except ValueError:
            continue
        if parsed.scheme in ("http", "https") and url not in out:
            out.append(url)
    return out


def _robots(scheme: str, host: str) -> Optional[RobotFileParser]:
    """The site's rules; None (everything allowed) when it has none that can be read."""
    try:
        got = safe_fetch.fetch(f"{scheme}://{host}/robots.txt", max_bytes=ROBOTS_MAX_BYTES,
                               timeout=ROBOTS_TIMEOUT_SECONDS, accept_types=("text/plain",))
    except (safe_fetch.FetchBlocked, safe_fetch.FetchFailed):
        return None
    if not got.matched:
        return None
    rules = RobotFileParser()
    rules.parse(got.text().splitlines())
    return rules


def _pages_today(user_email: str) -> int:
    with store.tx() as cur:
        cur.execute(
            "SELECT count(*) AS n FROM external_files WHERE user_email = %s AND id LIKE %s "
            "AND created_at > now() - interval '1 day'",
            (user_email, "crawl\\_%"),
        )
        return cur.fetchone()["n"]


def _save_place(card_id: str, user_email: str, state: dict, meta: str, *, release: bool = False) -> bool:
    """Write the crawl's place; False when the card is gone or no longer processing."""
    patch = {"crawl": state, "crawled": {"pages": state["pages"], "done": False}, "meta": meta}
    with store.tx() as cur:
        cur.execute(
            "UPDATE nodes SET data = (coalesce(data, '{}'::jsonb) - %s::text[]) || %s::jsonb, updated_at = now() "
            "WHERE id = %s AND user_email = %s AND status = 'processing' AND deleted_at IS NULL",
            (["ingestLease"] if release else [], json.dumps(patch), card_id, user_email),
        )
        return cur.rowcount == 1


def _drop_state(card_id: str, user_email: str, *keys: str) -> None:
    with store.tx() as cur:
        cur.execute("UPDATE nodes SET data = coalesce(data, '{}'::jsonb) - %s::text[] "
                    "WHERE id = %s AND user_email = %s", (list(keys), card_id, user_email))


def _fail(card_id: str, user_email: str, message: str) -> None:
    ingest._fail(card_id, user_email, message)
    _drop_state(card_id, user_email, "crawl")


def _location(url: str, section: int) -> str:
    return f"{clip_text(urlparse(url).path or '/', 60)} § {section}"


def _read_page(card_id: str, user_email: str, canvas_id, state: dict, page: dict) -> Optional[int]:
    """Chunk, embed and store one page. Returns its chunk count, or None when embedding failed."""
    room = min(CHUNKS_PER_PAGE, MAX_CHUNKS - state["chunks"])
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(page["text"])[:room]
    if not chunks:
        return 0
    embs = ingest.get_embeddings(chunks)
    if len(embs) != len(chunks) or ingest._all_zero(embs):
        return None
    metas = [{"loc": _location(page["url"], i + 1), "url": page["url"]} for i in range(len(chunks))]
    file_id = ingest._save_text_file(card_id, user_email, canvas_id, page.get("title") or page["url"],
                                     "text/html", page["text"], file_id=f"crawl_{card_id}_{state['pages']}")
    ingest._write_chunks(file_id, user_email, chunks, embs, metas)
    return len(chunks)


def _meta(state: dict, note: str = "") -> str:
    n = state["pages"]
    parts = [state["host"], f"{n} page{'' if n == 1 else 's'}"]
    if state["skipped"]:
        parts.append(f"{state['skipped']} skipped")
    if note:
        parts.append(note)
    return " · ".join(parts)


def _finish(card_id: str, user_email: str, state: dict, note: str = "") -> None:
    """Overview and key passages over everything read, then ready."""
    with store.tx() as cur:
        cur.execute(
            "SELECT fc.chunk_text, fc.metadata, fc.embedding::text AS embedding "
            "FROM file_chunks fc JOIN external_files ef ON ef.id = fc.file_id "
            "WHERE ef.node_id = %s AND ef.user_email = %s ORDER BY ef.created_at, fc.chunk_index",
            (card_id, user_email),
        )
        rows = cur.fetchall()
    chunks = [r["chunk_text"] for r in rows]
    metas = [r["metadata"] for r in rows]
    embs = [json.loads(r["embedding"]) for r in rows]
    summary = ingest.make_overview("\n\n".join(chunks), user_email)
    overview_emb = ingest.get_embedding(summary) or []
    if not any(overview_emb):
        overview_emb = embs[0]
    ingest.set_card_status(store, card_id, "ready", {
        "title": state["title"] or state["host"], "meta": _meta(state, note), "summary": summary,
        "excerpts": ingest.pick_key_passages(chunks, metas, embs, overview_emb),
        "crawled": {"pages": state["pages"], "done": True},
    }, user_email)
    _drop_state(card_id, user_email, "crawl")


def _start(url: str) -> dict:
    host, prefix = scope_of(url)
    return {"host": host, "prefix": prefix, "queue": [[url, 0]], "seen": [url], "hashes": [],
            "pages": 0, "chunks": 0, "skipped": 0, "slices": 0, "title": ""}


def _fetch_one(url: str) -> Optional[safe_fetch.Fetched]:
    try:
        return safe_fetch.fetch(url, max_bytes=MAX_DOWNLOAD_BYTES, timeout=FETCH_TIMEOUT_SECONDS,
                                accept_types=_ALLOWED_CONTENT_TYPES)
    except (safe_fetch.FetchBlocked, safe_fetch.FetchFailed):
        return None


def _follow(state: dict, html: str, base: str, depth: int) -> None:
    if depth >= MAX_DEPTH:
        return
    links = links_in(html, base)
    below = [u for u in links if in_scope(u, state["host"], state["prefix"]) and u not in state["seen"]]
    if depth == 0 and not below:  # docs sites end pages in a slash: /guide/intro/ is one page, not a folder
        state["prefix"] = parent_folder(state["prefix"])
    for link in links:
        if len(state["queue"]) >= MAX_QUEUE:
            return
        if link not in state["seen"] and in_scope(link, state["host"], state["prefix"]):
            state["seen"].append(link)
            state["queue"].append([link, depth + 1])


def ingest_crawl_slice(card_id: str, user_email: str) -> None:
    """Read pages until the site is done or this slice's time is up. Never raises."""
    try:
        _run_slice(card_id, user_email)
    except Exception as e:  # noqa: BLE001
        logger.error(f"crawl({card_id}) failed: {type(e).__name__}")
        _fail(card_id, user_email, _READ_ERROR)


def _run_slice(card_id: str, user_email: str) -> None:
    started = time.monotonic()
    with store.tx() as cur:
        cur.execute("SELECT data, canvas_id FROM nodes WHERE id = %s AND user_email = %s", (card_id, user_email))
        row = cur.fetchone()
    data = (row or {}).get("data") or {}
    if not data.get("url"):
        _fail(card_id, user_email, _READ_ERROR)
        return
    state = data.get("crawl") or _start(data["url"])
    state["slices"] += 1
    rules: dict[str, Optional[RobotFileParser]] = {}
    note = ""

    while state["queue"] and state["pages"] < MAX_PAGES and state["chunks"] < MAX_CHUNKS:
        if _pages_today(user_email) >= PAGES_PER_DAY:
            if not state["pages"]:
                _fail(card_id, user_email, _LIMIT_ERROR)
                return
            note = "daily limit reached"
            break
        url, depth = state["queue"].pop(0)
        first = not state["pages"] and depth == 0
        if not first:
            scheme = urlparse(url).scheme
            if scheme not in rules:
                rules[scheme] = _robots(scheme, state["host"])
            allowed = rules[scheme]
            if allowed and not allowed.can_fetch(_ROBOTS_AGENT, url):
                state["skipped"] += 1
                continue
            delay = (allowed.crawl_delay(_ROBOTS_AGENT) if allowed else None) or 0
            time.sleep(max(POLITE_DELAY_SECONDS, min(float(delay), MAX_ROBOTS_DELAY_SECONDS)))

        got = _fetch_one(url)
        if first and got is not None and "application/pdf" in got.content_type:
            _drop_state(card_id, user_email, "crawl", "crawled")  # one document, not a site
            ingest.ingest_web_source(card_id, user_email)
            return
        if first and got is not None and got.matched:  # a redirect shows where the site really is
            state["host"], state["prefix"] = scope_of(got.url)
        page = _page(got, PAGE_CHARS) if got is not None else None
        readable = bool(page) and len(page["text"].strip()) >= ingest.THIN_PAGE_CHARS
        if readable and not first and not in_scope(got.url, state["host"], state["prefix"]):
            readable = False  # a redirect off the site
        if not readable:
            if first:
                _fail(card_id, user_email, _READ_ERROR)
                return
            state["skipped"] += 1
        else:
            digest = hashlib.sha1(page["text"].encode("utf-8")).hexdigest()[:16]
            if digest not in state["hashes"]:
                state["hashes"].append(digest)
                count = _read_page(card_id, user_email, row["canvas_id"], state, page)
                if count is None:
                    _fail(card_id, user_email, "Failed to generate embeddings")
                    return
                state["pages"] += 1
                state["chunks"] += count
                state["title"] = state["title"] or page.get("title") or ""
            _follow(state, got.text(), got.url, depth)

        if not _save_place(card_id, user_email, state, _meta(state)):
            return  # deleted or reset while reading
        if time.monotonic() - started >= SLICE_SECONDS:
            if state["slices"] >= MAX_SLICES:
                break  # finish with what was read
            if state["queue"]:
                _save_place(card_id, user_email, state, _meta(state), release=True)
                return

    if not state["pages"]:
        _fail(card_id, user_email, _READ_ERROR)
        return
    _finish(card_id, user_email, state, note)
