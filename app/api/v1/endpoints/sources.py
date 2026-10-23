"""Source routes: file upload, link (web / YouTube), ingest and raw download (contract rows 15-17).

Upload and link only create the card (202, `processing`). Ingest runs inside
POST /boards/{b}/cards/{id}/ingest, which the client calls right after: on Vercel, work after
a response is best effort (contract amendment 2026-10-21), so nothing runs in BackgroundTasks.
"""
from __future__ import annotations

import re
from urllib.parse import quote
from typing import Optional
from urllib.parse import parse_qs, urlparse

import httpx
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from app.agent.helpers.link_preview import (
    PREVIEW_TIMEOUT_SECONDS, LinkBlocked, LinkUnreachable, fetch_link_preview,
)
from app.agent.helpers.web_fetch import host_is_public
from app.agent.ingest import ingest_file_source, ingest_web_source
from app.agent.store import card_store
from app.agent.store.board_store import fetch_card_json, require_board, require_card, store
from app.api.auth import get_current_user_id
from app.api.errors import ApiError
from app.api.limiter import BOARD_RATE, limiter, user_key
from app.core.logger import logger
from app.schemas.board import ID_PATTERN, CardCreate, LinkCreate

router = APIRouter()

from app.api.upload_limit import MAX_UPLOAD_BYTES  # noqa: E402  (4 * 1024 * 1024)
MAX_DESC_CHARS = 2000
MAX_FILE_NAME_CHARS = 255

TYPE_BY_EXT = {
    "pdf": ("pdf", "processing"), "docx": ("doc", "processing"), "txt": ("doc", "processing"),
    "md": ("doc", "processing"), "csv": ("csv", "processing"),
    "png": ("image", "soon"), "jpg": ("image", "soon"), "jpeg": ("image", "soon"),
    "webp": ("image", "soon"), "gif": ("image", "soon"),
    "mp3": ("audio", "soon"), "m4a": ("audio", "soon"), "wav": ("audio", "soon"),
}

# Stored/served mime is derived from the extension, never from the client's header,
# so an uploaded .txt can't be served back inline as text/html.
MIME_BY_EXT = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain", "md": "text/markdown", "csv": "text/csv",
    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp",
    "gif": "image/gif", "mp3": "audio/mpeg", "m4a": "audio/mp4", "wav": "audio/wav",
}

_YT_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com"}


def format_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{round(n / 1024)} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def parse_youtube_id(url: str) -> Optional[str]:
    # urlparse silently drops tab/CR/LF, which would let an id with a trailing newline through.
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        return None
    try:
        u = urlparse(url)
    except ValueError:
        return None
    host = (u.hostname or "").lower()
    cand = None
    if host in _YT_HOSTS:
        if u.path == "/watch":
            cand = (parse_qs(u.query).get("v") or [None])[0]
        elif u.path.startswith("/shorts/"):
            cand = u.path[len("/shorts/"):].split("/")[0]
    elif host == "youtu.be":
        cand = u.path.lstrip("/").split("/")[0]
    return cand if cand and _YT_ID.fullmatch(cand) else None


def fetch_youtube_oembed(url: str, timeout: float = 5) -> Optional[dict]:
    try:
        r = httpx.get("https://www.youtube.com/oembed", params={"url": url, "format": "json"},
                      timeout=timeout)
        r.raise_for_status()
        j = r.json()
        return {"title": j.get("title"), "thumbnail_url": j.get("thumbnail_url")}
    except Exception as e:  # noqa: BLE001 - oEmbed is best effort
        logger.warning(f"youtube oembed failed: {e}")
        return None


def _basename(name: str) -> str:
    return re.split(r"[\\/]", name or "")[-1]


def _create_file_card(user_id, board_id, card_id, x, y, desk_id, desc, name, ctype, raw, source_type, status):
    email = store.email_for(user_id)
    stem, _, ext = name.rpartition(".")
    data = {
        "title": stem or name, "fileName": name, "fileType": ctype, "fileSize": len(raw),
        "meta": f"{source_type.upper()} · {format_size(len(raw))}",
    }
    if desc:
        data["desc"] = desc
    body = CardCreate(id=card_id, kind="source", source_type=source_type, x=x, y=y, desk_id=desk_id,
                      status=None if status == "processing" else status, data=data)
    with store.tx() as cur:
        card, created = card_store.create_card(cur, email, board_id, body)
        if created:
            cur.execute(
                "INSERT INTO external_files (id, user_email, node_id, canvas_id, file_name, file_type, "
                "file_size, data, mime_type, processed) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,false) "
                "ON CONFLICT (id) DO NOTHING",
                (f"file_{card_id}", email, card_id, board_id, name, ctype, len(raw), raw,
                 MIME_BY_EXT[ext.lower()]),
            )
    return card, created, email


@router.post("/boards/{board_id}/sources/file")
@limiter.limit(BOARD_RATE, key_func=user_key)
async def api_upload_file(
    board_id: str, request: Request,
    id: str = Form(pattern=ID_PATTERN),
    file: UploadFile = File(...),
    x: float = Form(),
    y: float = Form(),
    deskId: Optional[str] = Form(default=None, pattern=ID_PATTERN),
    desc: Optional[str] = Form(default=None, max_length=MAX_DESC_CHARS),
    user_id: str = Depends(get_current_user_id),
):
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ApiError(413, "too_large", "files are limited to 4 MB")
    name = _basename(file.filename or "")
    ext = name.rpartition(".")[2].lower() if "." in name else ""
    if ext not in TYPE_BY_EXT:
        raise ApiError(415, "unsupported_type", "unsupported file type")
    source_type, status = TYPE_BY_EXT[ext]
    if len(name) > MAX_FILE_NAME_CHARS:  # keep the extension, trim the stem
        name = name[:MAX_FILE_NAME_CHARS - len(ext) - 1] + "." + ext
    ctype = (file.content_type or MIME_BY_EXT[ext])[:100]
    card, created, _email = await run_in_threadpool(
        _create_file_card, user_id, board_id, id, x, y, deskId, desc, name, ctype, raw, source_type, status)
    return JSONResponse(card, status_code=202 if created else 200)


@router.post("/boards/{board_id}/sources/link")
@limiter.limit(BOARD_RATE, key_func=user_key)
def api_add_link(
    board_id: str, body: LinkCreate, request: Request,
    user_id: str = Depends(get_current_user_id),
):
    email = store.email_for(user_id)
    try:
        u = urlparse(body.url.strip())
        host = u.hostname or ""
    except ValueError:
        raise ApiError(400, "invalid", "invalid url")
    if u.scheme not in ("http", "https") or not host:
        raise ApiError(400, "invalid", "only http(s) links are supported")
    url = body.url.strip()
    desc = {"desc": body.desc} if body.desc else {}

    # Re-post / missing board are answered before any network work (oEmbed, DNS).
    with store.tx() as cur:
        require_board(cur, board_id, email)
        existing = card_store._existing_own(cur, email, board_id, body.id)
    if existing is not None:
        return JSONResponse(existing, status_code=200)

    ytid = parse_youtube_id(url)
    if ytid:
        oembed = fetch_youtube_oembed(url) or {}
        source_type, status = "video", "soon"
        data = {"url": url, "ytid": ytid, "title": oembed.get("title") or "YouTube video",
                "meta": "YouTube", **desc}
    else:
        if not host_is_public(host):
            raise ApiError(400, "invalid", "this address can't be fetched")
        source_type, status = "web", None
        data = {"url": url, "title": host, "meta": host, **desc}

    card_body = CardCreate(id=body.id, kind="source", source_type=source_type, x=body.x, y=body.y,
                           desk_id=body.desk_id, status=status, data=data)
    with store.tx() as cur:
        card, created = card_store.create_card(cur, email, board_id, card_body)
    return JSONResponse(card, status_code=202 if created else 200)


# ── Ingest (inside the request) ───────────────────────────────────────────────

INGEST_LEASE_SECONDS = 300  # Vercel Hobby's per-request limit; an older lease is a dead run
_INGEST_KIND = {"pdf": "file", "doc": "file", "csv": "file", "web": "web"}


def claim_ingest_lease(cur, card_id: str, user_email: str) -> bool:
    """Stamp the lease on a `processing` source with no live lease, in one statement.

    True when this call owns the ingest; False when another call started it < 300 s ago (or
    the card is no longer processing). A missing or malformed stamp counts as expired.
    """
    cur.execute(
        "UPDATE nodes SET data = coalesce(data, '{}'::jsonb) "
        "|| jsonb_build_object('ingestLease', extract(epoch from clock_timestamp())), "
        "updated_at = now() "
        "WHERE id = %s AND user_email = %s AND kind = 'source' AND status = 'processing' "
        "AND deleted_at IS NULL AND CASE WHEN jsonb_typeof(data->'ingestLease') = 'number' "
        "THEN (data->>'ingestLease')::float8 < extract(epoch from clock_timestamp()) - %s "
        "ELSE true END RETURNING id",
        (card_id, user_email, INGEST_LEASE_SECONDS),
    )
    return cur.fetchone() is not None


@router.post("/boards/{board_id}/cards/{card_id}/ingest")
@limiter.limit(BOARD_RATE, key_func=user_key)
def api_ingest_source(
    board_id: str, card_id: str, request: Request, user_id: str = Depends(get_current_user_id),
):
    """Run a source's ingest inside this request and return the card with its final status.

    A sync route, so FastAPI runs it in the threadpool; ingest itself never holds a pooled
    connection across an embedding or LLM call. ready / failed / soon (and notes) return as is.
    """
    email = store.email_for(user_id)
    with store.tx() as cur:
        require_board(cur, board_id, email)
        card = require_card(cur, board_id, card_id, email)
        if card["kind"] != "source":
            raise ApiError(400, "invalid", "only sources are ingested")
        kind = _INGEST_KIND.get(card["source_type"])
        if card["status"] != "processing" or kind is None:
            return fetch_card_json(cur, board_id, card_id)
        claimed = claim_ingest_lease(cur, card_id, email)
    if not claimed:
        return JSONResponse({"status": "processing"}, status_code=202)
    if kind == "web":
        ingest_web_source(card_id, email)
    else:
        ingest_file_source(card_id, email)
    with store.tx() as cur:
        return fetch_card_json(cur, board_id, card_id)


# ── Link preview for the paste box (contract amendment 2026-10-23) ─────────────

LINK_PREVIEW_RATE = "60/minute"  # per user: each call makes an outbound request
MAX_PREVIEW_URL_CHARS = 2048


def _http_url_or_none(value) -> Optional[str]:
    value = str(value or "").strip()
    try:
        scheme = urlparse(value).scheme
    except ValueError:
        return None
    return value if scheme in ("http", "https") and len(value) <= MAX_PREVIEW_URL_CHARS else None


@router.get("/links/preview")
@limiter.limit(LINK_PREVIEW_RATE, key_func=user_key)
def api_link_preview(request: Request, url: str = "", user_id: str = Depends(get_current_user_id)):
    """{url, kind, title, site, description, image} for a pasted link; nothing is stored.

    Blocked addresses are 400 `invalid`, pages that cannot be read are 422 `unreachable`.
    Neither error carries anything the remote server sent.
    """
    url = (url or "").strip()
    try:
        scheme = urlparse(url).scheme
    except ValueError:
        scheme = ""
    if not url or len(url) > MAX_PREVIEW_URL_CHARS or scheme not in ("http", "https"):
        raise ApiError(400, "invalid", "paste an http(s) link")
    if parse_youtube_id(url):
        oembed = fetch_youtube_oembed(url, timeout=PREVIEW_TIMEOUT_SECONDS) or {}
        title = " ".join(str(oembed.get("title") or "").split())[:300] or None
        return {"url": url, "kind": "video", "title": title, "site": "YouTube", "description": None,
                "image": _http_url_or_none(oembed.get("thumbnail_url"))}
    try:
        preview = fetch_link_preview(url)
    except LinkBlocked:
        raise ApiError(400, "invalid", "this address can't be previewed") from None
    except LinkUnreachable:
        raise ApiError(422, "unreachable", "couldn't read this page") from None
    return {"url": url, "kind": "web", **preview}


@router.get("/files/{card_id}/raw")
@limiter.limit(BOARD_RATE, key_func=user_key)
def api_raw_file(card_id: str, request: Request, user_id: str = Depends(get_current_user_id)):
    email = store.email_for(user_id)
    with store.tx() as cur:
        cur.execute(
            "SELECT ef.data, ef.mime_type, ef.file_name FROM external_files ef "
            "JOIN nodes n ON n.id = ef.node_id "
            "JOIN canvases c ON c.id = n.canvas_id AND c.deleted_at IS NULL "
            "WHERE ef.node_id = %s AND ef.user_email = %s AND n.user_email = %s "
            "AND n.kind IS NOT NULL AND n.deleted_at IS NULL AND ef.data IS NOT NULL "
            "ORDER BY ef.created_at LIMIT 1",
            (card_id, email, email),
        )
        row = cur.fetchone()
    if not row:
        raise ApiError(404, "not_found")
    full = row["file_name"] or "file"
    ascii_name = re.sub(r'[^\x20-\x7e]|"|\\', "_", full) or "file"
    return Response(
        bytes(row["data"]),
        media_type=row["mime_type"] or "application/octet-stream",
        headers={
            "Content-Disposition": (
                f"inline; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(full, safe='')}"
            ),
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "sandbox",
        },
    )


PASSAGE_SOURCE_TYPES = ("pdf", "doc", "web", "csv")


@router.get("/files/{card_id}/passages")
@limiter.limit(BOARD_RATE, key_func=user_key)
def api_passages(card_id: str, request: Request, user_id: str = Depends(get_current_user_id)):
    """A source's chunks in document order, for the side panel.

    A source that is still processing or failed returns 200 with whatever chunks exist (usually
    none), so the panel can render an empty state instead of an error. A missing, deleted or
    foreign card is 404; a card that has no passages by kind (note, chat, video, ...) is 400.
    """
    email = store.email_for(user_id)
    with store.tx() as cur:
        cur.execute(
            "SELECT n.kind, n.source_type FROM nodes n "
            "JOIN canvases c ON c.id = n.canvas_id AND c.deleted_at IS NULL "
            "WHERE n.id = %s AND n.user_email = %s AND n.kind IS NOT NULL AND n.deleted_at IS NULL",
            (card_id, email),
        )
        node = cur.fetchone()
        if node is None:
            raise ApiError(404, "not_found")
        if node["kind"] != "source" or node["source_type"] not in PASSAGE_SOURCE_TYPES:
            raise ApiError(400, "invalid", "this card has no passages")
        cur.execute(
            "SELECT fc.metadata->>'loc' AS loc, fc.metadata->>'page' AS page, fc.chunk_text "
            "FROM file_chunks fc JOIN external_files ef ON fc.file_id = ef.id "
            "WHERE ef.node_id = %s AND ef.user_email = %s "
            "ORDER BY ef.created_at, fc.chunk_index",
            (card_id, email),
        )
        rows = cur.fetchall()
    passages = []
    for r in rows:
        try:
            page = int(r["page"]) + 1  # metadata page is 0-based; the API's is 1-based
        except (TypeError, ValueError):
            page = None
        passages.append({"loc": r["loc"], "page": page, "text": r["chunk_text"]})
    return {"passages": passages}
