"""Source routes: file upload, link (web / YouTube) and raw download (contract rows 15-17)."""
from __future__ import annotations

import re
from urllib.parse import quote
from typing import Optional
from urllib.parse import parse_qs, urlparse

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from app.agent.helpers.web_fetch import host_is_public
from app.agent.ingest import ingest_file_source, ingest_web_source
from app.agent.store import card_store
from app.agent.store.board_store import require_board, store
from app.api.auth import get_current_user_id
from app.api.errors import ApiError
from app.api.limiter import BOARD_RATE, limiter, user_key
from app.core.logger import logger
from app.schemas.board import ID_PATTERN, CardCreate, LinkCreate

router = APIRouter()

from app.api.upload_limit import MAX_UPLOAD_BYTES  # noqa: E402  (4 * 1024 * 1024)
MAX_DESC_CHARS = 2000

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


def fetch_youtube_oembed(url: str) -> Optional[dict]:
    try:
        r = httpx.get("https://www.youtube.com/oembed", params={"url": url, "format": "json"}, timeout=5)
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
    board_id: str, request: Request, background_tasks: BackgroundTasks,
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
    ctype = (file.content_type or MIME_BY_EXT[ext])[:100]
    card, created, email = await run_in_threadpool(
        _create_file_card, user_id, board_id, id, x, y, deskId, desc, name, ctype, raw, source_type, status)
    if created and card["status"] == "processing":
        background_tasks.add_task(ingest_file_source, id, email)
    return JSONResponse(card, status_code=202 if created else 200)


@router.post("/boards/{board_id}/sources/link")
@limiter.limit(BOARD_RATE, key_func=user_key)
def api_add_link(
    board_id: str, body: LinkCreate, request: Request, background_tasks: BackgroundTasks,
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
    if created and card["status"] == "processing":
        background_tasks.add_task(ingest_web_source, body.id, email)
    return JSONResponse(card, status_code=202 if created else 200)


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
