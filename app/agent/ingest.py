"""V3 source ingestion: turn an uploaded file or fetched page into readable source data.

Produces chunks with locators (page / section / rows), an overview (one call to
SUMMARY_MODEL_NAME), up to five key passages, and drives nodes.status
processing -> ready | failed. No DB connection is held across an embedding or
LLM call (read -> close -> embed/summarise -> short tx to write).
"""
from __future__ import annotations

import csv
import io
import json
import math
from typing import Optional
from urllib.parse import urlparse

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.agent.helpers.get_llm import get_llm
from app.agent.helpers.web_fetch import fetch_url_content
from app.agent.memory import sanitize_summary_text
from app.agent.store.board_store import BoardStore, store
from app.agent.utils.embeddings import get_embedding, get_embeddings
from app.agent.utils.text import clip_text as _clip_text
from app.api.v1.endpoints.files import load_and_split_documents
from app.core.config import settings
from app.core.logger import logger

CSV_ROWS_PER_CHUNK = 40
CSV_TABLE_ROWS = 200
KEY_PASSAGES = 5
PASSAGE_CHARS = 280
OVERVIEW_INPUT_CHARS = 12_000
OVERVIEW_MAX_CHARS = 600
WEB_INGEST_MAX_CHARS = 60_000
DOC_TEXT_MAX_CHARS = 100_000

_OVERVIEW_PROMPT = (
    "Summarise this source in 2–3 plain sentences for someone deciding whether to read it.\n\n"
)


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def parse_csv(data: bytes) -> tuple[list[str], list[dict], dict]:
    """Return (chunks, metadatas with loc 'rows a–b', table {columns, rows <= 200})."""
    all_rows = [r for r in csv.reader(io.StringIO(_decode(data))) if r]
    if not all_rows:
        return [], [], {"columns": [], "rows": []}
    columns, rows = all_rows[0], all_rows[1:]
    header_line = ",".join(columns)
    chunks: list[str] = []
    metas: list[dict] = []
    for start in range(0, len(rows), CSV_ROWS_PER_CHUNK):
        part = rows[start:start + CSV_ROWS_PER_CHUNK]
        chunks.append("\n".join([header_line] + [",".join(r) for r in part]))
        metas.append({"loc": f"rows {start + 1}–{start + len(part)}", "row_end": start + len(part)})
    return chunks, metas, {"columns": columns, "rows": rows[:CSV_TABLE_ROWS]}


def add_locators(source_type: str, metadatas: list[dict]) -> list[dict]:
    out = []
    for i, meta in enumerate(metadatas):
        meta = dict(meta)
        if not meta.get("loc"):
            if source_type == "pdf" and isinstance(meta.get("page"), int):
                meta["loc"] = f"p. {meta['page'] + 1}"
            else:
                meta["loc"] = f"§ {i + 1}"
        out.append(meta)
    return out


def make_overview(text: str, user_email: str) -> str:
    fallback = _clip_text(text, 300)
    try:
        llm = get_llm(settings.SUMMARY_MODEL_NAME, user_id=user_email)
        if not llm:
            return fallback
        reply = llm.invoke(_OVERVIEW_PROMPT + text[:OVERVIEW_INPUT_CHARS])
        out = _clip_text(sanitize_summary_text(getattr(reply, "content", reply)), OVERVIEW_MAX_CHARS)
        return out or fallback
    except Exception as e:  # noqa: BLE001 - overview is best effort
        logger.warning(f"overview failed: {e}")
        return fallback


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def pick_key_passages(chunks, metadatas, chunk_embs, overview_emb, n: int = KEY_PASSAGES) -> list[dict]:
    ranked = sorted(
        range(min(len(chunks), len(chunk_embs))),
        key=lambda i: _cosine(chunk_embs[i], overview_emb),
        reverse=True,
    )
    return [
        {
            "loc": (metadatas[i] or {}).get("loc", f"§ {i + 1}"),
            "text": _clip_text(chunks[i], PASSAGE_CHARS),
            "on": True,
        }
        for i in ranked[:n]
    ]


def set_card_status(store: BoardStore, card_id: str, status: str, data_patch: Optional[dict] = None) -> None:
    with store.tx() as cur:
        cur.execute(
            "UPDATE nodes SET status=%s, data = coalesce(data,'{}') || %s::jsonb, updated_at=now() "
            "WHERE id=%s",
            (status, json.dumps(data_patch or {}), card_id),
        )


def _vec(emb: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in emb) + "]"


def _all_zero(embs: list[list[float]]) -> bool:
    return not embs or all(not any(v) for v in embs)


def _fail(card_id: str, message: str) -> None:
    try:
        set_card_status(store, card_id, "failed", {"error": _clip_text(message, 300)})
    except Exception as e:  # noqa: BLE001
        logger.error(f"could not mark card {card_id} failed: {e}")


def _write_chunks(file_id: str, chunks, embs, metas) -> None:
    """DELETE old chunks, INSERT new ones and mark processed in ONE tx."""
    with store.tx() as cur:
        cur.execute("DELETE FROM file_chunks WHERE file_id = %s", (file_id,))
        for idx, (text, emb, meta) in enumerate(zip(chunks, embs, metas)):
            cur.execute(
                "INSERT INTO file_chunks (file_id, chunk_index, chunk_text, embedding, metadata) "
                "VALUES (%s, %s, %s, %s::vector, %s::jsonb)",
                (file_id, idx, text, _vec(emb), json.dumps(meta)),
            )
        cur.execute(
            "UPDATE external_files SET processed = true, updated_at = NOW() WHERE id = %s",
            (file_id,),
        )


def _embed_and_finish(card_id, file_id, user_email, chunks, metas, overview_text, patch) -> None:
    """Shared tail: embed -> overview -> key passages -> write chunks -> ready."""
    embs = get_embeddings(chunks)
    if len(embs) != len(chunks) or _all_zero(embs):
        _fail(card_id, "Failed to generate embeddings")
        return
    summary = make_overview(overview_text, user_email)
    overview_emb = get_embedding(summary) or []
    if not any(overview_emb):
        overview_emb = embs[0]
    excerpts = pick_key_passages(chunks, metas, embs, overview_emb)
    _write_chunks(file_id, chunks, embs, metas)
    set_card_status(store, card_id, "ready", {**patch, "summary": summary, "excerpts": excerpts})


def _source_kind(source_type: Optional[str], file_name: str, mime: str) -> str:
    lower = (file_name or "").lower()
    if source_type == "csv" or lower.endswith(".csv") or mime == "text/csv":
        return "csv"
    if source_type == "pdf" or lower.endswith(".pdf") or mime == "application/pdf":
        return "pdf"
    return "doc"


def ingest_file_source(card_id: str, user_email: str) -> None:
    try:
        with store.tx() as cur:
            cur.execute(
                "SELECT ef.id, ef.file_name, ef.mime_type, ef.data, n.source_type "
                "FROM external_files ef JOIN nodes n ON n.id = ef.node_id "
                "WHERE ef.node_id = %s AND ef.user_email = %s AND n.user_email = %s "
                "ORDER BY ef.created_at LIMIT 1",
                (card_id, user_email, user_email),
            )
            row = cur.fetchone()
        if not row or row["data"] is None:
            _fail(card_id, "File not found")
            return
        file_id, name, mime = row["id"], row["file_name"] or "", row["mime_type"] or ""
        data = bytes(row["data"])
        kind = _source_kind(row["source_type"], name, mime)

        patch: dict = {}
        if kind == "csv":
            chunks, metas, table = parse_csv(data)
            if not chunks:
                _fail(card_id, "No readable content found")
                return
            patch.update(meta=f"CSV · {metas[-1]['row_end']} rows", table=table)
            full_text = "\n".join(chunks)
        else:
            chunks, metas, full_text, err = load_and_split_documents(data, mime, name)
            if err:
                _fail(card_id, err)
                return
            if not chunks:
                _fail(card_id, "No readable content found")
                return
            metas = add_locators(kind, metas)
            if kind == "pdf":
                pages = max((m.get("page", 0) for m in metas), default=0) + 1
                patch["meta"] = f"PDF · {pages} pages"
            else:
                patch["text"] = _clip_text(full_text, DOC_TEXT_MAX_CHARS)
        _embed_and_finish(card_id, file_id, user_email, chunks, metas, full_text, patch)
    except Exception as e:  # noqa: BLE001
        logger.error(f"ingest_file_source({card_id}) failed: {e}")
        _fail(card_id, str(e) or "Ingestion failed")


def ingest_web_source(card_id: str, user_email: str) -> None:
    try:
        with store.tx() as cur:
            cur.execute(
                "SELECT data->>'url' AS url, canvas_id FROM nodes WHERE id = %s AND user_email = %s",
                (card_id, user_email),
            )
            row = cur.fetchone()
        if not row or not row["url"]:
            _fail(card_id, "Couldn't read this page")
            return
        page = fetch_url_content(row["url"], WEB_INGEST_MAX_CHARS)
        if not page or not (page.get("text") or "").strip():
            _fail(card_id, "Couldn't read this page")
            return
        text = page["text"]
        file_id = f"file_{card_id}"
        with store.tx() as cur:
            cur.execute(
                "INSERT INTO external_files (id, user_email, node_id, canvas_id, file_name, file_type, "
                "mime_type, content, data, processed) "
                "VALUES (%s,%s,%s,%s,%s,'text/html','text/html',%s,NULL,false) "
                "ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content, processed = false, "
                "updated_at = NOW()",
                (file_id, user_email, card_id, row["canvas_id"],
                 _clip_text(page.get("title") or row["url"], 200), text),
            )
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(text)
        if not chunks:
            _fail(card_id, "Couldn't read this page")
            return
        metas = add_locators("web", [{} for _ in chunks])
        host = urlparse(row["url"]).hostname or ""
        patch = {"title": page.get("title") or host, "meta": host}
        _embed_and_finish(card_id, file_id, user_email, chunks, metas, text, patch)
    except Exception as e:  # noqa: BLE001
        logger.error(f"ingest_web_source({card_id}) failed: {e}")
        _fail(card_id, str(e) or "Ingestion failed")
