"""
V3 context builder: turns a chat's read set into prompt context (spec 5.2, 5.3, 5.5).

Only readable items are used: files are searched by vector, switched-off excerpts are
skipped, notes and image summaries go in whole, linked chats become a LINKED_ANSWERS block.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import ceil

from app.agent.reads import (
    ANSWER_MAX_CHARS, ANSWER_SUMMARY_MAX_CHARS, FILE_BACKED, FILE_SNIPPET_CHARS,
    NOTE_MAX_CHARS, Reads,
)
from app.agent.utils.text import clip_text
from app.core.logger import logger

TYPE_LABEL = {"pdf": "PDF", "doc": "Document", "csv": "CSV", "web": "Web page", "video": "Video",
              "audio": "Audio", "note": "Note", "image": "Image"}
DESC_HEADER_CHARS = 160
LABEL_CHARS = 120
IMAGE_SUMMARY_CHARS = 4000
_DELIMS = re.compile(
    r"<\s*/?\s*(EXTERNAL_CONTEXT|LINKED_ANSWERS|THREAD_SUMMARY|MEMORY_FACTS|RELEVANT_CONTEXT)\s*>", re.I)


def _safe(text) -> str:
    """Remove prompt block delimiters from board-sourced text."""
    return _DELIMS.sub("", str(text or ""))


def _label(text, limit: int = LABEL_CHARS) -> str:
    """One line, no delimiters, clipped: for titles, names and descriptions."""
    return clip_text(" ".join(_safe(text).split()), limit)


def source_header(card: dict) -> str:
    data = card.get("data") or {}
    title = _label(data.get("title") or data.get("fileName")
                   or " ".join(str(data.get("text") or "").split())[:40]) or "Untitled"
    label = TYPE_LABEL.get(card.get("source_type"), str(card.get("source_type") or "Source"))
    desc = _label(data.get("desc"), DESC_HEADER_CHARS)
    head = f"[source | {label} | {title}"
    return f"{head} | for: {desc}]" if desc else f"{head}]"


@dataclass
class ScoredChunk:
    card_id: str
    loc: str | None
    text: str
    score: float


@dataclass
class BoardContext:
    notes: list[str] = field(default_factory=list)
    chunks: list[ScoredChunk] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)
    linked_answers: str | None = None


def has_meaningful_embedding(embedding) -> bool:
    return bool(embedding) and any(abs(float(x)) > 1e-8 for x in embedding)


def _vec(emb: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in emb) + "]"


def search_board_chunks(cur, user_email: str, card_ids: list[str], query_embedding: list[float],
                        limit: int, off: dict[str, set[str]] | None = None) -> list[ScoredChunk]:
    """Nearest chunks of the given cards; `off` ({card: {loc}}) is filtered in SQL."""
    if not card_ids or limit <= 0:
        return []
    pairs = [(c, loc) for c, locs in (off or {}).items() for loc in locs]
    cur.execute(
        "SELECT ef.node_id, fc.metadata->>'loc' AS loc, fc.chunk_text, "
        "(fc.embedding <=> %s::vector) AS distance "
        "FROM file_chunks fc JOIN external_files ef ON fc.file_id = ef.id "
        "WHERE ef.node_id = ANY(%s) AND ef.user_email = %s AND fc.embedding IS NOT NULL "
        "AND NOT EXISTS (SELECT 1 FROM unnest(%s::text[], %s::text[]) AS o(n, l) "
        "WHERE o.n = ef.node_id AND o.l = fc.metadata->>'loc') "
        "ORDER BY distance ASC LIMIT %s",
        (_vec(query_embedding), list(card_ids), user_email,
         [c for c, _ in pairs], [loc for _, loc in pairs], limit),
    )
    return [ScoredChunk(r["node_id"], r["loc"], _safe(r["chunk_text"]), 1.0 - float(r["distance"]))
            for r in cur.fetchall()]


def _off_locs(data: dict) -> set[str]:
    ex = data.get("excerpts")
    if not isinstance(ex, list):
        return set()
    return {str(e.get("loc")) for e in ex
            if isinstance(e, dict) and e.get("on") is False and e.get("loc") is not None}


def _linked_answers(entries: list[str]) -> str | None:
    if not entries:
        return None
    return "<LINKED_ANSWERS>\n" + "\n\n".join(entries) + "\n</LINKED_ANSWERS>"


def build_board_context(store, reads: Reads, user_email: str, query_embedding: list[float] | None,
                        top_k: int) -> BoardContext:
    """`user_email` is the already-resolved tenancy key (the one used for resolve_reads)."""
    ctx = BoardContext()
    ready_ids = [i.card_id for i in reads.readable()]
    if not ready_ids:
        return ctx
    with store.tx() as cur:
        cur.execute(
            "SELECT id, kind, source_type, data, summary FROM nodes "
            "WHERE id = ANY(%s) AND user_email = %s AND kind IS NOT NULL AND deleted_at IS NULL",
            (ready_ids, user_email),
        )
        cards = {r["id"]: r for r in cur.fetchall()}
        if len(cards) < len(set(ready_ids)):
            logger.warning("board context: %d of %d readable cards found for chat %s "
                           "(user_email mismatch?)", len(cards), len(set(ready_ids)), reads.chat_id)
        chat_ids = [i for i in ready_ids if i in cards and cards[i]["kind"] == "chat"]
        latest: dict[str, str] = {}
        if chat_ids:
            cur.execute(
                "SELECT DISTINCT ON (node_id) node_id, content FROM messages "
                "WHERE node_id = ANY(%s) AND user_email = %s AND role = 'assistant' AND NOT inherited "
                "ORDER BY node_id, position DESC",
                (chat_ids, user_email),
            )
            latest = {r["node_id"]: r["content"] or "" for r in cur.fetchall()}

        file_ids: list[str] = []
        off: dict[str, set[str]] = {}
        entries: list[str] = []
        for cid in ready_ids:
            row = cards.get(cid)
            if row is None:
                continue
            data = row["data"] or {}
            if row["kind"] == "chat":
                if cid not in latest:
                    continue
                parts = [f"[answer | {_label(data.get('name')) or 'Chat'}]"]
                if (row["summary"] or "").strip():
                    parts.append("Summary: " + clip_text(_safe(row["summary"]), ANSWER_SUMMARY_MAX_CHARS))
                parts.append(clip_text(_safe(latest[cid]), ANSWER_MAX_CHARS))
                entries.append("\n".join(parts))
                continue
            t = row["source_type"]
            header = source_header(row)
            if t == "note":
                ctx.headers[cid] = header
                ctx.notes.append(header + "\n" + clip_text(_safe(data.get("text")), NOTE_MAX_CHARS))
            elif t == "image":
                if data.get("summary"):
                    ctx.headers[cid] = header
                    ctx.notes.append(header + "\n" + clip_text(_safe(data["summary"]), IMAGE_SUMMARY_CHARS))
            elif t in FILE_BACKED:
                ctx.headers[cid] = header
                file_ids.append(cid)
                off[cid] = _off_locs(data)
        ctx.linked_answers = _linked_answers(entries)

        if file_ids and top_k > 0 and has_meaningful_embedding(query_embedding):
            ctx.chunks = search_board_chunks(cur, user_email, file_ids, query_embedding, top_k, off)
    return ctx


def _rendered(c: ScoredChunk) -> str:
    return f"({c.loc}) {c.text}" if c.loc is not None else c.text


def render_source_blocks(ctx: BoardContext, chunks: list[ScoredChunk]) -> list[str]:
    order: list[str] = []
    by_card: dict[str, list[str]] = {}
    for c in chunks:
        if c.card_id not in by_card:
            order.append(c.card_id)
            by_card[c.card_id] = []
        prefix = f"({c.loc}) " if c.loc is not None else ""
        by_card[c.card_id].append(prefix + clip_text(c.text, FILE_SNIPPET_CHARS))
    return [(ctx.headers.get(cid) or f"[source | {cid}]") + "\n" + "\n".join(by_card[cid])
            for cid in order]


def trim_chunks_to_budget(chunks: list[ScoredChunk], fixed_chars: int,
                          budget_tokens: int | None) -> list[ScoredChunk]:
    if budget_tokens is None:
        return chunks
    keep = list(range(len(chunks)))
    by_score = sorted(keep, key=lambda i: (chunks[i].score, -i))  # lowest score drops first

    def fits() -> bool:
        return ceil((fixed_chars + sum(len(_rendered(chunks[i])) for i in keep)) / 4) <= budget_tokens

    while keep and not fits():
        keep.remove(by_score.pop(0))
    return [chunks[i] for i in keep]
