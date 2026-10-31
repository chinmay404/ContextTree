"""
V3 context builder: turns a chat's read set into prompt context (spec 5.2, 5.3, 5.5).

Only readable items are used: files are searched by vector, switched-off excerpts are
skipped, notes and image summaries go in whole, linked chats become a LINKED_ANSWERS block.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from math import ceil

from app.agent.citations import QUOTE_MAX_CHARS, neutralise_markers, strip_markers
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
RRF_K = 60  # the standard reciprocal-rank constant: rank matters, raw scores don't have to compare
# Without model scores, notes and linked answers may fill this share of the room while file
# passages are waiting; the rest is theirs. Otherwise a note-heavy desk sends no documents at all.
NON_CHUNK_SHARE = 0.6
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
    page: int | None = None  # 1-based (chunk metadata stores it 0-based)


@dataclass
class BoardContext:
    notes: list[str] = field(default_factory=list)
    chunks: list[ScoredChunk] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)
    linked_answers: str | None = None
    # The same notes / linked answers as parts, for numbering: (card_id, header, body, quote)
    # and (card_id, head lines, body, quote). `notes` / `linked_answers` stay unnumbered.
    note_parts: list[tuple[str, str, str, str]] = field(default_factory=list)
    answer_parts: list[tuple[str, list[str], str, str]] = field(default_factory=list)


@dataclass
class NumberedContext:
    notes: list[str]
    sources: list[str]
    linked_answers: str | None
    passages: dict[int, dict]  # n -> Citation {n, cardId, loc, page, quote}


def _sent_and_quote(text, limit: int, for_prompt=neutralise_markers) -> tuple[str, str]:
    """The clipped prompt copy, and the quote: the stored text clipped the same way, without the
    clip ellipsis. Only the prompt copy has bracketed numbers rewritten (or, for another chat's
    answer, its old markers removed), so they cannot pose as this turn's passage numbers."""
    raw = str(text or "").strip()
    sent = clip_text(for_prompt(raw), limit)
    quote = raw if len(raw) <= limit else raw[: limit - 1].rstrip()
    return sent, quote[:QUOTE_MAX_CHARS]


def _page(value) -> int | None:
    try:
        return int(value) + 1
    except (TypeError, ValueError):
        return None


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
        "SELECT ef.node_id, fc.metadata->>'loc' AS loc, fc.metadata->>'page' AS page, fc.chunk_text, "
        "(fc.embedding <=> %s::vector) AS distance "
        "FROM file_chunks fc JOIN external_files ef ON fc.file_id = ef.id "
        "WHERE ef.node_id = ANY(%s) AND ef.user_email = %s AND fc.embedding IS NOT NULL "
        "AND NOT EXISTS (SELECT 1 FROM unnest(%s::text[], %s::text[]) AS o(n, l) "
        "WHERE o.n = ef.node_id AND o.l = fc.metadata->>'loc') "
        "ORDER BY distance ASC LIMIT %s",
        (_vec(query_embedding), list(card_ids), user_email,
         [c for c, _ in pairs], [loc for _, loc in pairs], limit),
    )
    return [ScoredChunk(r["node_id"], r["loc"], _safe(r["chunk_text"]), 1.0 - float(r["distance"]),
                        _page(r["page"]))
            for r in cur.fetchall()]


_OFF_FILTER = ("AND NOT EXISTS (SELECT 1 FROM unnest(%s::text[], %s::text[]) AS o(n, l) "
               "WHERE o.n = ef.node_id AND o.l = fc.metadata->>'loc') ")


def gather_board_chunks(cur, user_email: str, card_ids: list[str], query_embedding: list[float] | None,
                        search_text: str | None, per_source: int,
                        off: dict[str, set[str]] | None = None) -> list[ScoredChunk]:
    """The best `per_source` passages of each card, by vector and keyword rank fused (RRF).

    Ranks are taken inside each card, so a large source cannot use up a small one's slots. Both
    searches run over this chat's cards only, which keeps an unindexed tsvector cheap. Keyword
    search needs no embedding provider, so documents survive an embedding outage.
    """
    if not card_ids or per_source <= 0:
        return []
    pairs = [(c, loc) for c, locs in (off or {}).items() for loc in locs]
    off_args = ([c for c, _ in pairs], [loc for _, loc in pairs])
    found: dict[int, ScoredChunk] = {}
    fused: dict[int, float] = {}

    def take(rows):
        for r in rows:
            cid = r["id"]
            if cid not in found:
                found[cid] = ScoredChunk(r["node_id"], r["loc"], _safe(r["chunk_text"]), 0.0, _page(r["page"]))
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + r["rn"])

    if has_meaningful_embedding(query_embedding):
        cur.execute(
            "SELECT * FROM (SELECT fc.id, ef.node_id, fc.metadata->>'loc' AS loc, fc.metadata->>'page' AS page, "
            "fc.chunk_text, ROW_NUMBER() OVER (PARTITION BY ef.node_id ORDER BY fc.embedding <=> %s::vector, fc.id) AS rn "
            "FROM file_chunks fc JOIN external_files ef ON fc.file_id = ef.id "
            "WHERE ef.node_id = ANY(%s) AND ef.user_email = %s AND fc.embedding IS NOT NULL "
            + _OFF_FILTER + ") s WHERE rn <= %s",
            (_vec(query_embedding), list(card_ids), user_email, *off_args, per_source),
        )
        take(cur.fetchall())
    if (search_text or "").strip():
        # plainto_tsquery ANDs its terms; a question rarely has them all in one passage, so OR them.
        cur.execute(
            "SELECT * FROM (SELECT fc.id, ef.node_id, fc.metadata->>'loc' AS loc, fc.metadata->>'page' AS page, "
            "fc.chunk_text, ROW_NUMBER() OVER (PARTITION BY ef.node_id "
            "ORDER BY ts_rank_cd(to_tsvector('english', fc.chunk_text), q.q) DESC, fc.id) AS rn "
            "FROM file_chunks fc JOIN external_files ef ON fc.file_id = ef.id, "
            "(SELECT NULLIF(replace(plainto_tsquery('english', %s)::text, ' & ', ' | '), '')::tsquery AS q) q "
            "WHERE ef.node_id = ANY(%s) AND ef.user_email = %s AND to_tsvector('english', fc.chunk_text) @@ q.q "
            + _OFF_FILTER + ") s WHERE rn <= %s",
            (search_text, list(card_ids), user_email, *off_args, per_source),
        )
        take(cur.fetchall())

    chunks = [replace(found[cid], score=score) for cid, score in fused.items()]
    chunks.sort(key=lambda c: -c.score)
    return chunks


def keep_per_source(chunks: list[ScoredChunk], per_source: int) -> list[ScoredChunk]:
    """The first `per_source` of each card, in the given (best-first) order."""
    seen: dict[str, int] = {}
    kept = []
    for c in chunks:
        seen[c.card_id] = seen.get(c.card_id, 0) + 1
        if seen[c.card_id] <= per_source:
            kept.append(c)
    return kept


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


def _add_note(ctx: BoardContext, cid: str, header: str, text, limit: int) -> None:
    body, quote = _sent_and_quote(_safe(text), limit)
    ctx.note_parts.append((cid, header, body, quote))
    ctx.notes.append(header + "\n" + body)


def build_board_context(store, reads: Reads, user_email: str, query_embedding: list[float] | None,
                        top_k: int, search_text: str | None = None) -> BoardContext:
    """`user_email` is the already-resolved tenancy key (the one used for resolve_reads).
    `top_k` is passages per source, the same limit the read-set token meter counts."""
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
                    parts.append("Summary: " + clip_text(strip_markers(_safe(row["summary"])),
                                                         ANSWER_SUMMARY_MAX_CHARS))
                body, quote = _sent_and_quote(_safe(latest[cid]), ANSWER_MAX_CHARS, strip_markers)
                ctx.answer_parts.append((cid, list(parts), body, quote))
                entries.append("\n".join(parts + [body]))
                continue
            t = row["source_type"]
            header = source_header(row)
            if t == "note":
                ctx.headers[cid] = header
                _add_note(ctx, cid, header, data.get("text"), NOTE_MAX_CHARS)
            elif t == "image":
                if data.get("summary"):
                    ctx.headers[cid] = header
                    _add_note(ctx, cid, header, data["summary"], IMAGE_SUMMARY_CHARS)
            elif t in FILE_BACKED:
                ctx.headers[cid] = header
                file_ids.append(cid)
                off[cid] = _off_locs(data)
        ctx.linked_answers = _linked_answers(entries)

        if file_ids and top_k > 0:
            ctx.chunks = gather_board_chunks(cur, user_email, file_ids, query_embedding, search_text, top_k, off)
    return ctx


def _rendered(c: ScoredChunk) -> str:
    return f"({c.loc}) {c.text}" if c.loc is not None else c.text


def number_passages(ctx: BoardContext, chunks: list[ScoredChunk]) -> NumberedContext:
    """Render the blocks a V3 turn sends, each passage prefixed ``[n]`` in prompt order.

    Prompt order is EXTERNAL_CONTEXT (notes, then file chunks grouped by card) and then
    LINKED_ANSWERS. Call it with the chunks that survived the budget: only what is sent
    gets a number.
    """
    passages: dict[int, dict] = {}

    def add(card_id: str, loc, page, quote: str) -> int:
        n = len(passages) + 1
        passages[n] = {"n": n, "cardId": card_id, "loc": loc, "page": page, "quote": quote}
        return n

    notes = [f"{header}\n[{add(cid, None, None, quote)}] {body}"
             for cid, header, body, quote in ctx.note_parts]

    by_card: dict[str, list[ScoredChunk]] = {}
    for c in chunks:
        by_card.setdefault(c.card_id, []).append(c)  # dicts keep first-seen card order
    sources = []
    for cid, group in by_card.items():
        lines = []
        for c in group:
            sent, quote = _sent_and_quote(c.text, FILE_SNIPPET_CHARS)
            prefix = f"({c.loc}) " if c.loc is not None else ""
            lines.append(f"[{add(cid, c.loc, c.page, quote)}] {prefix}{sent}")
        sources.append((ctx.headers.get(cid) or f"[source | {cid}]") + "\n" + "\n".join(lines))

    entries = ["\n".join(head + [f"[{add(cid, None, None, quote)}] {body}"])
               for cid, head, body, quote in ctx.answer_parts]
    return NumberedContext(notes, sources, _linked_answers(entries), passages)


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


@dataclass
class Fit:
    ctx: BoardContext  # note_parts / answer_parts hold only what was kept
    chunks: list[ScoredChunk]
    dropped: list[dict]  # {cardId, loc, score, reason}, what the budget left out


def fit_to_budget(ctx: BoardContext, chunks: list[ScoredChunk], fixed_chars: int, budget_tokens: int | None,
                  scores: dict[tuple[str, str | None], float] | None = None) -> Fit:
    """Choose what fits in `budget_tokens`, counting every block, and say what did not.

    With model `scores` ({(cardId, loc): chance it helps}) everything competes on score. Without
    them, notes and answers go first up to NON_CHUNK_SHARE of the room when files are waiting,
    then the passages best-first, then any notes and answers that still fit.
    """
    if budget_tokens is None:
        return Fit(ctx, chunks, [])
    room = budget_tokens * 4 - fixed_chars
    # (kind, index, cardId, loc, chars, score)
    others = [("note", i, p[0], None, len(p[1]) + len(p[2]), None) for i, p in enumerate(ctx.note_parts)]
    others += [("answer", i, p[0], None, sum(map(len, p[1])) + len(p[2]), None) for i, p in enumerate(ctx.answer_parts)]
    passages = [("chunk", i, c.card_id, c.loc, len(_rendered(c)), c.score) for i, c in enumerate(chunks)]
    kept: set[tuple[str, int]] = set()

    def offer(items, limit=None):
        nonlocal room
        left = []
        spent = 0
        for item in items:
            chars = item[4]
            if chars <= room and (limit is None or spent + chars <= limit):
                room -= chars
                spent += chars
                kept.add(item[:2])
            else:
                left.append(item)
        return left

    if scores is not None:
        left = offer(sorted(others + passages, key=lambda it: -scores.get((it[2], it[3]), 0.0)))
    else:
        left = offer(others, limit=NON_CHUNK_SHARE * max(room, 0) if passages else None)
        left = offer(sorted(passages, key=lambda it: -it[5])) + left
        left = offer(left)
    dropped = [{"cardId": card_id, "loc": loc, "reason": "budget",
                "score": scores.get((card_id, loc)) if scores is not None else score}
               for _, _, card_id, loc, _, score in left]
    out = replace(ctx,
                  note_parts=[p for i, p in enumerate(ctx.note_parts) if ("note", i) in kept],
                  answer_parts=[p for i, p in enumerate(ctx.answer_parts) if ("answer", i) in kept])
    return Fit(out, [c for i, c in enumerate(chunks) if ("chunk", i) in kept], dropped)


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
