"""
The read set: which cards a chat reads. Server version of the prototype ctx()
(spec 5.2). compute_reads is pure over one BoardGraph snapshot; the context builder
feeds its result to the model, so it must match the frontend port exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from math import ceil

from app.agent.store.board_store import (
    CARD_LINK_JOINS, LIVE_DESK_ID, LIVE_PILE_ID, require_board, require_card)
from app.api.errors import ApiError

FILE_BACKED = frozenset({"pdf", "doc", "csv", "web", "video", "audio"})
NOTE_MAX_CHARS = 4000
ANSWER_MAX_CHARS = 3000
ANSWER_SUMMARY_MAX_CHARS = 1200
FILE_SNIPPET_CHARS = 900


@dataclass(frozen=True)
class CardRow:
    id: str
    kind: str
    source_type: str | None
    desk_id: str | None
    pile_id: str | None
    parent_id: str | None
    status: str | None
    data: dict
    created_at: datetime | None = None


def _by_created(cards: list[CardRow]) -> list[CardRow]:
    return sorted(cards, key=lambda c: (c.created_at is None, c.created_at or 0, c.id))


@dataclass
class BoardGraph:
    cards: dict[str, CardRow] = field(default_factory=dict)          # live only
    wires: list[tuple[str, str]] = field(default_factory=list)       # (source, chat), created_at order
    answers: list[tuple[str, str]] = field(default_factory=list)     # (from_chat, to_chat)
    has_answer: set[str] = field(default_factory=set)
    chunk_chars: dict[str, int] = field(default_factory=dict)
    answer_chars: dict[str, int] = field(default_factory=dict)
    summary_chars: dict[str, int] = field(default_factory=dict)

    def desk_sources(self, chat_id: str) -> list[CardRow]:
        return _by_created([c for c in self.cards.values()
                            if c.kind == "source" and c.desk_id == chat_id and c.pile_id is None])

    def members(self, pile_id: str) -> list[CardRow]:
        return _by_created([c for c in self.cards.values() if c.pile_id == pile_id])

    def wired_sources(self, chat_id: str) -> list[CardRow]:
        return [self.cards[s] for s, c in self.wires
                if c == chat_id and s in self.cards and self.cards[s].kind == "source"]

    def answer_chats(self, chat_id: str) -> list[CardRow]:
        return [self.cards[f] for f, t in self.answers
                if t == chat_id and f in self.cards and self.cards[f].kind == "chat"]


@dataclass(frozen=True)
class ReadItem:
    card_id: str
    how: str
    via: str
    inherited: bool
    hidden: bool
    state: str
    tokens: int

    def to_json(self) -> dict:
        return {"cardId": self.card_id, "how": self.how, "via": self.via,
                "inherited": self.inherited, "hidden": self.hidden,
                "state": self.state, "tokens": self.tokens}


@dataclass
class Reads:
    chat_id: str
    items: list[ReadItem]
    total_tokens: int

    def to_json(self) -> dict:
        return {"chatId": self.chat_id, "items": [i.to_json() for i in self.items],
                "totalTokens": self.total_tokens}

    def readable(self) -> list[ReadItem]:
        return [i for i in self.items if not i.hidden and i.state == "ready"]


def estimate_tokens(card: CardRow, g: BoardGraph, top_k: int) -> int:
    data = card.data or {}
    desc = len(data.get("desc") or "")
    if card.kind == "chat":
        return ceil((min(g.answer_chars.get(card.id, 0), ANSWER_MAX_CHARS)
                     + min(g.summary_chars.get(card.id, 0), ANSWER_SUMMARY_MAX_CHARS)) / 4)
    t = card.source_type
    if t == "note":
        return ceil((min(len(data.get("text") or ""), NOTE_MAX_CHARS) + desc) / 4)
    if t in FILE_BACKED:
        return ceil((min(g.chunk_chars.get(card.id, 0), top_k * FILE_SNIPPET_CHARS) + desc) / 4)
    if t == "image":
        return ceil((len(data.get("summary") or "") + desc) / 4)
    return 0


def _top_k(data: dict) -> int:
    adv = data.get("advanced")
    raw = adv.get("externalContextTopK") if isinstance(adv, dict) else None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw != raw:
        return 5
    return max(0, min(10, int(raw)))


def compute_reads(g: BoardGraph, chat_id: str) -> Reads:
    chat = g.cards[chat_id]
    top_k = _top_k(chat.data)

    def item(card, how, via):
        if card.kind == "chat":
            state = "ready" if card.id in g.has_answer else "waiting"
        else:
            state = card.status or "processing"
        tokens = estimate_tokens(card, g, top_k) if state == "ready" else 0
        return ReadItem(card.id, how, via, False, False, state, tokens)

    def expand(src, how, via):  # piles read as their members not switched off
        if src.source_type == "pile":
            return [item(m, how, via) for m in g.members(src.id) if not m.data.get("off")]
        return [item(src, how, via)]

    def dedupe_first(items):
        seen, out = set(), []
        for i in items:
            if i.card_id not in seen:
                seen.add(i.card_id)
                out.append(i)
        return out

    def own(c):
        out = []
        for s in g.desk_sources(c.id):
            out += expand(s, "desk", c.id)
        for s in g.wired_sources(c.id):
            if s.desk_id != c.id:
                out += expand(s, "wire", c.id)
        out += [item(a, "answer", c.id) for a in g.answer_chats(c.id)]
        return dedupe_first(out)

    def effective(c, seen, keep_hidden=False):
        items = own(c)
        parent = g.cards.get(c.parent_id or "")
        if (parent and parent.kind == "chat" and c.data.get("inheritDesk", True) is not False
                and parent.id not in seen and parent.id != c.id):
            have = {i.card_id for i in items}
            items += [replace(i, inherited=True)
                      for i in effective(parent, seen | {c.id}) if i.card_id not in have]
        hide = set(c.data.get("hide") or [])

        def is_hidden(i):  # hide applies to inherited items only; a hidden pile hides its members
            return i.inherited and (i.card_id in hide or (g.cards[i.card_id].pile_id or "") in hide)

        if keep_hidden:  # only the asking chat lists its hidden items
            return [replace(i, hidden=is_hidden(i)) for i in items]
        return [i for i in items if not is_hidden(i)]  # hidden upstream = gone for branches

    items = effective(chat, frozenset(), keep_hidden=True)
    total = sum(i.tokens for i in items if not i.hidden and i.state == "ready")
    return Reads(chat_id, items, total)


def load_board_graph(cur, user_email: str, board_id: str) -> BoardGraph:
    """Four queries: live cards (desk/pile links to deleted cards read as NULL, like CARD_SELECT), live strings, assistant-message lengths, chunk chars."""
    g = BoardGraph()
    cur.execute(
        f"SELECT n.id, n.kind, n.source_type, {LIVE_DESK_ID} AS desk_id, {LIVE_PILE_ID} AS pile_id, "
        "n.parent_node_id, n.status, n.data, n.created_at, "
        f"COALESCE(length(n.summary), 0) AS summary_len FROM nodes n {CARD_LINK_JOINS} "
        "WHERE n.canvas_id = %s AND n.user_email = %s AND n.kind IS NOT NULL AND n.deleted_at IS NULL "
        "ORDER BY n.created_at, n.id",
        (board_id, user_email),
    )
    for r in cur.fetchall():
        g.cards[r["id"]] = CardRow(
            id=r["id"], kind=r["kind"], source_type=r["source_type"], desk_id=r["desk_id"],
            pile_id=r["pile_id"], parent_id=r["parent_node_id"], status=r["status"],
            data=r["data"] or {}, created_at=r["created_at"])
        if r["kind"] == "chat":
            g.summary_chars[r["id"]] = int(r["summary_len"])

    cur.execute(
        "SELECT kind, from_node, to_node FROM edges "
        "WHERE canvas_id = %s AND user_email = %s AND kind IN ('wire','answer') "
        "AND deleted_at IS NULL ORDER BY created_at, id",
        (board_id, user_email),
    )
    for r in cur.fetchall():
        if r["from_node"] in g.cards and r["to_node"] in g.cards:
            (g.wires if r["kind"] == "wire" else g.answers).append((r["from_node"], r["to_node"]))

    cur.execute(
        "SELECT DISTINCT ON (node_id) node_id, length(content) AS n FROM messages "
        "WHERE node_id IN (SELECT id FROM nodes WHERE canvas_id = %s AND user_email = %s) "
        "AND role = 'assistant' AND NOT inherited "
        "ORDER BY node_id, position DESC",
        (board_id, user_email),
    )
    for r in cur.fetchall():
        if r["node_id"] in g.cards:
            g.has_answer.add(r["node_id"])
            g.answer_chars[r["node_id"]] = int(r["n"] or 0)

    cur.execute(
        "SELECT ef.node_id, COALESCE(sum(length(fc.chunk_text)), 0) AS n "
        "FROM external_files ef LEFT JOIN file_chunks fc ON fc.file_id = ef.id "
        "WHERE ef.node_id IN (SELECT id FROM nodes WHERE canvas_id = %s AND user_email = %s) "
        "GROUP BY ef.node_id",
        (board_id, user_email),
    )
    for r in cur.fetchall():
        if r["node_id"] in g.cards:
            g.chunk_chars[r["node_id"]] = int(r["n"])
    return g


def resolve_reads(store, user_email: str, board_id: str, chat_id: str) -> Reads:
    with store.tx() as cur:
        require_board(cur, board_id, user_email)
        card = require_card(cur, board_id, chat_id, user_email)
        if card["kind"] != "chat":
            raise ApiError(400, "invalid", "reads are defined for chats only")
        g = load_board_graph(cur, user_email, board_id)
    return compute_reads(g, chat_id)
