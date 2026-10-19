"""The prototype seed (pebble-desks-prototype.html:1265-1287) as an in-memory BoardGraph."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.agent.reads import BoardGraph, CardRow

NOTE_TEXT = "Costs ≈ $0.40 per active user a month. Never go below $8."
PDF_DESC = "Our main benchmark. Trust it over blog posts."
_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def prototype_graph(*, off=(), no_inherit=(), hide=None, extra_chats=None, extra_wires=(),
                    extra_answers=(), no_answer=(), status=None) -> BoardGraph:
    hide = hide or {}
    status = status or {}
    g = BoardGraph()
    n = [0]

    def add(cid, kind, *, source_type=None, st=None, desk=None, pile=None, parent=None, data=None):
        n[0] += 1
        d = dict(data or {})
        if cid in off:
            d["off"] = True
        if cid in no_inherit:
            d["inheritDesk"] = False
        if cid in hide:
            d["hide"] = list(hide[cid])
        g.cards[cid] = CardRow(
            id=cid, kind=kind, source_type=source_type, desk_id=desk, pile_id=pile,
            parent_id=parent, status=status.get(cid, st), data=d,
            created_at=_T0 + timedelta(seconds=n[0]))

    add("main", "chat")
    add("side", "chat", parent="main")
    add("second", "chat", parent="main")
    add("deep", "chat", parent="second")
    for cid, parent in (extra_chats or {}).items():
        add(cid, "chat", parent=parent)
    g.has_answer |= {"main", "side", "second", "deep"} | set(extra_chats or {})
    g.has_answer -= set(no_answer)
    for c in g.has_answer:
        g.answer_chars[c] = 100
        g.summary_chars[c] = 40

    add("video", "source", source_type="video", st="soon", desk="main")
    add("pdf", "source", source_type="pdf", st="ready", desk="main", data={"desc": PDF_DESC})
    g.chunk_chars["pdf"] = 5000
    add("note", "source", source_type="note", st="ready", desk="main", data={"text": NOTE_TEXT})
    add("web", "source", source_type="web", st="ready", desk="second")
    g.chunk_chars["web"] = 2000
    add("interviews", "source", source_type="pile", desk="second")
    for i in range(1, 5):
        add(f"iv{i}", "source", source_type="pdf", st="ready", pile="interviews")
        g.chunk_chars[f"iv{i}"] = 1000
    add("podcast", "source", source_type="audio", st="soon")
    add("csv", "source", source_type="csv", st="ready")
    g.chunk_chars["csv"] = 800
    add("shot", "source", source_type="image", st="soon", desk="deep")

    g.wires.append(("csv", "deep"))
    g.wires.extend(extra_wires)
    g.answers.append(("side", "deep"))
    g.answers.extend(extra_answers)
    return g
