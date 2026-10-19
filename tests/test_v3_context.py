"""Task 9: the V3 context builder feeds the prompt from the server read set."""
import json

import pytest

from app.agent.board_context import (
    BoardContext, ScoredChunk, build_board_context, render_source_blocks, source_header,
    trim_chunks_to_budget,
)
from app.agent.memory import build_memory_context_block
from app.agent.reads import resolve_reads
from app.agent.store.board_store import store
from tests.v3_support import (
    _vec, db, new_id, requires_db, seed_board, seed_card, seed_edge, seed_message, unit_emb, user,
)

DESC = "Our main benchmark. Trust it over blog posts."


def _scaled(a, b, wa, wb):
    v = [0.0] * 768
    v[a], v[b] = wa, wb
    return v


def _chunk(db, email, board, card, idx, loc, emb, text=None):
    with db.cursor() as cur:
        cur.execute("SELECT id FROM external_files WHERE node_id = %s", (card,))
        row = cur.fetchone()
        if row:
            fid = row[0]
        else:
            fid = new_id("f")
            cur.execute("INSERT INTO external_files (id, node_id, canvas_id, user_email, file_name, "
                        "file_type) VALUES (%s,%s,%s,%s,'f','pdf')", (fid, card, board, email))
        cur.execute("INSERT INTO file_chunks (file_id, chunk_index, chunk_text, metadata, embedding) "
                    "VALUES (%s,%s,%s,%s::jsonb,%s::vector)",
                    (fid, idx, text or f"text {loc}", json.dumps({"loc": loc} if loc else {}), _vec(emb)))


@pytest.fixture
def board(db, user):
    b = seed_board(db, user)
    q = seed_card(db, user, b, kind="chat", data={"advanced": {"externalContextTopK": 2}})
    pdf = seed_card(db, user, b, kind="source", source_type="pdf", status="ready", desk_id=q, data={
        "title": "Pricing survey 2025", "desc": DESC,
        "excerpts": [{"loc": "p. 19", "text": "Methodology", "on": False}]})
    note = seed_card(db, user, b, kind="source", source_type="note", status="ready", desk_id=q,
                     data={"title": "My note", "text": "N" * 5000})
    web2 = seed_card(db, user, b, kind="source", source_type="web", status="ready", data={"title": "Blog"})
    a = seed_card(db, user, b, kind="chat", data={"name": "Side"})
    w = seed_card(db, user, b, kind="chat", data={"name": "Waiting"})
    with db.cursor() as cur:
        cur.execute("UPDATE nodes SET summary = %s WHERE id = %s", ("S" * 2000, a))
    seed_message(db, user, a, "assistant", "A" * 5000, 1)
    seed_edge(db, user, b, a, q, "answer")
    seed_edge(db, user, b, w, q, "answer")
    _chunk(db, user, b, pdf, 0, "p. 19", unit_emb(1))
    _chunk(db, user, b, pdf, 1, "p. 3", _scaled(1, 2, 0.99, 0.01))
    _chunk(db, user, b, pdf, 2, "p. 7", unit_emb(2))
    _chunk(db, user, b, web2, 0, "w", unit_emb(1))
    return dict(b=b, q=q, pdf=pdf, note=note, web2=web2, a=a, w=w, email=user)


def _ctx(bd, emb=None, k=2):
    reads = resolve_reads(store, bd["email"], bd["b"], bd["q"])
    return build_board_context(store, reads, bd["email"], emb or unit_emb(1), k)


@requires_db
def test_file_chunks_limited_to_read_files(board):
    ctx = _ctx(board)
    assert ctx.chunks and all(c.card_id != board["web2"] for c in ctx.chunks)


@requires_db
def test_switched_off_excerpt_chunk_skipped(board):
    assert [c.loc for c in _ctx(board).chunks] == ["p. 3", "p. 7"]


@requires_db
def test_desc_on_header_line(board):
    ctx = _ctx(board)
    assert ctx.headers[board["pdf"]] == (
        "[source | PDF | Pricing survey 2025 | for: Our main benchmark. Trust it over blog posts.]")


@requires_db
def test_note_whole_text_capped_4000(board):
    ctx = _ctx(board)
    assert len(ctx.notes) == 1
    header, body = ctx.notes[0].split("\n", 1)
    assert header.startswith("[source | Note | My note")
    assert 0 < len(body) <= 4000


@requires_db
def test_hidden_and_not_ready_sources_not_sent(board, db):
    with db.cursor() as cur:
        cur.execute("UPDATE nodes SET status = 'processing' WHERE id = %s", (board["pdf"],))
    ctx = _ctx(board)
    assert ctx.chunks == [] and len(ctx.notes) == 1
    # hide applies to inherited items only: a branch hides the parent's note
    q2 = seed_card(db, board["email"], board["b"], kind="chat", parent_id=board["q"],
                   data={"hide": [board["note"]]})
    reads = resolve_reads(store, board["email"], board["b"], q2)
    assert build_board_context(store, reads, board["email"], unit_emb(1), 2).notes == []


@requires_db
def test_linked_answers_block(board):
    la = _ctx(board).linked_answers
    assert la.startswith("<LINKED_ANSWERS>\n[answer | Side]\nSummary: ")
    assert la.endswith("</LINKED_ANSWERS>")
    lines = la.split("\n")
    assert len(lines[2][len("Summary: "):]) <= 1200
    assert 0 < len(lines[3]) <= 3000
    assert "Waiting" not in la


@requires_db
def test_no_ready_answers_gives_none(board, db):
    with db.cursor() as cur:
        cur.execute("DELETE FROM messages WHERE node_id = %s", (board["a"],))
    assert _ctx(board).linked_answers is None


@requires_db
def test_v3_blocks_survive_zero_query_embedding(board):
    ctx = _ctx(board, emb=[0.0] * 768)
    assert len(ctx.notes) == 1 and ctx.linked_answers and ctx.chunks == []


def _cs():
    return [ScoredChunk("a", "p. 1", "x" * 400, 0.9), ScoredChunk("b", "p. 2", "y" * 400, 0.2),
            ScoredChunk("c", "p. 3", "z" * 400, 0.5)]


def test_budget_trims_lowest_score_first():
    assert [c.card_id for c in trim_chunks_to_budget(_cs(), fixed_chars=400, budget_tokens=320)] == ["a", "c"]


def test_no_budget_no_trim():
    cs = _cs()
    assert trim_chunks_to_budget(cs, 10**9, None) == cs


def test_budget_smaller_than_fixed_drops_all():
    assert trim_chunks_to_budget(_cs(), fixed_chars=4000, budget_tokens=100) == []


def test_source_header_fallbacks():
    assert source_header({"source_type": "pdf", "data": {"fileName": "a.pdf"}}) == "[source | PDF | a.pdf]"
    assert source_header({"source_type": "note", "data": {"text": "t" * 60}}) == f"[source | Note | {'t' * 40}]"
    assert source_header({"source_type": "web", "data": {}}) == "[source | Web page | Untitled]"


def test_render_source_blocks_groups_by_card():
    ctx = BoardContext([], [], {"a": "[H a]", "b": "[H b]"}, None)
    out = render_source_blocks(ctx, [ScoredChunk("a", "p. 1", "one", 1), ScoredChunk("b", None, "two", 1),
                                     ScoredChunk("a", "p. 2", "three", 1)])
    assert out == ["[H a]\n(p. 1) one\n(p. 2) three", "[H b]\ntwo"]


def test_memory_block_includes_linked_answers():
    out = build_memory_context_block(summary="", memory_facts={}, context=[], external_context=None,
                                     linked_answers="<LINKED_ANSWERS>\nx\n</LINKED_ANSWERS>")
    assert "<LINKED_ANSWERS>\nx\n</LINKED_ANSWERS>" in out


def _graph(monkeypatch, emb):
    from app.agent import main
    g = main.getGraphResponse.__new__(main.getGraphResponse)
    g.mongo_store = store
    monkeypatch.setattr(main, "get_embedding", lambda q: emb)
    monkeypatch.setattr(main, "board_store", store)
    return g


@requires_db
def test_build_retrieved_context_v3_ignores_context_node_ids(board, monkeypatch):
    g = _graph(monkeypatch, unit_emb(1))
    calls = []
    monkeypatch.setattr(store, "get_related_file_context", lambda *a, **k: calls.append(1) or [])
    reads = resolve_reads(store, board["email"], board["b"], board["q"])
    ctx, ext, hits, la = g._build_retrieved_context(
        "hello", board["email"], board["q"], context_node_ids=["zzz"], board_reads=reads)
    assert calls == []
    assert "[source | PDF | Pricing survey 2025 | for: " in ext
    assert la.startswith("<LINKED_ANSWERS>")


@requires_db
def test_build_retrieved_context_v3_zero_embedding_keeps_4_tuple(board, monkeypatch):
    g = _graph(monkeypatch, [0.0] * 768)
    reads = resolve_reads(store, board["email"], board["b"], board["q"])
    out = g._build_retrieved_context("hello", board["email"], board["q"], board_reads=reads)
    assert len(out) == 4
    ctx, ext, hits, la = out
    assert ctx == [] and hits == [] and "[source | Note | My note" in ext and "p. 3" not in ext
    assert la.startswith("<LINKED_ANSWERS>")


def test_build_retrieved_context_v2_zero_embedding_4_tuple(monkeypatch):
    g = _graph(monkeypatch, [0.0] * 768)
    assert g._build_retrieved_context("hello", "u", "t") == ([], None, [], None)


@requires_db
def test_build_retrieved_context_budget_trims_chunks(board, monkeypatch):
    g = _graph(monkeypatch, unit_emb(1))
    reads = resolve_reads(store, board["email"], board["b"], board["q"])
    out = g._build_retrieved_context("hello", board["email"], board["q"], board_reads=reads,
                                     budget_tokens=1, fixed_chars=0)
    assert "p. 3" not in (out[1] or "") and "[source | Note" in out[1]
