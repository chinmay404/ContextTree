"""Context engine: what a desk full of sources sends (hybrid search, per-source limit, budget)."""
import json

import pytest

from app.agent.board_context import (
    BoardContext, ScoredChunk, build_board_context, fit_to_budget, gather_board_chunks, keep_per_source,
)
from app.agent.memory import extract_memory_facts
from app.agent.reads import resolve_reads
from app.agent.store.board_store import store
from tests.v3_support import _vec, db, new_id, requires_db, seed_board, seed_card, unit_emb, user


def _chunk(db, email, board, card, idx, loc, emb, text):
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
                    (fid, idx, text, json.dumps({"loc": loc}), _vec(emb)))


def _pdf(db, email, board, chat, title):
    return seed_card(db, email, board, kind="source", source_type="pdf", status="ready", desk_id=chat,
                     data={"title": title})


@pytest.fixture
def desk(db, user):
    b = seed_board(db, user)
    q = seed_card(db, user, b, kind="chat", data={})
    return dict(b=b, q=q, email=user)


def _gather(desk, emb, text, per_source=2):
    reads = resolve_reads(store, desk["email"], desk["b"], desk["q"])
    ids = [i.card_id for i in reads.readable()]
    with store.tx() as cur:
        return gather_board_chunks(cur, desk["email"], ids, emb, text, per_source)


@requires_db
def test_one_big_source_cannot_take_every_slot(db, desk):
    big = _pdf(db, desk["email"], desk["b"], desk["q"], "Big")
    small = _pdf(db, desk["email"], desk["b"], desk["q"], "Small")
    for i in range(6):
        _chunk(db, desk["email"], desk["b"], big, i, f"b{i}", unit_emb(1), f"big {i}")
    _chunk(db, desk["email"], desk["b"], small, 0, "s0", unit_emb(2), "small")
    chunks = _gather(desk, unit_emb(1), None)
    assert {c.card_id for c in chunks} == {big, small}
    assert sum(c.card_id == big for c in chunks) == 2


@requires_db
def test_an_exact_name_is_found_when_vector_search_misses_it(db, desk):
    pdf = _pdf(db, desk["email"], desk["b"], desk["q"], "Report")
    _chunk(db, desk["email"], desk["b"], pdf, 0, "near", unit_emb(1), "general market commentary")
    _chunk(db, desk["email"], desk["b"], pdf, 1, "mid", unit_emb(2), "regional growth outlook")
    _chunk(db, desk["email"], desk["b"], pdf, 2, "exact", unit_emb(3), "Acme Bundesliga licence fee")
    chunks = _gather(desk, unit_emb(1), "What is the Bundesliga licence?", per_source=2)
    assert "exact" in [c.loc for c in chunks]


@requires_db
def test_documents_still_found_when_embedding_failed(db, desk):
    pdf = _pdf(db, desk["email"], desk["b"], desk["q"], "Report")
    _chunk(db, desk["email"], desk["b"], pdf, 0, "a", unit_emb(1), "the Bundesliga licence")
    _chunk(db, desk["email"], desk["b"], pdf, 1, "b", unit_emb(2), "something else")
    chunks = _gather(desk, [0.0] * 768, "Bundesliga licence?")
    assert [c.loc for c in chunks] == ["a"]


@requires_db
def test_switched_off_excerpt_stays_out_of_keyword_search(db, desk):
    pdf = seed_card(db, desk["email"], desk["b"], kind="source", source_type="pdf", status="ready",
                    desk_id=desk["q"], data={"title": "R", "excerpts": [{"loc": "a", "on": False}]})
    _chunk(db, desk["email"], desk["b"], pdf, 0, "a", unit_emb(1), "Bundesliga licence")
    reads = resolve_reads(store, desk["email"], desk["b"], desk["q"])
    ctx = build_board_context(store, reads, desk["email"], [0.0] * 768, 2, search_text="Bundesliga")
    assert ctx.chunks == []


def _c(card, loc, score, chars=100):
    return ScoredChunk(card, loc, "x" * chars, score)


def test_keep_per_source_keeps_the_best_of_each():
    chunks = [_c("a", "1", 0.9), _c("a", "2", 0.8), _c("a", "3", 0.7), _c("b", "1", 0.1)]
    assert [(c.card_id, c.loc) for c in keep_per_source(chunks, 2)] == [("a", "1"), ("a", "2"), ("b", "1")]


def _ctx(notes=(), answers=()):
    ctx = BoardContext()
    for cid, chars in notes:
        ctx.note_parts.append((cid, f"[source | {cid}]", "n" * chars, "q"))
    for cid, chars in answers:
        ctx.answer_parts.append((cid, [f"[answer | {cid}]"], "a" * chars, "q"))
    return ctx


def test_many_notes_cannot_push_every_file_passage_out():
    ctx = _ctx(notes=[(f"n{i}", 4000) for i in range(30)])
    fit = fit_to_budget(ctx, [_c("pdf", "1", 0.02, 900)], fixed_chars=0, budget_tokens=4000)
    assert [c.card_id for c in fit.chunks] == ["pdf"]
    assert len(fit.ctx.note_parts) < 30


def test_nothing_is_cut_when_it_all_fits():
    ctx = _ctx(notes=[("n", 100)], answers=[("a", 100)])
    fit = fit_to_budget(ctx, [_c("pdf", "1", 0.02)], fixed_chars=0, budget_tokens=4000)
    assert len(fit.ctx.note_parts) == 1 and len(fit.ctx.answer_parts) == 1 and len(fit.chunks) == 1
    assert fit.dropped == []


def test_everything_dropped_is_listed():
    ctx = _ctx(notes=[("n", 4000)])
    fit = fit_to_budget(ctx, [_c("pdf", "1", 0.5, 400), _c("pdf", "2", 0.1, 4000)], fixed_chars=0, budget_tokens=500)
    assert {(d["cardId"], d["loc"]) for d in fit.dropped} >= {("n", None), ("pdf", "2")}
    assert all(d["reason"] == "budget" for d in fit.dropped)


def test_no_budget_means_no_cut():
    ctx = _ctx(notes=[("n", 40000)])
    fit = fit_to_budget(ctx, [_c("pdf", "1", 0.5)], fixed_chars=0, budget_tokens=None)
    assert len(fit.ctx.note_parts) == 1 and len(fit.chunks) == 1 and fit.dropped == []


def test_jev_scores_rank_notes_and_passages_together():
    ctx = _ctx(notes=[("weak", 800)], answers=[("strong", 800)])
    chunks = [_c("pdf", "1", 0.02, 800)]
    scores = {("weak", None): 0.1, ("strong", None): 0.9, ("pdf", "1"): 0.8}
    fit = fit_to_budget(ctx, chunks, fixed_chars=0, budget_tokens=420, scores=scores)
    assert [p[0] for p in fit.ctx.answer_parts] == ["strong"]
    assert [c.loc for c in fit.chunks] == ["1"]
    assert fit.ctx.note_parts == []


def test_newest_decision_survives_the_cap():
    msgs = [{"role": "user", "text": f"let's use option {i} for storage"} for i in range(1, 9)]
    facts = extract_memory_facts({}, msgs)
    assert facts["decisions"][-1] == "let's use option 8 for storage"
    assert len(facts["decisions"]) == 6


def test_saying_you_are_confused_is_not_a_constraint():
    msgs = [{"role": "user", "text": "I don't know what this means"},
            {"role": "user", "text": "this only happens on Tuesdays"},
            {"role": "user", "text": "never use more than two colours"}]
    assert extract_memory_facts({}, msgs)["constraints"] == ["never use more than two colours"]
