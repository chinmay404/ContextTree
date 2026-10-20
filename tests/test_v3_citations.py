"""Citations plan E2: numbered passages, the citation instruction, the citations frame, persistence."""
import json
from pathlib import Path

import pytest

from app.agent.board_context import BoardContext, ScoredChunk, build_board_context, number_passages
from app.agent.citations import (
    CITE_INSTRUCTION, QUOTE_MAX_CHARS, extract_citations, neutralise_markers, strip_markers,
)
from app.agent.reads import resolve_reads
from app.agent.store.board_store import store
from tests.v3_support import (  # noqa: F401
    _vec, api, db, new_id, requires_db, seed_board, seed_card, seed_edge, seed_message, unit_emb, user,
)

SQL = (Path(__file__).resolve().parent.parent / "db/migrations/010_message_citations.sql").read_text(
    encoding="utf-8")


# ── numbering ────────────────────────────────────────────────────────────────

def _ctx():
    ctx = BoardContext(headers={"pa": "[H pa]", "pb": "[H pb]"})
    ctx.notes = ["[H n]\nnote body"]
    ctx.note_parts = [("n1", "[H n]", "note body", "note body")]
    ctx.answer_parts = [("ch", ["[answer | Side]", "Summary: s"], "the answer", "the answer")]
    ctx.linked_answers = "<LINKED_ANSWERS>\n[answer | Side]\nSummary: s\nthe answer\n</LINKED_ANSWERS>"
    return ctx


def test_number_passages_prompt_order_notes_chunks_answers():
    chunks = [ScoredChunk("pa", "p. 4", "alpha", 0.9, page=4), ScoredChunk("pb", None, "beta", 0.8),
              ScoredChunk("pa", "p. 5", "gamma", 0.7, page=5)]
    out = number_passages(_ctx(), chunks)
    assert out.notes == ["[H n]\n[1] note body"]
    # chunks are grouped by card in the prompt, so numbering follows the grouping
    assert out.sources == ["[H pa]\n[2] (p. 4) alpha\n[3] (p. 5) gamma", "[H pb]\n[4] beta"]
    assert out.linked_answers == (
        "<LINKED_ANSWERS>\n[answer | Side]\nSummary: s\n[5] the answer\n</LINKED_ANSWERS>")
    assert out.passages == {
        1: {"n": 1, "cardId": "n1", "loc": None, "page": None, "quote": "note body"},
        2: {"n": 2, "cardId": "pa", "loc": "p. 4", "page": 4, "quote": "alpha"},
        3: {"n": 3, "cardId": "pa", "loc": "p. 5", "page": 5, "quote": "gamma"},
        4: {"n": 4, "cardId": "pb", "loc": None, "page": None, "quote": "beta"},
        5: {"n": 5, "cardId": "ch", "loc": None, "page": None, "quote": "the answer"},
    }


def test_number_passages_empty():
    out = number_passages(BoardContext(), [])
    assert out.notes == [] and out.sources == [] and out.linked_answers is None and out.passages == {}


def test_long_chunk_quote_is_the_sent_text_without_the_clip_ellipsis():
    out = number_passages(BoardContext(headers={"a": "[H]"}), [ScoredChunk("a", None, "w" * 5000, 1.0)])
    q = out.passages[1]["quote"]
    assert len(q) == 899 and set(q) == {"w"}  # chunks are sent clipped to 900 incl. the ellipsis
    assert out.sources[0].endswith("…")  # the prompt still shows the clip


@requires_db
def test_long_note_quote_capped_at_1200(db, user):
    b = seed_board(db, user)
    q = seed_card(db, user, b, kind="chat")
    seed_card(db, user, b, kind="source", source_type="note", status="ready", desk_id=q,
              data={"title": "Long", "text": "n" * 5000})
    ctx = build_board_context(store, resolve_reads(store, user, b, q), user, unit_emb(1), 2)
    out = number_passages(ctx, [])
    assert len(out.passages[1]["quote"]) == QUOTE_MAX_CHARS == 1200
    assert len(out.notes[0].split("\n", 1)[1]) == len("[1] ") + 4000


# ── the parser ───────────────────────────────────────────────────────────────

P = {n: {"n": n, "cardId": f"c{n}", "loc": None, "page": None, "quote": f"q{n}"} for n in (1, 2, 3)}


@pytest.mark.parametrize("text,want", [
    ("Prices rose [2]. Costs fell [1].", [1, 2]),
    ("Unknown [7] and [0] are dropped [3].", [3]),
    ("Grouped [2, 3] and [1,2] work.", [1, 2, 3]),
    ("Adjacent [3][1] work.", [1, 3]),
    ("Repeated [2] then [2] again [2].", [2]),
    ("No markers at all.", []),
    ("```\nx = a[1]\n```\nReal claim [2].", [2]),
    ("~~~python\nb[3]\n~~~\nok", []),
    ("Inline `a[1]` and ``b[3]`` code [2].", [2]),
    ("Unclosed fence [2]\n```\nc[3]", [2]),
    ("<think>maybe [1]</think>Answer [3].", [3]),
    ("Spaced [ 2 ] marker.", [2]),
    # a stray backtick must not open a code span across paragraphs (review M2)
    ("Press the ` key.\n\nClaim one [1].\n\nClaim two [2].\n\nRun `ls` [3].", [1, 2, 3]),
])
def test_extract_citations(text, want):
    assert [c["n"] for c in extract_citations(text, P)] == want


@pytest.mark.parametrize("text,want", [
    ("Side answer [4] with an old marker.", "Side answer with an old marker."),
    ("Claim [2]. Next [1][3], done [2, 3].", "Claim. Next, done."),
    ("[1] Leading marker.", "Leading marker."),
    ("[1][2] Two leading.\n[3] Next line.", "Two leading.\nNext line."),
    ("Line one [1]\nline two", "Line one\nline two"),
    ("Code `a[1]` stays [2].", "Code `a[1]` stays."),
])
def test_strip_markers_leaves_no_double_spaces(text, want):
    assert strip_markers(text) == want


def test_neutralise_markers_rewrites_bracketed_numbers_outside_code():
    assert neutralise_markers("see [7] and [1, 2] but `x[3]` and [a]") == "see (7) and (1, 2) but `x[3]` and [a]"


def test_extract_citations_returns_map_entries():
    assert extract_citations("x [2]", P) == [P[2]]
    assert extract_citations("x [1]", {}) == []


# ── fake LLM ─────────────────────────────────────────────────────────────────

def test_fake_llm_cites_after_first_sentence_only_when_passages_sent(monkeypatch):
    from langchain_core.messages import HumanMessage, SystemMessage

    from app.agent.helpers.get_llm import get_llm
    from app.core.config import settings
    monkeypatch.setattr(settings, "ENGINE_FAKE_LLM", True)
    llm = get_llm(None)
    with_passages = [SystemMessage(content="persona"), SystemMessage(content=CITE_INSTRUCTION),
                     HumanMessage(content="what. then more")]
    assert llm.invoke(with_passages).content == "Fake answer to: what.[1] then more"
    no_stop = [SystemMessage(content=CITE_INSTRUCTION), HumanMessage(content="hi there")]
    assert llm.invoke(no_stop).content == "Fake answer to: hi there[1]"
    assert llm.invoke([HumanMessage(content="hi there")]).content == "Fake answer to: hi there"


def test_assistant_node_adds_citation_instruction_only_when_given(monkeypatch):
    from langchain_core.messages import HumanMessage

    from app.agent.nodes import assistant_node
    from app.core.config import settings
    monkeypatch.setattr(settings, "ENGINE_FAKE_LLM", True)
    seen = []
    real = assistant_node.get_llm

    def spy(*a, **k):
        llm = real(*a, **k)
        orig = llm._prime

        def prime(messages):
            seen.append(messages)
            return orig(messages)
        object.__setattr__(llm, "_prime", prime)
        return llm
    monkeypatch.setattr(assistant_node, "get_llm", spy)
    node = assistant_node.AgentNodes()
    out = node.assistant({"messages": [HumanMessage(content="q")], "count": 0, "user_id": "u@x.org",
                          "citation_note": CITE_INSTRUCTION})
    assert out["messages"][-1].content == "Fake answer to: q[1]"
    systems = [m.content for m in seen[-1] if m.type == "system"]
    assert systems.count(CITE_INSTRUCTION) == 1
    out = node.assistant({"messages": [HumanMessage(content="q")], "count": 0, "user_id": "u@x.org"})
    assert out["messages"][-1].content == "Fake answer to: q"
    assert CITE_INSTRUCTION not in [m.content for m in seen[-1]]


@pytest.fixture
def llm_inputs(monkeypatch):
    """Every message list the (fake) assistant LLM is called with."""
    from app.agent.nodes import assistant_node
    from app.core.config import settings
    monkeypatch.setattr(settings, "ENGINE_FAKE_LLM", True)
    seen = []
    real = assistant_node.get_llm

    def spy(*a, **k):
        llm = real(*a, **k)
        orig = llm._prime

        def prime(messages):
            seen.append(list(messages))
            return orig(messages)
        object.__setattr__(llm, "_prime", prime)
        return llm
    monkeypatch.setattr(assistant_node, "get_llm", spy)
    return seen


def test_v3_turn_strips_old_markers_from_every_past_answer_copy(llm_inputs):
    from langchain_core.messages import AIMessage, HumanMessage

    from app.agent.nodes.assistant_node import AgentNodes
    old = AIMessage(content="Old answer [1] said so [2, 3].", id="a1")
    state = {
        "messages": [HumanMessage(content="first", id="u1"), old, HumanMessage(content="next [5]?", id="u2")],
        "count": 0, "user_id": "u@x.org", "citation_note": CITE_INSTRUCTION,
        "context": ["[sim 1 | role=assistant | score=0.900] id=m1: prior [4] text"],
        "summary": "Earlier we cited [6] twice.",
        "external_context": "[source | Note | N]\n[1] passage text",
        "linked_answers": "<LINKED_ANSWERS>\n[answer | X]\n[2] linked body\n</LINKED_ANSWERS>",
    }
    AgentNodes().assistant(state)
    sent = llm_inputs[-1]
    ai = [m for m in sent if m.type == "ai"]
    assert [m.content for m in ai] == ["Old answer said so."]
    memory = sent[1].content
    assert "id=m1: prior text" in memory and "Earlier we cited twice." in memory
    assert "[1] passage text" in memory and "[2] linked body" in memory  # this turn's numbers stay
    assert [m.content for m in sent if m.type == "human"] == ["first", "next [5]?"]  # user text untouched
    assert old.content == "Old answer [1] said so [2, 3]."  # copies only: state is not modified

    state["citation_note"] = None  # no numbered passages this turn: nothing is rewritten
    AgentNodes().assistant(state)
    assert [m.content for m in llm_inputs[-1] if m.type == "ai"] == ["Old answer [1] said so [2, 3]."]


# ── DB: numbering after budget trimming ──────────────────────────────────────

def _chunk(db, email, board, card, idx, loc, emb, text, page=None):
    with db.cursor() as cur:
        cur.execute("SELECT id FROM external_files WHERE node_id = %s", (card,))
        row = cur.fetchone()
        if row:
            fid = row[0]
        else:
            fid = new_id("f")
            cur.execute("INSERT INTO external_files (id, node_id, canvas_id, user_email, file_name, "
                        "file_type) VALUES (%s,%s,%s,%s,'f','pdf')", (fid, card, board, email))
        meta = {"loc": loc}
        if page is not None:
            meta["page"] = page
        cur.execute("INSERT INTO file_chunks (file_id, chunk_index, chunk_text, metadata, embedding) "
                    "VALUES (%s,%s,%s,%s::jsonb,%s::vector)", (fid, idx, text, json.dumps(meta), _vec(emb)))


@pytest.fixture
def board(db, user):
    b = seed_board(db, user)
    q = seed_card(db, user, b, kind="chat", data={"advanced": {"externalContextTopK": 2}})
    pdf = seed_card(db, user, b, kind="source", source_type="pdf", status="ready", desk_id=q,
                    data={"title": "Survey"})
    note = seed_card(db, user, b, kind="source", source_type="note", status="ready", desk_id=q,
                     data={"title": "N", "text": "Note text [7]. Second sentence."})
    side = seed_card(db, user, b, kind="chat", data={"name": "Side"})
    with db.cursor() as cur:
        cur.execute("UPDATE nodes SET summary = 'Side summary [3] here.' WHERE id = %s", (side,))
    seed_message(db, user, side, "assistant", "Side answer [4] with an old marker.", 1)
    seed_edge(db, user, b, side, q, "answer")
    _chunk(db, user, b, pdf, 0, "p. 4", unit_emb(1), "Close chunk text [12].", page=3)
    _chunk(db, user, b, pdf, 1, "p. 9", [0.6 if i in (1, 2) else 0.0 for i in range(768)], "Far chunk.", page=8)
    return dict(b=b, q=q, pdf=pdf, note=note, side=side, email=user)


def _graph(monkeypatch, emb):
    from app.agent import main
    g = main.getGraphResponse.__new__(main.getGraphResponse)
    g.mongo_store = store
    monkeypatch.setattr(main, "get_embedding", lambda q: emb)
    monkeypatch.setattr(main, "board_store", store)
    return g


@requires_db
def test_build_context_reads_page_one_based(board):
    reads = resolve_reads(store, board["email"], board["b"], board["q"])
    ctx = build_board_context(store, reads, board["email"], unit_emb(1), 2)
    assert [(c.loc, c.page) for c in ctx.chunks] == [("p. 4", 4), ("p. 9", 9)]


@requires_db
def test_retrieved_context_numbers_what_is_sent(board, monkeypatch):
    g = _graph(monkeypatch, unit_emb(1))
    reads = resolve_reads(store, board["email"], board["b"], board["q"])
    passages = {}
    ctx, ext, hits, la = g._build_retrieved_context("hello", board["email"], board["q"], board_reads=reads,
                                                    user_email=board["email"], passages=passages)
    # source footnotes are rewritten in the prompt copy only (review M3)
    assert "\n[1] Note text (7). Second sentence." in ext and "[7]" not in ext
    assert "[2] (p. 4) Close chunk text (12)." in ext and "[3] (p. 9) Far chunk." in ext
    # the other chat's own markers are removed from its answer and its summary (review I1, M8)
    assert "\nSummary: Side summary here.\n[4] Side answer with an old marker." in la
    assert [(p["cardId"], p["loc"], p["page"]) for p in passages.values()] == [
        (board["note"], None, None), (board["pdf"], "p. 4", 4), (board["pdf"], "p. 9", 9),
        (board["side"], None, None)]
    # quotes keep the stored text
    assert passages[1]["quote"] == "Note text [7]. Second sentence."
    assert passages[2]["quote"] == "Close chunk text [12]."
    assert passages[4]["quote"] == "Side answer [4] with an old marker."


@requires_db
def test_trimmed_chunks_get_no_number(board, monkeypatch):
    g = _graph(monkeypatch, unit_emb(1))
    reads = resolve_reads(store, board["email"], board["b"], board["q"])
    passages = {}
    _, ext, _, la = g._build_retrieved_context("hello", board["email"], board["q"], board_reads=reads,
                                               user_email=board["email"], budget_tokens=1, passages=passages)
    assert "(p. 4)" not in ext and "(p. 9)" not in ext
    assert [p["cardId"] for p in passages.values()] == [board["note"], board["side"]]
    assert "[2] Side answer" in la


@requires_db
def test_zero_embedding_still_numbers_notes_and_answers(board, monkeypatch):
    g = _graph(monkeypatch, [0.0] * 768)
    reads = resolve_reads(store, board["email"], board["b"], board["q"])
    passages = {}
    out = g._build_retrieved_context("hello", board["email"], board["q"], board_reads=reads,
                                     user_email=board["email"], passages=passages)
    assert len(out) == 4 and "[1] Note text (7)." in out[1] and "[2] Side answer" in out[3]
    assert sorted(passages) == [1, 2]


def test_v2_context_unnumbered_and_passages_untouched(monkeypatch):
    g = _graph(monkeypatch, [0.0] * 768)
    passages = {}
    assert g._build_retrieved_context("hello", "u", "t", passages=passages) == ([], None, [], None)
    assert passages == {}


# ── migration 010 ────────────────────────────────────────────────────────────

def _citations_col(cur):
    cur.execute("SELECT data_type, is_nullable FROM information_schema.columns WHERE table_schema = "
                "current_schema() AND table_name = 'messages' AND column_name = 'citations'")
    return cur.fetchall()


@requires_db
def test_010_recorded_and_column_is_nullable_jsonb(db):
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM schema_migrations WHERE filename = '010_message_citations.sql'")
        assert cur.fetchone()[0] == 1
        assert _citations_col(cur) == [("jsonb", "YES")]


@requires_db
def test_010_twice_is_noop(db):
    with db.cursor() as cur:
        cur.execute(SQL)
        before = _citations_col(cur)
        cur.execute(SQL)
        assert _citations_col(cur) == before == [("jsonb", "YES")]


@requires_db
def test_010_sets_local_lock_timeout_inside_its_transaction():
    import psycopg2

    from tests.v3_support import TEST_DB
    conn = psycopg2.connect(TEST_DB)
    try:
        with conn.cursor() as cur:
            cur.execute(SQL)
            cur.execute("SHOW lock_timeout")
            assert cur.fetchone()[0] == "5s"
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute("SHOW lock_timeout")
            assert cur.fetchone()[0] == "0"
    finally:
        conn.close()


@requires_db
def test_010_applies_on_v2_frontend_shaped_messages(db):
    import uuid
    schema = f"v2msg_{uuid.uuid4().hex[:8]}"
    with db.cursor() as cur:
        cur.execute(f"CREATE SCHEMA {schema}")
        try:
            cur.execute(f"SET search_path TO {schema}")
            cur.execute("create table messages (id text primary key, node_id text not null, "
                        "role text not null, content text, created_at timestamptz not null default now())")
            cur.execute("insert into messages (id, node_id, role, content) values ('m1', 'n', 'user', 'x')")
            cur.execute(SQL)
            cur.execute(SQL)
            assert _citations_col(cur) == [("jsonb", "YES")]
            cur.execute("SELECT citations FROM messages WHERE id = 'm1'")
            assert cur.fetchone()[0] is None
        finally:
            cur.execute("RESET search_path")
            cur.execute(f"DROP SCHEMA {schema} CASCADE")


# ── end to end: the stream, the frame order, persistence ─────────────────────

def _frames(text: str) -> list:
    out = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block.startswith("data: "):
            continue
        raw = block[len("data: "):]
        out.append(raw if raw == "[DONE]" else json.loads(raw))
    return out


@pytest.fixture
def fake_mode(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "ENGINE_FAKE_LLM", True)


def _turn(api, node, **extra):
    r = api.post("/chat/stream", json={"message": "What does it say? Tell me.", "message_id": new_id("m"),
                                       "nodeId": node, "model": "z-ai/glm-5.2", **extra})
    assert r.status_code == 200, r.text
    return _frames(r.text)


def _kinds(frames):
    return ["done" if f == "[DONE]" else next(iter(f)) for f in frames]


@requires_db
def test_v3_stream_sends_citations_before_done_and_stores_them(api, db, user, fake_mode):
    b = seed_board(db, user)
    q = seed_card(db, user, b, kind="chat")
    note = seed_card(db, user, b, kind="source", source_type="note", status="ready", desk_id=q,
                     data={"title": "N", "text": "The sky is green here."})
    frames = _turn(api, q, boardId=b)
    kinds = _kinds(frames)
    assert kinds.count("citations") == 1 and kinds[-1] == "done" and "error" not in kinds
    ci = kinds.index("citations")
    assert set(kinds[:ci]) == {"message"} and kinds.index("summary") > ci
    text = "".join(f["message"] for f in frames if isinstance(f, dict) and "message" in f)
    assert text == "Fake answer to: What does it say?[1] Tell me."
    want = [{"n": 1, "cardId": note, "loc": None, "page": None, "quote": "The sky is green here."}]
    assert frames[ci]["citations"] == want

    msgs = api.get(f"/boards/{b}/cards/{q}/messages").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert "citations" not in msgs[0]
    assert msgs[1]["citations"] == want and msgs[1]["content"] == text


@requires_db
def test_v3_stream_without_passages_sends_empty_citations(api, db, user, fake_mode):
    b = seed_board(db, user)
    q = seed_card(db, user, b, kind="chat")
    frames = _turn(api, q, boardId=b)
    kinds = _kinds(frames)
    assert frames[kinds.index("citations")] == {"citations": []}
    assert "[1]" not in "".join(f["message"] for f in frames if isinstance(f, dict) and "message" in f)
    msgs = api.get(f"/boards/{b}/cards/{q}/messages").json()["messages"]
    assert msgs[1]["citations"] == []


def _assistant_calls(llm_inputs):
    return [s for s in llm_inputs if any(m.type == "system" and "<MEMORY_FACTS>" in str(m.content) for m in s)]


def _text(frames):
    return "".join(f["message"] for f in frames if isinstance(f, dict) and "message" in f)


@requires_db
def test_v3_second_turn_prompt_holds_no_old_marker(api, db, user, llm_inputs):
    import re
    b = seed_board(db, user)
    q = seed_card(db, user, b, kind="chat")
    seed_card(db, user, b, kind="source", source_type="note", status="ready", desk_id=q,
              data={"title": "N", "text": "The sky is green here."})
    first = _turn(api, q, boardId=b)
    assert _text(first) == "Fake answer to: What does it say?[1] Tell me."
    second = _turn(api, q, boardId=b)
    assert _kinds(second)[-1] == "done"
    prompt = _assistant_calls(llm_inputs)[-1]
    ai = [m.content for m in prompt if m.type == "ai"]
    assert ai == ["Fake answer to: What does it say? Tell me."]  # turn 1's answer, marker removed
    assert not any("?[1]" in str(m.content) for m in prompt)
    for m in prompt:
        if m.type == "system":
            for line in str(m.content).splitlines():
                if line.startswith("[sim ") or "Fake answer" in line:  # past answers in any block
                    assert not re.search(r"\[\d+\]", line), line
    with db.cursor() as cur:  # the stored answer keeps its marker
        cur.execute("SELECT content FROM messages WHERE node_id = %s AND role = 'assistant' "
                    "ORDER BY position LIMIT 1", (q,))
        assert cur.fetchone()[0] == "Fake answer to: What does it say?[1] Tell me."


@requires_db
def test_v3_instruction_does_not_carry_over_to_a_turn_without_passages(api, db, user, llm_inputs):
    b = seed_board(db, user)
    q = seed_card(db, user, b, kind="chat")
    note = seed_card(db, user, b, kind="source", source_type="note", status="ready", desk_id=q,
                     data={"title": "N", "text": "The sky is green here."})
    first = _turn(api, q, boardId=b)
    assert [c["n"] for c in first[_kinds(first).index("citations")]["citations"]] == [1]
    assert CITE_INSTRUCTION in [m.content for m in _assistant_calls(llm_inputs)[-1]]
    with db.cursor() as cur:
        cur.execute("UPDATE nodes SET deleted_at = now() WHERE id = %s", (note,))
    second = _turn(api, q, boardId=b)
    assert second[_kinds(second).index("citations")] == {"citations": []}
    assert CITE_INSTRUCTION not in [m.content for m in _assistant_calls(llm_inputs)[-1]]
    assert "[1]" not in _text(second)


@requires_db
def test_v3_old_assistant_rows_read_back_as_empty_citations(api, db, user):
    b = seed_board(db, user)
    q = seed_card(db, user, b, kind="chat")
    seed_message(db, user, q, "user", "u", 1)
    seed_message(db, user, q, "assistant", "a", 2)
    msgs = api.get(f"/boards/{b}/cards/{q}/messages").json()["messages"]
    assert "citations" not in msgs[0] and msgs[1]["citations"] == []


@requires_db
def test_v2_stream_has_no_citations_frame_and_stores_none(api, db, user, fake_mode):
    node = new_id("v2n")
    frames = _turn(api, node)
    kinds = _kinds(frames)
    assert "citations" not in kinds and kinds[-1] == "done" and "error" not in kinds
    assert "".join(f["message"] for f in frames if isinstance(f, dict) and "message" in f) == (
        "Fake answer to: What does it say? Tell me.")
    with db.cursor() as cur:
        cur.execute("SELECT role, citations FROM messages WHERE node_id = %s ORDER BY position", (node,))
        assert cur.fetchall() == [("user", None), ("assistant", None)]
