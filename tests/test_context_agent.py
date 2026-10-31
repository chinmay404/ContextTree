"""Context engine: the agent that goes deeper reads only what the chat can read, and stops."""
import json

import pytest
from langchain_core.messages import AIMessage

from app.agent.board_context import load_outlines
from app.agent.context_agent import MAX_STEPS, BoardTools, explore
from app.agent.helpers.v3_llm import KeyRejectedError
from app.agent.reads import resolve_reads
from app.agent.store.board_store import store
from tests.v3_support import _vec, db, new_id, requires_db, seed_board, seed_card, seed_edge, seed_message, unit_emb, user


def _chunk(db, email, board, card, idx, loc, text, emb=1):
    with db.cursor() as cur:
        cur.execute("SELECT id FROM external_files WHERE node_id = %s", (card,))
        row = cur.fetchone()
        fid = row[0] if row else new_id("f")
        if not row:
            cur.execute("INSERT INTO external_files (id, node_id, canvas_id, user_email, file_name, file_type) "
                        "VALUES (%s,%s,%s,%s,'f','pdf')", (fid, card, board, email))
        cur.execute("INSERT INTO file_chunks (file_id, chunk_index, chunk_text, metadata, embedding) "
                    "VALUES (%s,%s,%s,%s::jsonb,%s::vector)", (fid, idx, text, json.dumps({"loc": loc}), _vec(unit_emb(emb))))


@pytest.fixture
def board(db, user):
    b = seed_board(db, user)
    root = seed_card(db, user, b, kind="chat", data={"name": "Root"})
    keep = seed_message(db, user, root, "user", "before the fork", 1)
    seed_message(db, user, root, "assistant", "after the fork", 2)
    chat = seed_card(db, user, b, kind="chat", parent_id=root, data={"name": "Mine"})
    with db.cursor() as cur:
        cur.execute("UPDATE nodes SET forked_from_message_id = %s WHERE id = %s", (keep, chat))
    pdf = seed_card(db, user, b, kind="source", source_type="pdf", status="ready", desk_id=chat,
                    data={"title": "Report", "desc": "market size", "excerpts": [{"loc": "p. 9", "on": False}]})
    _chunk(db, user, b, pdf, 0, "p. 1", "Intro line\nmore intro")
    _chunk(db, user, b, pdf, 1, "p. 7", "Pricing table Bundesliga")
    _chunk(db, user, b, pdf, 2, "p. 9", "switched off text")
    secret = seed_card(db, user, b, kind="source", source_type="pdf", status="ready", data={"title": "Secret"})
    _chunk(db, user, b, secret, 0, "p. 1", "secret Bundesliga text")
    sibling = seed_card(db, user, b, kind="chat", parent_id=root, data={"name": "Sibling"})
    seed_message(db, user, sibling, "assistant", "sibling words", 1)
    linked = seed_card(db, user, b, kind="chat", data={"name": "Linked"})
    seed_message(db, user, linked, "user", "linked question", 1)
    seed_message(db, user, linked, "assistant", "linked latest answer", 2)
    seed_edge(db, user, b, linked, chat, "answer", data={"why": "legal limits"})
    return dict(b=b, chat=chat, root=root, pdf=pdf, secret=secret, sibling=sibling, linked=linked, email=user)


def _tools(bd):
    return BoardTools(store, resolve_reads(store, bd["email"], bd["b"], bd["chat"]), bd["email"])


def _alias(tools, card):
    return next(a for a, c in tools.aliases.items() if c == card)


@requires_db
def test_list_connected_shows_only_what_the_chat_can_read(board):
    text = _tools(board).list_connected()
    assert "Report" in text and "Root" in text and "Linked" in text
    assert "Secret" not in text and "Sibling" not in text


@requires_db
def test_a_real_card_id_is_not_accepted_only_an_alias(board):
    tools = _tools(board)
    assert tools.read_section(id=board["secret"], section="p. 1") == "Unknown source."
    assert tools.read_chat(id=board["sibling"]) == "Unknown chat."


@requires_db
def test_why_connected_uses_the_string_then_the_description(board):
    tools = _tools(board)
    assert tools.why_connected(id=_alias(tools, board["linked"])) == "legal limits"
    assert tools.why_connected(id=_alias(tools, board["pdf"])) == "market size"


@requires_db
def test_read_section_lists_sections_then_reads_one(board):
    tools = _tools(board)
    a = _alias(tools, board["pdf"])
    listing = tools.read_section(id=a)
    assert "p. 1: Intro line" in listing and "p. 7: Pricing table" in listing
    assert "p. 9" not in listing
    assert "Pricing table" in tools.read_section(id=a, section="p. 7")
    assert [c.loc for c in tools.gathered] == ["p. 7"]


@requires_db
def test_switched_off_section_stays_closed(board):
    tools = _tools(board)
    assert tools.read_section(id=_alias(tools, board["pdf"]), section="p. 9") == "No such section."
    assert tools.gathered == []


@requires_db
def test_search_never_reaches_unconnected_sources(board, monkeypatch):
    from app.agent import context_agent
    monkeypatch.setattr(context_agent, "get_embedding", lambda t: unit_emb(1))
    tools = _tools(board)
    text = tools.search(query="Bundesliga")
    assert "Pricing table" in text and "secret" not in text
    assert "switched off" not in text


@requires_db
def test_parent_chat_is_read_only_up_to_the_fork(board):
    tools = _tools(board)
    text = tools.read_chat(id=_alias(tools, board["root"]))
    assert "before the fork" in text and "after the fork" not in text


@requires_db
def test_linked_chat_gives_its_latest_answer_not_its_history(board):
    tools = _tools(board)
    text = tools.read_chat(id=_alias(tools, board["linked"]))
    assert text == "linked latest answer"


class Scripted:
    """A model that calls the tools it is told to, then says DONE."""

    def __init__(self, *turns, error=None):
        self.turns, self.error, self.seen, self.bound = list(turns), error, [], None

    def bind_tools(self, tools):
        self.bound = tools
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        if self.error:
            raise self.error
        calls = self.turns.pop(0) if self.turns else []
        return AIMessage(content="DONE" if not calls else "", tool_calls=[
            {"name": n, "args": a, "id": f"c{i}", "type": "tool_call"} for i, (n, a) in enumerate(calls)])


@requires_db
def test_explore_keeps_what_it_read_and_records_each_step(board):
    tools = _tools(board)
    a = _alias(tools, board["pdf"])
    llm = Scripted([("read_section", {"id": a})], [("read_section", {"id": a, "section": "p. 7"})])
    out = explore(llm, "q", "card", tools, "openai")
    assert [s["tool"] for s in out.steps] == ["read_section", "read_section"]
    assert [c.loc for c in out.chunks] == ["p. 7"] and out.headers[board["pdf"]].startswith("[source | PDF | Report")
    assert not out.failed


@requires_db
def test_explore_stops_at_the_step_limit(board):
    tools = _tools(board)
    llm = Scripted(*[[("list_connected", {})] for _ in range(MAX_STEPS + 10)])
    assert len(explore(llm, "q", "card", tools, "openai").steps) == MAX_STEPS


@requires_db
def test_explore_stops_at_the_token_cap(board):
    tools = _tools(board)

    class Hungry(Scripted):
        def invoke(self, messages):
            reply = super().invoke(messages)
            reply.usage_metadata = {"input_tokens": 15000, "output_tokens": 0, "total_tokens": 15000}
            return reply

    llm = Hungry(*[[("list_connected", {})] for _ in range(4)])
    assert len(explore(llm, "q", "card", tools, "openai").steps) == 2


@requires_db
def test_a_model_without_tools_is_a_failure_not_a_crash(board):
    tools = _tools(board)

    class NoTools(Scripted):
        def bind_tools(self, tools):
            raise NotImplementedError

    out = explore(NoTools(), "q", "card", tools, "openai")
    assert out.failed and out.chunks == []


@requires_db
def test_a_refused_key_is_not_swallowed(board):
    tools = _tools(board)

    class Refused(Exception):
        status_code = 401

    with pytest.raises(KeyRejectedError):
        explore(Scripted(error=Refused()), "q", "card", tools, "openai")


@requires_db
def test_bad_tool_name_or_arguments_do_not_crash(board):
    tools = _tools(board)
    assert tools.run("drop_table", {}) == "Unknown tool."
    assert tools.run("read_chat", {"nonsense": "x"}) == "Bad arguments."


@requires_db
def test_outline_reads_each_sections_first_line(db, board):
    with store.tx() as cur:
        outline = load_outlines(cur, board["email"], [board["pdf"]])[board["pdf"]]
    assert outline[0] == "p. 1: Intro line"
    with store.tx() as cur:
        assert load_outlines(cur, "other@example.com", [board["pdf"]]) == {}
