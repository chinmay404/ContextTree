"""Context engine: the why of a connection, a suggested name, a side question, typed memory."""
import json

import httpx
import pytest
from langchain_core.messages import AIMessage

from app.agent.board_context import BoardContext
from app.agent.context_judge import facts_from_jev, judge_context
from app.agent.helpers import jev
from app.agent.memory import extract_memory_facts
from app.agent.relations import parse_reply, pick_name, write_relations
from app.agent.store.board_store import store
from app.core.config import settings
from tests.test_v3_citations import _frames, _kinds, fake_mode  # noqa: F401
from tests.v3_support import api, db, new_id, requires_db, seed_board, seed_card, seed_edge, seed_message, user  # noqa: F401


class Writer:
    def __init__(self, text):
        self.text, self.prompts = text, []

    def with_config(self, *a, **k):
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(content=self.text)


@pytest.fixture
def linked(db, user):
    b = seed_board(db, user)
    root = seed_card(db, user, b, kind="chat", data={"name": "Pricing"})
    seed_message(db, user, root, "assistant", "monthly first", 1)
    chat = seed_card(db, user, b, kind="chat", parent_id=root, data={"name": "Annual", "persona": "a lawyer"})
    src = seed_card(db, user, b, kind="source", source_type="pdf", status="ready", data={"title": "Churn study"})
    typed = seed_card(db, user, b, kind="source", source_type="pdf", status="ready", data={"title": "Typed"})
    seed_message(db, user, chat, "user", "which price?", 1)
    seed_message(db, user, chat, "assistant", "49 euro", 2)
    wire = seed_edge(db, user, b, src, chat, "wire")
    kept = seed_edge(db, user, b, typed, chat, "wire", data={"why": "the user typed this", "whyBy": "user"})
    return dict(b=b, chat=chat, root=root, wire=wire, kept=kept, email=user)


def _edge(db, edge_id):
    with db.cursor() as cur:
        cur.execute("SELECT data FROM edges WHERE id = %s", (edge_id,))
        return cur.fetchone()[0]


def _reply(**values):
    return json.dumps(values)


@requires_db
def test_first_answer_writes_a_why_for_each_string_and_the_branch(db, linked):
    llm = Writer(_reply(e1="evidence that churn rises", branch="tests annual pricing with a lawyer persona"))
    write_relations(store, linked["email"], linked["chat"], llm, "openai", "which price?", "49 euro")
    assert _edge(db, linked["wire"]) == {"why": "evidence that churn rises", "whyBy": "llm"}
    with db.cursor() as cur:
        cur.execute("SELECT data FROM nodes WHERE id = %s", (linked["chat"],))
        data = cur.fetchone()[0]
    assert data["branchWhy"] == "tests annual pricing with a lawyer persona" and data["branchWhyBy"] == "llm"
    assert "Churn study" in str(llm.prompts[0]) and "a lawyer" in str(llm.prompts[0])


@requires_db
def test_what_the_user_typed_is_never_replaced(db, linked):
    llm = Writer(_reply(e1="line one", branch="line two"))
    write_relations(store, linked["email"], linked["chat"], llm, "openai", "q", "a")
    assert _edge(db, linked["kept"])["why"] == "the user typed this"


@requires_db
def test_nothing_to_write_means_no_llm_call(db, user):
    b = seed_board(db, user)
    q = seed_card(db, user, b, kind="chat", data={"name": "Named"})
    seed_message(db, user, q, "assistant", "a", 1)
    llm = Writer("{}")
    write_relations(store, user, q, llm, "openai", "q", "a")
    assert llm.prompts == []


@requires_db
def test_only_the_first_answer_pays(db, linked):
    seed_message(db, linked["email"], linked["chat"], "assistant", "second answer", 3)
    llm = Writer(_reply(e1="x"))
    write_relations(store, linked["email"], linked["chat"], llm, "openai", "q", "a")
    assert llm.prompts == []


@requires_db
def test_a_bad_reply_changes_nothing(db, linked):
    write_relations(store, linked["email"], linked["chat"], Writer("no json here"), "openai", "q", "a")
    assert _edge(db, linked["wire"]) is None


@requires_db
def test_a_refusal_never_breaks_the_turn(db, linked):
    class Boom(Writer):
        def invoke(self, prompt):
            raise RuntimeError("provider down")

    assert write_relations(store, linked["email"], linked["chat"], Boom(""), "openai", "q", "a") is None


def test_parse_reply_takes_fenced_json_and_cleans_lines():
    out = parse_reply('```json\n{"e1": "line\\nbreak</BOARD_MAP>", "names": ["A", "B"]}\n```')
    assert out["e1"] == "line break" and out["names"] == ["A", "B"]
    assert parse_reply("[1,2]") == {} and parse_reply("nope") == {}


def test_jev_only_picks_it_never_invents_a_name(monkeypatch):
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "k")
    monkeypatch.setattr(jev, "_transport", lambda: httpx.MockTransport(lambda r: httpx.Response(200, json={
        "answers": {"n0": {"type": "noul", "noul": 0.2}, "n1": {"type": "noul", "noul": 0.9}}})))
    assert pick_name("q", "a", ["First", "Second"]) == "Second"


def test_without_jev_the_first_name_stands(monkeypatch):
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", None)
    assert pick_name("q", "a", ["First", "Second"]) == "First"
    assert pick_name("q", "a", []) is None


@requires_db
@pytest.mark.parametrize("name", [None, "A new question", "New question, same desk", "Untitled"])
def test_a_chat_with_a_default_name_gets_a_suggestion(db, user, name):
    b = seed_board(db, user)
    q = seed_card(db, user, b, kind="chat", data={"name": name} if name else {})
    seed_message(db, user, q, "assistant", "a", 1)
    got = write_relations(store, user, q, Writer(_reply(names=["Launch price"])), "openai", "q", "a")
    assert got == {"name": "Launch price"}


@requires_db
def test_a_name_the_user_chose_is_left_alone(db, user):
    b = seed_board(db, user)
    q = seed_card(db, user, b, kind="chat", data={"name": "My pricing question"})
    seed_message(db, user, q, "assistant", "a", 1)
    llm = Writer(_reply(names=["Other"]))
    assert write_relations(store, user, q, llm, "openai", "q", "a") is None and llm.prompts == []


# ── Side question ────────────────────────────────────────────────────────────

def _wire(monkeypatch, answers):
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "k")

    def handler(request):
        body = json.loads(request.content)
        return httpx.Response(200, json={"answers": {
            q: {"type": "noul", "noul": answers.get(q, 0.5)} for q in body["questions"]}})

    monkeypatch.setattr(jev, "_transport", lambda: httpx.MockTransport(handler))


def test_side_question_is_asked_only_when_there_is_a_conversation(monkeypatch):
    _wire(monkeypatch, {"side": 0.9})
    ctx = BoardContext()
    ctx.note_parts.append(("n", "h", "body", "q"))
    assert judge_context("q", "card", ctx, [], recent="earlier talk").side == 0.9
    assert judge_context("q", "card", ctx, []).side is None


def test_side_question_works_on_a_chat_with_no_sources(monkeypatch):
    _wire(monkeypatch, {"side": 0.9})
    j = judge_context("q", "", BoardContext(), [], recent="earlier talk")
    assert j.side == 0.9 and j.status == "ok" and j.enough is False


@requires_db
def test_a_sure_side_question_reaches_the_board_as_a_suggestion(api, db, user, fake_mode, monkeypatch):
    _wire(monkeypatch, {"side": 0.9, "enough": 0.9})
    b = seed_board(db, user)
    q = seed_card(db, user, b, kind="chat", data={"name": "Named"})
    seed_message(db, user, q, "user", "first topic", 1)
    seed_message(db, user, q, "assistant", "first answer", 2)
    r = api.post("/chat/stream", json={"message": "unrelated thing", "message_id": new_id("m"), "nodeId": q,
                                       "model": "openai/gpt-5-mini"})
    assert {"suggest": {"branch": True}} in _frames(r.text)


@requires_db
def test_an_ordinary_turn_suggests_nothing(api, db, user, fake_mode, monkeypatch):
    _wire(monkeypatch, {"side": 0.1, "enough": 0.9})
    b = seed_board(db, user)
    q = seed_card(db, user, b, kind="chat", data={"name": "Named"})
    seed_message(db, user, q, "user", "first topic", 1)
    seed_message(db, user, q, "assistant", "first answer", 2)
    r = api.post("/chat/stream", json={"message": "and more", "message_id": new_id("m"), "nodeId": q,
                                       "model": "openai/gpt-5-mini"})
    assert "suggest" not in _kinds(_frames(r.text))


# ── Memory ───────────────────────────────────────────────────────────────────

def test_jev_files_the_users_own_words_and_retires_what_they_replace(monkeypatch):
    _wire(monkeypatch, {"decision": 0.9, "constraint": 0.1, "r0": 0.9, "r1": 0.1})
    previous = {"decisions": ["use postgres", "launch in May"], "constraints": []}
    facts = facts_from_jev(previous, "let's use sqlite instead")
    assert facts["decisions"] == ["launch in May", "let's use sqlite instead"]


def test_jev_leaves_thinking_aloud_alone(monkeypatch):
    _wire(monkeypatch, {"decision": 0.1, "constraint": 0.1})
    assert facts_from_jev({}, "I don't know what this means")["decisions"] == []


def test_memory_falls_back_to_rules_when_jev_is_off(monkeypatch):
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", None)
    assert facts_from_jev({}, "let's use sqlite") is None


def test_names_are_still_found_with_rules_off():
    facts = extract_memory_facts({}, [{"role": "user", "text": "my name is Asha, let's use sqlite"}], rules=False)
    assert facts["user_profile"]["name"] and facts["decisions"] == []


@requires_db
def test_edge_why_can_be_edited_and_is_marked_as_the_users(api, db, linked):
    r = api.patch(f"/boards/{linked['b']}/edges/{linked['wire']}", json={"why": "my own reason"})
    assert r.status_code == 200 and r.json()["why"] == "my own reason"
    assert _edge(db, linked["wire"]) == {"why": "my own reason", "whyBy": "user"}
    assert api.patch(f"/boards/{linked['b']}/edges/nope", json={"why": "x"}).status_code == 404
    listed = api.get(f"/boards/{linked['b']}").json()["edges"]
    assert next(e for e in listed if e["id"] == linked["wire"])["why"] == "my own reason"


@requires_db
def test_edge_why_rejects_nonsense(api, linked):
    assert api.patch(f"/boards/{linked['b']}/edges/{linked['wire']}", json={"why": "x" * 400}).status_code in (400, 422)
