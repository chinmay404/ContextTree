"""
The turn drawn in context-eng.excalidraw, end to end through /chat/stream on the fake model.

The hard guarantee is checked on the real prompt: a sibling's words and sources never reach it.
"""
import json

import httpx
import pytest

from app.agent.helpers import jev
from app.agent.store.board_store import store
from app.core.config import settings
from tests.test_context_lineage import germany  # noqa: F401 - the board fixture
from tests.test_context_retrieval import _chunk
from tests.test_v3_citations import _frames, _kinds, fake_mode, llm_inputs  # noqa: F401
from tests.v3_support import api, db, new_id, requires_db, seed_message, unit_emb, user  # noqa: F401

QUESTION = "Which price should we launch at, given what legal said?"


def _ask(api, chat, question=QUESTION):
    r = api.post("/chat/stream", json={"message": question, "message_id": new_id("m"), "nodeId": chat,
                                       "model": "openai/gpt-5-mini"})
    assert r.status_code == 200, r.text
    return _frames(r.text)


def _answer_prompt(llm_inputs):
    return [s for s in llm_inputs if any(m.type == "system" and "<BOARD_DATA>" in str(m.content) for m in s)][-1]


def _all_text(prompt):
    return "\n".join(str(m.content) for m in prompt)


@pytest.fixture
def filled(db, germany):
    g = germany
    e, b = g["email"], g["b"]
    with db.cursor() as cur:
        cur.execute("UPDATE nodes SET is_initialized = true WHERE id = %s", (g["d"],))
    _chunk(db, e, b, g["market"], 0, "p. 41", unit_emb(1), "Buyers in Germany pay around 45 euro a seat price")
    _chunk(db, e, b, g["survey"], 0, "row 3", unit_emb(1), "Survey says 49 euro price feels fair")
    _chunk(db, e, b, g["churn"], 0, "p. 7", unit_emb(2), "Churn doubles above 59 euro price")
    _chunk(db, e, b, g["gdpr"], 0, "p. 2", unit_emb(1), "GDPR SECRET consent price text")
    _chunk(db, e, b, g["hidden"], 0, "row 1", unit_emb(1), "HIDDEN competitor price list")
    seed_message(db, e, g["pricing"], "assistant", "PRICING AFTER FORK price words", 3)
    return g


@requires_db
def test_the_prompt_holds_what_the_chat_reads_and_nothing_else(api, filled, llm_inputs, fake_mode, monkeypatch):
    from app.agent import main
    monkeypatch.setattr(main, "get_embedding", lambda t: unit_emb(1))
    frames = _ask(api, filled["d"])
    assert _kinds(frames)[-1] == "done"
    text = _all_text(_answer_prompt(llm_inputs))
    for present in ("45 euro", "49 euro", "59 euro", "Budget is 50k", "consent is required",
                    "YOU ARE IN: Annual vs monthly", 'at "annual or monthly?"'):
        assert present in text, present
    for absent in ("GDPR SECRET", "HIDDEN competitor", "ads answer", "Ads", "PRICING AFTER FORK", "Interview 4"):
        assert absent not in text, absent


@requires_db
def test_sources_reach_the_model_as_data_not_as_system_text(api, filled, llm_inputs, fake_mode, monkeypatch):
    from app.agent import main
    monkeypatch.setattr(main, "get_embedding", lambda t: unit_emb(1))
    _ask(api, filled["d"])
    prompt = _answer_prompt(llm_inputs)
    system = "".join(str(m.content) for m in prompt if m.type == "system")
    assert "49 euro" not in system
    assert "49 euro" in str(prompt[-1].content) and str(prompt[-1].content).endswith(QUESTION)


@requires_db
def test_the_turn_is_traced_and_the_route_shows_it(api, filled, fake_mode, monkeypatch):
    from app.agent import main
    monkeypatch.setattr(main, "get_embedding", lambda t: unit_emb(1))
    _ask(api, filled["d"])
    r = api.get(f"/boards/{filled['b']}/cards/{filled['d']}/context-trace")
    assert r.status_code == 200
    trace = r.json()["trace"]
    assert trace["jev"] == "off" and trace["boardMap"] is True and trace["embeddingFailed"] is False
    assert filled["survey"] in {p["cardId"] for p in trace["sent"]}
    assert filled["gdpr"] not in {p["cardId"] for p in trace["sent"]}


@requires_db
def test_another_users_trace_is_hidden(api, filled, fake_mode, monkeypatch, db):
    from tests.v3_support import seed_board, seed_card
    from app.agent.store import trace_store
    other = f"o_{new_id('u')}@example.com"
    with db.cursor() as cur:
        cur.execute("INSERT INTO users (email, name) VALUES (%s, 'x')", (other,))
    b = seed_board(db, other)
    c = seed_card(db, other, b, kind="chat")
    trace_store.save_trace(store, c, other, "m", {"secret": True})
    assert api.get(f"/boards/{b}/cards/{c}/context-trace").status_code == 404
    with db.cursor() as cur:
        cur.execute("DELETE FROM context_traces WHERE node_id = %s", (c,))
        cur.execute("DELETE FROM nodes WHERE id = %s", (c,))
        cur.execute("DELETE FROM canvases WHERE id = %s", (b,))
        cur.execute("DELETE FROM users WHERE email = %s", (other,))


@requires_db
def test_embedding_outage_keeps_documents_and_warns(api, filled, fake_mode, monkeypatch, llm_inputs):
    from app.agent import main
    monkeypatch.setattr(main, "get_embedding", lambda t: [0.0] * 768)
    frames = _ask(api, filled["d"])
    assert {"context": {"embeddingFailed": True, "agentSteps": []}} in frames
    assert "49 euro" in _all_text(_answer_prompt(llm_inputs))  # found by keyword


@requires_db
def test_a_quiet_turn_sends_no_context_frame(api, filled, fake_mode, monkeypatch):
    from app.agent import main
    monkeypatch.setattr(main, "get_embedding", lambda t: unit_emb(1))
    assert "context" not in _kinds(_ask(api, filled["d"]))


class _Jev:
    def __init__(self, answers):
        self.answers = answers

    def __call__(self, request):
        body = json.loads(request.content)
        return httpx.Response(200, json={"answers": {
            q: {"type": "noul", "noul": self.answers(q, body)} for q in body["questions"]}})


@requires_db
def test_jev_drops_what_it_is_sure_about_and_the_trace_says_so(api, filled, fake_mode, monkeypatch, llm_inputs):
    from app.agent import main
    monkeypatch.setattr(main, "get_embedding", lambda t: unit_emb(1))
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "k")

    def answers(q, body):
        state = json.loads(body["state"])
        return 0.05 if "45 euro" in state["passages"].get(q, "") else 0.9

    monkeypatch.setattr(jev, "_transport", lambda: httpx.MockTransport(_Jev(answers)))
    _ask(api, filled["d"])
    assert "45 euro" not in _all_text(_answer_prompt(llm_inputs))
    trace = api.get(f"/boards/{filled['b']}/cards/{filled['d']}/context-trace").json()["trace"]
    assert trace["jev"] == "ok" and trace["enough"] is True
    assert any(d["cardId"] == filled["market"] and d["reason"] == "jev" for d in trace["dropped"])


@requires_db
def test_jev_failure_changes_nothing_for_the_user(api, filled, fake_mode, monkeypatch, llm_inputs):
    from app.agent import main
    monkeypatch.setattr(main, "get_embedding", lambda t: unit_emb(1))
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "k")
    monkeypatch.setattr(jev, "_transport", lambda: httpx.MockTransport(lambda r: httpx.Response(529)))
    frames = _ask(api, filled["d"])
    assert _kinds(frames)[-1] == "done"
    assert "49 euro" in _all_text(_answer_prompt(llm_inputs))
    trace = api.get(f"/boards/{filled['b']}/cards/{filled['d']}/context-trace").json()["trace"]
    assert trace["jev"] == "error" and trace["agent"]["ran"] is False


@requires_db
def test_not_enough_sends_the_agent_and_says_so(api, filled, fake_mode, monkeypatch):
    from app.agent import main
    monkeypatch.setattr(main, "get_embedding", lambda t: unit_emb(1))
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "k")
    monkeypatch.setattr(jev, "_transport", lambda: httpx.MockTransport(
        _Jev(lambda q, body: 0.1 if q == "enough" else 0.9)))
    _ask(api, filled["d"])
    trace = api.get(f"/boards/{filled['b']}/cards/{filled['d']}/context-trace").json()["trace"]
    assert trace["agent"]["ran"] is True and trace["enough"] is False  # the fake model calls no tools


# ── Forks ────────────────────────────────────────────────────────────────────

@requires_db
def test_siblings_inherit_one_frozen_block_and_pay_for_it_once(api, db, user, fake_mode, llm_inputs, monkeypatch):
    from tests.v3_support import seed_board, seed_card
    from app.api.v1.endpoints import chat as chat_route
    made = []
    real = chat_route.cover_messages

    def counting(store_, email, node, messages, summarize):
        def spy(rows, previous=""):
            made.append((len(rows), previous))
            return summarize(rows, previous)
        return real(store_, email, node, messages, spy)

    monkeypatch.setattr(chat_route, "cover_messages", counting)
    b = seed_board(db, user)
    root = seed_card(db, user, b, kind="chat", data={"name": "Root"})
    ids = [seed_message(db, user, root, "user" if i % 2 else "assistant", f"parent message {i}", i) for i in range(1, 13)]
    children = []
    for name in ("One", "Two"):
        child = seed_card(db, user, b, kind="chat", parent_id=root, data={"name": name})
        with db.cursor() as cur:
            cur.execute("UPDATE nodes SET forked_from_message_id = %s WHERE id = %s", (ids[9], child))
        children.append(child)
    _ask(api, children[0], "first branch question")
    calls_after_first = len(made)
    _ask(api, children[1], "second branch question")
    assert calls_after_first >= 1 and len(made) == calls_after_first
    with db.cursor() as cur:
        cur.execute("SELECT id, lineage_summary, summary FROM nodes WHERE id = ANY(%s)", (children,))
        rows = {r[0]: r for r in cur.fetchall()}
    assert rows[children[0]][1] and rows[children[0]][1] == rows[children[1]][1]
    assert not rows[children[0]][2]


@requires_db
def test_a_later_parent_message_never_reaches_the_branch(api, db, user, fake_mode, llm_inputs):
    from tests.v3_support import seed_board, seed_card
    b = seed_board(db, user)
    root = seed_card(db, user, b, kind="chat", data={"name": "Root"})
    ids = [seed_message(db, user, root, "user" if i % 2 else "assistant", f"early message {i}", i) for i in range(1, 7)]
    seed_message(db, user, root, "user", "LATER PARENT WORDS", 7)
    child = seed_card(db, user, b, kind="chat", parent_id=root, data={"name": "Child"})
    with db.cursor() as cur:
        cur.execute("UPDATE nodes SET forked_from_message_id = %s WHERE id = %s", (ids[5], child))
    _ask(api, child, "branch question")
    assert "LATER PARENT WORDS" not in _all_text(_answer_prompt(llm_inputs))


@requires_db
def test_a_grandchild_inherits_the_whole_lineage(api, db, user, fake_mode, llm_inputs):
    from tests.v3_support import seed_board, seed_card
    b = seed_board(db, user)
    root = seed_card(db, user, b, kind="chat", data={"name": "Root"})
    ids = [seed_message(db, user, root, "user" if i % 2 else "assistant", f"root {i}", i) for i in range(1, 7)]
    mid = seed_card(db, user, b, kind="chat", parent_id=root, data={"name": "Mid"})
    with db.cursor() as cur:
        cur.execute("UPDATE nodes SET forked_from_message_id = %s WHERE id = %s", (ids[5], mid))
    _ask(api, mid, "mid question")
    with db.cursor() as cur:
        cur.execute("SELECT lineage_summary FROM nodes WHERE id = %s", (mid,))
        mid_lineage = cur.fetchone()[0]
        cur.execute("SELECT id FROM messages WHERE node_id = %s AND NOT inherited ORDER BY position DESC LIMIT 1", (mid,))
        last = cur.fetchone()[0]
    child = seed_card(db, user, b, kind="chat", parent_id=mid, data={"name": "Child"})
    with db.cursor() as cur:
        cur.execute("UPDATE nodes SET forked_from_message_id = %s WHERE id = %s", (last, child))
    _ask(api, child, "child question")
    with db.cursor() as cur:
        cur.execute("SELECT lineage_summary FROM nodes WHERE id = %s", (child,))
        assert cur.fetchone()[0].startswith(mid_lineage)
    assert mid_lineage in _all_text(_answer_prompt(llm_inputs))
