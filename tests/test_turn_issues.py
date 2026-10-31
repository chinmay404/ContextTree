"""When something breaks in production, the user is told and the operator can find it."""
import logging
from pathlib import Path

import pytest

from app.agent import main as main_module
from app.agent.store import trace_store
from app.agent.store.PostgresStore import PostgresConversationStore
from app.api.v1.endpoints import health as health_route
from app.core.config import settings
from tests.test_v3_citations import _frames, _kinds, fake_mode  # noqa: F401
from tests.v3_support import api, db, new_id, requires_db, seed_board, seed_card, seed_message, unit_emb, user  # noqa: F401


def _ask(api, chat, **extra):
    return api.post("/chat/stream", json={"message": "a question", "message_id": new_id("m"), "nodeId": chat,
                                          "model": "openai/gpt-5-mini", **extra})


def _issue_codes(frames):
    return [i["code"] for f in frames if isinstance(f, dict) and "issues" in f for i in f["issues"]]


@pytest.fixture
def chat(db, user):
    b = seed_board(db, user)
    q = seed_card(db, user, b, kind="chat", data={"name": "Named"})
    seed_card(db, user, b, kind="source", source_type="note", status="ready", desk_id=q, data={"title": "N", "text": "note"})
    return dict(b=b, q=q, email=user)


@pytest.fixture
def working_search(monkeypatch):
    monkeypatch.setattr(main_module, "get_embedding", lambda t: unit_emb(1))


@requires_db
def test_a_healthy_turn_reports_nothing(api, chat, fake_mode, working_search):
    frames = _frames(_ask(api, chat["q"]).text)
    assert _issue_codes(frames) == [] and _kinds(frames)[-1] == "done"


@requires_db
def test_sources_that_cannot_be_read_are_reported_not_hidden(api, chat, fake_mode, working_search, monkeypatch, caplog):
    def broken(*a, **k):
        raise RuntimeError("db gone")
    monkeypatch.setattr(main_module, "build_board_context", broken)
    with caplog.at_level(logging.ERROR):
        frames = _frames(_ask(api, chat["q"]).text)
    assert "sources_unavailable" in _issue_codes(frames)
    assert _kinds(frames)[-1] == "done"  # the user still gets an answer, and knows what it is missing
    assert "ENGINE_ALERT" in caplog.text and "sources_unavailable" in caplog.text


@requires_db
def test_the_issue_comes_with_words_a_person_can_read(api, chat, fake_mode, working_search, monkeypatch):
    monkeypatch.setattr(main_module, "build_board_context", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    frames = _frames(_ask(api, chat["q"]).text)
    issue = next(f["issues"][0] for f in frames if isinstance(f, dict) and "issues" in f)
    assert "source" in issue["message"].lower() and "RuntimeError" not in issue["message"]


@requires_db
def test_an_answer_that_was_not_saved_says_so(api, chat, fake_mode, working_search, monkeypatch, caplog):
    def broken(self, *a, **k):
        raise RuntimeError("write failed")
    monkeypatch.setattr(PostgresConversationStore, "add_message", broken)
    with caplog.at_level(logging.ERROR):
        frames = _frames(_ask(api, chat["q"]).text)
    assert "not_saved" in _issue_codes(frames)
    assert "ENGINE_ALERT" in caplog.text
    assert _kinds(frames)[-1] == "done"


@requires_db
def test_a_failed_web_search_is_reported(api, chat, fake_mode, working_search, monkeypatch):
    monkeypatch.setattr(main_module, "search_web", lambda *a, **k: [])
    frames = _frames(_ask(api, chat["q"], web_search=True).text)
    assert "web_search_empty" in _issue_codes(frames)


@requires_db
def test_the_inspector_can_show_what_went_wrong(api, chat, fake_mode, working_search, monkeypatch):
    monkeypatch.setattr(main_module, "build_board_context", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    _ask(api, chat["q"])
    trace = trace_store.latest_trace(main_module.board_store, chat["q"], chat["email"])
    assert [i["code"] for i in trace["issues"]] == ["sources_unavailable"]


@requires_db
def test_problems_only_the_operator_can_act_on_stay_out_of_the_users_way(api, chat, fake_mode, working_search, monkeypatch, caplog):
    def broken(*a, **k):
        raise RuntimeError("map broke")
    monkeypatch.setattr(main_module, "build_graph_card", broken)
    with caplog.at_level(logging.ERROR):
        frames = _frames(_ask(api, chat["q"]).text)
    assert _issue_codes(frames) == []
    assert "ENGINE_ALERT" in caplog.text and "board_map_failed" in caplog.text
    trace = trace_store.latest_trace(main_module.board_store, chat["q"], chat["email"])
    assert [i["code"] for i in trace["issues"]] == ["board_map_failed"]


# ── Branches ─────────────────────────────────────────────────────────────────

@pytest.fixture
def branch(db, user):
    b = seed_board(db, user)
    root = seed_card(db, user, b, kind="chat", data={"name": "Root"})
    ids = [seed_message(db, user, root, "user" if i % 2 else "assistant", f"root {i}", i) for i in range(1, 7)]
    child = seed_card(db, user, b, kind="chat", parent_id=root, data={"name": "Child"})
    with db.cursor() as cur:
        cur.execute("UPDATE nodes SET forked_from_message_id = %s WHERE id = %s", (ids[5], child))
    return dict(b=b, root=root, child=child, email=user)


@requires_db
def test_a_branch_that_cannot_start_refuses_instead_of_answering_blind(api, branch, fake_mode, working_search, monkeypatch, caplog, db):
    """A missing migration once meant: no history, no error, a confident wrong answer."""
    monkeypatch.setattr(PostgresConversationStore, "fork_thread", lambda self, **k: False)
    with caplog.at_level(logging.ERROR):
        r = _ask(api, branch["child"])
    assert r.status_code == 503 and r.json()["error"] == "branch_init_failed"
    assert "history" in r.json()["detail"]
    assert "ENGINE_ALERT" in caplog.text
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM messages WHERE node_id = %s AND role = 'assistant'", (branch["child"],))
        assert cur.fetchone()[0] == 0


@requires_db
def test_a_refused_branch_gives_the_daily_message_back(api, branch, fake_mode, working_search, monkeypatch, db):
    from tests.test_v3_chat_route import _quota  # the quota helper that test file already trusts
    monkeypatch.setattr(PostgresConversationStore, "fork_thread", lambda self, **k: False)
    _ask(api, branch["child"])
    assert _quota(db, branch["email"]) == 0


@requires_db
def test_an_unreadable_lineage_is_reported(api, branch, fake_mode, working_search, monkeypatch):
    frames = _frames(_ask(api, branch["child"]).text)  # starts the branch normally
    assert _issue_codes(frames) == []
    monkeypatch.setattr(PostgresConversationStore, "get_thread_lineage_summary", lambda self, t: None)
    frames = _frames(_ask(api, branch["child"]).text)
    assert "history_unavailable" in _issue_codes(frames)


@requires_db
def test_failed_history_blocks_are_reported_on_the_first_turn(api, branch, fake_mode, working_search, monkeypatch):
    from app.api.v1.endpoints import chat as chat_route

    def broken(*a, **k):
        raise RuntimeError("summary failed")
    monkeypatch.setattr(chat_route, "cover_messages", broken)
    frames = _frames(_ask(api, branch["child"]).text)
    assert "history_partial" in _issue_codes(frames)


# ── The operator's check ─────────────────────────────────────────────────────

def test_expected_migration_is_the_newest_file():
    newest = sorted(p.name for p in (Path(__file__).parent.parent / "db" / "migrations").glob("*.sql"))[-1]
    assert health_route.EXPECTED_MIGRATION == newest


@requires_db
def test_deep_health_is_green_when_all_is_well(api, working_search, monkeypatch):
    monkeypatch.setattr(health_route, "get_embedding", lambda t: unit_emb(1))
    r = api.get("/health/deep")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["checks"]["database"] == "ok" and body["checks"]["migrations"] == "ok" and body["checks"]["embeddings"] == "ok"
    assert body["embeddingProvider"] == settings.EMBEDDING_PROVIDER


@requires_db
def test_deep_health_is_red_when_embeddings_fail(api, monkeypatch):
    monkeypatch.setattr(health_route, "get_embedding", lambda t: [0.0] * 768)
    r = api.get("/health/deep")
    assert r.status_code == 503 and r.json()["checks"]["embeddings"] == "failing"


@requires_db
def test_deep_health_is_red_when_a_migration_is_missing(api, monkeypatch):
    monkeypatch.setattr(health_route, "get_embedding", lambda t: unit_emb(1))
    monkeypatch.setattr(health_route, "EXPECTED_MIGRATION", "999_not_applied.sql")
    r = api.get("/health/deep")
    assert r.status_code == 503 and r.json()["checks"]["migrations"] == "behind"


def test_deep_health_never_shows_exception_text(api, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("postgresql://user:secret@host/db")
    monkeypatch.setattr(health_route, "_check_database", broken)
    monkeypatch.setattr(health_route, "get_embedding", lambda t: unit_emb(1))
    r = api.get("/health/deep")
    assert r.status_code == 503 and "secret" not in r.text
