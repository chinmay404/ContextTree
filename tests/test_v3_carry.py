"""Carry modes (summary / whole / fresh) in fork init (spec 5.4)."""
from types import SimpleNamespace

import pytest

from tests.v3_support import (db, user, FakeLLM, new_id, raising_llm, requires_db, seed_board, seed_card,
                              unit_emb, TEST_DB)

pytestmark = requires_db
_ = (db, user)


@pytest.fixture
def env(db, user, monkeypatch):
    from app.agent.store.PostgresStore import PostgresConversationStore
    from app.api.v1.endpoints import chat

    store = PostgresConversationStore(db_url=TEST_DB)
    board = seed_board(db, user)
    monkeypatch.setattr(chat, "get_embedding", lambda *a, **k: unit_emb(5))

    def make_parent(n, size=None):
        parent = seed_card(db, user, board, kind="chat")
        texts = []
        for i in range(n):
            text = f"parent msg {i}" if size is None else f"{i}" * size
            texts.append(text)
            store.add_message(user_id=user, thread_id=parent, message_id=new_id("m"),
                              role="user" if i % 2 == 0 else "assistant", text=text,
                              embedding=unit_emb(100 + i))
        child = seed_card(db, user, board, kind="chat", parent_id=parent)
        return parent, child, texts

    def run(child, carry=None):
        from app.schemas.item import ChatMessage
        msg = ChatMessage(message="hi", message_id=new_id("m"), nodeId=child)
        graph = SimpleNamespace(mongo_store=store)
        kw = {} if carry is None else {"carry": carry}
        chat._init_fork_if_needed(msg, graph, "test", user, **kw)

    def rows(child):
        with db.cursor() as cur:
            cur.execute("SELECT content, inherited FROM messages WHERE node_id=%s "
                        "ORDER BY position", (child,))
            return cur.fetchall()

    def node(child):
        with db.cursor() as cur:
            cur.execute("SELECT summary, is_initialized, ancestor_ids FROM nodes WHERE id=%s",
                        (child,))
            return cur.fetchone()

    return SimpleNamespace(chat=chat, store=store, user=user, make_parent=make_parent,
                           run=run, rows=rows, node=node, monkeypatch=monkeypatch)


def test_summary_mode_unchanged_default(env):
    parent, child, texts = env.make_parent(2)
    env.monkeypatch.setattr(env.chat, "get_llm", raising_llm)
    env.run(child)
    assert [r[0] for r in env.rows(child)] == texts
    assert all(r[1] for r in env.rows(child))


def test_unknown_carry_is_summary(env):
    parent, child, texts = env.make_parent(2)
    env.monkeypatch.setattr(env.chat, "get_llm", raising_llm)
    env.run(child, "bogus")
    assert [r[0] for r in env.rows(child)] == texts


def test_whole_copies_parent_messages_no_summary(env):
    parent, child, texts = env.make_parent(5)
    env.monkeypatch.setattr(env.chat, "get_llm", raising_llm)
    env.run(child, "whole")
    rows = env.rows(child)
    assert [r[0] for r in rows] == texts
    assert all(r[1] for r in rows)
    assert not (env.node(child)[0] or "")


def test_whole_caps_at_40_and_summarizes_older(env):
    parent, child, texts = env.make_parent(45)
    llm = FakeLLM("OLDER")
    env.monkeypatch.setattr(env.chat, "get_llm", lambda *a, **k: llm)
    env.run(child, "whole")
    rows = env.rows(child)
    assert [r[0] for r in rows] == texts[-40:]
    assert all(r[1] for r in rows)
    assert env.node(child)[0] == "OLDER"
    assert len(llm.calls) == 1
    assert "parent msg 0" in str(llm.calls[0])
    assert "parent msg 44" not in str(llm.calls[0])


def test_whole_caps_at_12000_tokens(env):
    parent, child, texts = env.make_parent(3, size=20_000)
    env.monkeypatch.setattr(env.chat, "get_llm", lambda *a, **k: FakeLLM("OLDER"))
    env.run(child, "whole")
    rows = env.rows(child)
    assert [r[0] for r in rows] == texts[1:]
    assert env.node(child)[0] == "OLDER"


def test_whole_not_reinitialized_on_later_message(env):
    parent, child, texts = env.make_parent(5)
    env.monkeypatch.setattr(env.chat, "get_llm", raising_llm)
    env.run(child, "whole")
    env.store.add_message(user_id=env.user, thread_id=child, message_id=new_id("m"),
                          role="user", text="native", embedding=unit_emb(7))
    env.run(child, "whole")
    assert len(env.rows(child)) == 6


def test_fresh_inherits_nothing_but_keeps_lineage(env):
    parent, child, texts = env.make_parent(4)
    env.monkeypatch.setattr(env.chat, "get_llm", raising_llm)
    env.run(child, "fresh")
    assert env.rows(child) == []
    summary, init, ancestors = env.node(child)
    assert init is True
    assert list(ancestors) == [parent]
    assert summary is None


def test_fresh_second_message_does_not_reinherit(env):
    parent, child, texts = env.make_parent(4)
    env.monkeypatch.setattr(env.chat, "get_llm", raising_llm)
    env.run(child, "fresh")
    for i in range(2):
        env.store.add_message(user_id=env.user, thread_id=child, message_id=new_id("m"),
                              role="user" if i == 0 else "assistant", text=f"own {i}",
                              embedding=unit_emb(8 + i))
    env.run(child, "fresh")
    assert [r[0] for r in env.rows(child)] == ["own 0", "own 1"]


def test_fresh_works_with_empty_parent(env):
    parent, child, _ = env.make_parent(0)
    env.monkeypatch.setattr(env.chat, "get_llm", raising_llm)
    env.run(child, "fresh")
    assert env.node(child)[1] is True
