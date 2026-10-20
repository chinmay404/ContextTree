import asyncio
import math

import pytest

from app.core.config import Settings, settings


@pytest.fixture
def fake_mode(monkeypatch):
    monkeypatch.setattr(settings, "ENGINE_FAKE_LLM", True)


def test_fake_llm_invoke(fake_mode):
    from app.agent.helpers.get_llm import get_llm

    out = get_llm("anthropic/x").invoke("hi")
    assert out.content == "Fake answer to: hi"


def test_fake_llm_truncates_question_to_60(fake_mode):
    from app.agent.helpers.get_llm import get_llm

    q = "word " * 40
    out = get_llm("anthropic/x").invoke(q)
    assert out.content == "Fake answer to: " + q[:60]


def test_fake_llm_streams_word_by_word(fake_mode):
    from app.agent.helpers.get_llm import get_llm

    async def run():
        llm = get_llm(None)
        chunks = []
        async for ev in llm.astream_events("alpha beta gamma", version="v2"):
            if ev["event"] == "on_chat_model_stream":
                chunks.append(ev["data"]["chunk"].content)
        return chunks

    chunks = asyncio.run(run())
    assert len(chunks) > 3
    assert "".join(chunks) == "Fake answer to: alpha beta gamma"


def test_fake_embedding_deterministic_unit(fake_mode):
    from app.agent.utils.embeddings import get_embedding, get_embeddings

    a = get_embedding("a")
    assert a == get_embedding("a")
    assert a != get_embedding("b")
    assert len(a) == 768
    assert math.isclose(math.sqrt(sum(x * x for x in a)), 1.0, rel_tol=1e-6)
    assert get_embeddings(["a", "b"]) == [a, get_embedding("b")]
    assert get_embeddings([]) == []


LOCAL_DB = "postgresql://u:p@localhost:5432/x"


def _clear_railway(monkeypatch):
    import os

    for k in list(os.environ):
        if k.upper().startswith("RAILWAY_"):
            monkeypatch.delenv(k, raising=False)


@pytest.mark.parametrize(
    "var,val",
    [
        ("RAILWAY_ENVIRONMENT", "production"),
        ("RAILWAY_PROJECT_ID", "abc"),
        ("RAILWAY_SERVICE_ID", "svc"),
        ("RAILWAY_ENVIRONMENT", ""),
    ],
)
def test_fake_mode_refused_on_railway(monkeypatch, var, val):
    _clear_railway(monkeypatch)
    monkeypatch.setenv(var, val)
    with pytest.raises(RuntimeError, match="local runs only"):
        Settings(ENGINE_FAKE_LLM=True, DATABASE_URL=LOCAL_DB)


def test_guard_fires_on_env_driven_settings(monkeypatch):
    _clear_railway(monkeypatch)
    monkeypatch.setenv("RAILWAY_SERVICE_ID", "svc")
    monkeypatch.setenv("ENGINE_FAKE_LLM", "1")
    monkeypatch.setenv("DATABASE_URL", LOCAL_DB)
    with pytest.raises(RuntimeError, match="local runs only"):
        Settings()


@pytest.mark.parametrize(
    "url", ["postgresql://u:p@db.example.com:5432/x", "postgresql://u:p@10.0.0.5/x"]
)
def test_fake_mode_refused_for_non_local_db(monkeypatch, url):
    _clear_railway(monkeypatch)
    with pytest.raises(RuntimeError, match="local runs only"):
        Settings(ENGINE_FAKE_LLM=True, DATABASE_URL=url)


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://u:p@localhost:55432/x",
        "postgresql://u:p@127.0.0.1/x",
        "postgresql://u:p@[::1]:5432/x",
    ],
)
def test_fake_mode_allowed_locally_and_warns(monkeypatch, caplog, url):
    _clear_railway(monkeypatch)
    with caplog.at_level("WARNING"):
        s = Settings(ENGINE_FAKE_LLM=True, DATABASE_URL=url)
    assert s.ENGINE_FAKE_LLM is True
    assert "ENGINE_FAKE_LLM is ON" in caplog.text


@pytest.mark.parametrize("val", ["0", "false", "False"])
def test_env_off_values(monkeypatch, val):
    _clear_railway(monkeypatch)
    monkeypatch.setenv("ENGINE_FAKE_LLM", val)
    monkeypatch.setenv("RAILWAY_SERVICE_ID", "svc")  # would raise if it were on
    assert Settings(DATABASE_URL="postgresql://u:p@db.example.com/x").ENGINE_FAKE_LLM is False


def test_env_on_value(monkeypatch):
    _clear_railway(monkeypatch)
    monkeypatch.setenv("ENGINE_FAKE_LLM", "1")
    monkeypatch.setenv("DATABASE_URL", LOCAL_DB)
    assert Settings().ENGINE_FAKE_LLM is True


# ── flag unset: real path, fake never returned ──────────────────────────────
def test_flag_off_get_llm_uses_real_path(monkeypatch):
    from app.agent.helpers import get_llm as mod

    monkeypatch.setattr(settings, "ENGINE_FAKE_LLM", False)
    sentinel = object()
    monkeypatch.setattr(mod, "get_anthropic_llm", lambda *a, **k: sentinel)
    assert mod.get_llm("anthropic/x") is sentinel


def test_flag_off_get_embedding_uses_real_path(monkeypatch):
    from app.agent.utils import embeddings as mod

    monkeypatch.setattr(settings, "ENGINE_FAKE_LLM", False)
    monkeypatch.setattr(mod, "_embed_query", lambda t: [9.0])
    monkeypatch.setattr(mod, "_embed_documents", lambda ts: [[9.0] for _ in ts])
    assert mod.get_embedding("a") == [9.0]
    assert mod.get_embeddings(["a"]) == [[9.0]]


# ── every runtime path, network disabled, flag on ───────────────────────────
@pytest.fixture
def no_network(monkeypatch):
    import socket

    import httpx
    import requests

    def boom(*a, **k):
        raise AssertionError("network used in fake mode")

    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(httpx.Client, "send", boom)
    monkeypatch.setattr(httpx.AsyncClient, "send", boom)
    monkeypatch.setattr(requests.Session, "request", boom)


def test_path_chat_turn(fake_mode, no_network):
    from langchain_core.messages import HumanMessage

    from app.agent.nodes.assistant_node import AgentNodes

    out = AgentNodes().assistant(
        {
            "messages": [HumanMessage(content="what is up")],
            "model": "anthropic/claude-x",
            "user_id": "u@x.org",
            "count": 0,
        }
    )
    assert out["messages"][-1].content == "Fake answer to: what is up"


def test_path_summary(fake_mode, no_network):
    from langchain_core.messages import AIMessage, HumanMessage

    from app.agent.nodes.assistant_node import AgentNodes

    msgs = [HumanMessage(content="q1", id="1"), AIMessage(content="a1", id="2")] * 5
    out = AgentNodes().summurize(
        {"messages": msgs, "summary": "", "model": "openai/gpt-5", "user_id": "u@x.org"}
    )
    assert out["summary"].startswith("Fake answer to:")


def test_path_fork_summarise_model_and_access(fake_mode, no_network):
    from app.agent.helpers.get_llm import get_llm, validate_model_access
    from app.agent.helpers.memory_model import get_summary_model_name

    for m in ("anthropic/x", "openai/gpt-5", "litellm/custom", None):
        assert validate_model_access(m, "u@x.org") is None
        llm = get_llm(get_summary_model_name(m), user_id="u@x.org")
        assert llm.invoke("sum this").content.startswith("Fake answer to:")


def test_path_ingest_overview_and_embeddings(fake_mode, no_network):
    from app.agent import ingest
    from app.agent.utils.embeddings import get_embedding, get_embeddings

    out = ingest.make_overview("The quick brown fox " * 20, "u@x.org")
    assert out.startswith("Fake answer to:")
    assert len(get_embedding(out)) == 768
    assert all(len(v) == 768 for v in get_embeddings(["a", "b"]))


def test_no_structured_output_users():
    import pathlib

    hits = [
        str(p)
        for p in pathlib.Path("app").rglob("*.py")
        if "with_structured_output" in p.read_text(encoding="utf-8", errors="ignore")
    ]
    assert hits == []
