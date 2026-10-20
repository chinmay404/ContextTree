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


@pytest.mark.parametrize("var", ["RAILWAY_ENVIRONMENT", "RAILWAY_PROJECT_ID"])
def test_fake_mode_refused_on_railway(monkeypatch, var):
    monkeypatch.setenv(var, "production")
    with pytest.raises(RuntimeError, match="local runs only"):
        Settings(ENGINE_FAKE_LLM=True)


def test_fake_mode_allowed_locally(monkeypatch):
    monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)
    monkeypatch.delenv("RAILWAY_PROJECT_ID", raising=False)
    assert Settings(ENGINE_FAKE_LLM=True).ENGINE_FAKE_LLM is True
