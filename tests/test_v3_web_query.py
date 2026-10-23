"""S1: a V3 web search uses one standalone query, rewritten by the chat's own model on the user's key."""
import secrets
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.helpers import v3_llm, web_query
from app.agent.helpers.v3_llm import TurnKey, v3_route

GPT = "openai/gpt-5-mini"
CLAUDE = "anthropic/claude-sonnet-4-20250514"


class Recorder:
    """Stands in for the chat model: records how it was built and what it was asked."""

    def __init__(self, reply="OWASP ZAP capabilities for developers", delay=0.0, error=None):
        self.reply, self.delay, self.error = reply, delay, error
        self.built, self.prompts, self.bound = [], [], []

    def build(self, model_id, user_email, *a, **k):
        self.built.append({"model": model_id, "email": user_email, **k})
        return self

    def bind(self, **kw):
        self.bound.append(kw)
        return self

    def with_config(self, *a, **k):
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return AIMessage(content=self.reply)


@pytest.fixture
def model(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "ENGINE_FAKE_LLM", False)

    def no_lookup(*a, **k):
        raise AssertionError("the rewrite must use the turn's key, not look one up")

    monkeypatch.setattr(v3_llm, "fetch_user_provider_api_key", no_lookup)
    rec = Recorder()
    monkeypatch.setattr(web_query, "get_v3_llm", rec.build)
    return rec


# A random stand-in for the user's stored key, made per run (no fixed value in the source).
USER_KEY = f"user-{secrets.token_hex(6)}"


def _config(model_id=GPT):
    return {"configurable": {"model": model_id,
                             v3_llm.TURN_KEY_CONFIG: TurnKey(v3_route(model_id), USER_KEY)}}


HISTORY = [HumanMessage(content="What is OWASP ZAP?"),
           AIMessage(content="ZAP is an open-source web app security scanner.")]


def _rewrite(message="what it can do for me", config=None, **kw):
    kw.setdefault("chat_name", "OWASP ZAP")
    kw.setdefault("history", HISTORY)
    kw.setdefault("source_titles", ["ZAP user guide"])
    return web_query.rewrite_search_query(message, config if config is not None else _config(), **kw)


def test_a_follow_up_is_rewritten_with_the_chats_topic(model):
    assert _rewrite() == "OWASP ZAP capabilities for developers"
    prompt = str(model.prompts[0])
    for part in ("OWASP ZAP", "What is OWASP ZAP?", "open-source web app security scanner",
                 "ZAP user guide", "what it can do for me"):
        assert part in prompt
    [built] = model.built
    assert built["model"] == GPT and built["temperature"] == 0
    assert built["max_output_tokens"] == web_query.REWRITE_MAX_TOKENS == 32
    assert built["resolved"] == (v3_route(GPT), USER_KEY)  # the turn's key, no lookup


def test_openai_reasoning_models_think_minimally_for_the_rewrite(model):
    _rewrite()
    assert model.bound == [{"reasoning_effort": "minimal"}]
    model.bound.clear()
    _rewrite(config=_config(CLAUDE))
    assert model.bound == []


def test_the_rewrite_reads_only_the_last_three_turns(model):
    history = []
    for i in range(10):
        history += [HumanMessage(content=f"question {i}"), AIMessage(content=f"answer {i}")]
    _rewrite(history=history)
    prompt = str(model.prompts[0])
    assert "question 7" in prompt and "answer 9" in prompt and "question 6" not in prompt


def test_the_reply_is_cleaned_to_one_line(model):
    model.reply = '  "OWASP ZAP features"\nBecause the user asked...'
    assert _rewrite() == "OWASP ZAP features"
    model.reply = "Query: zap active scan"
    assert _rewrite() == "zap active scan"


@pytest.mark.parametrize("case", ["error", "auth", "empty", "oversized", "timeout"])
def test_fallback_is_chat_name_and_message(model, monkeypatch, case):
    if case == "error":
        model.error = RuntimeError("provider down")
    elif case == "auth":  # a refused key: plain fallback; the answer then reports key_rejected
        err = Exception("401 refused")
        err.status_code = 401
        model.error = err
    elif case == "empty":
        model.reply = "   "
    elif case == "oversized":
        model.reply = "x" * (web_query.MAX_QUERY_CHARS + 1)
    elif case == "timeout":
        monkeypatch.setattr(web_query, "REWRITE_TIMEOUT_SECONDS", 0.2)
        model.delay = 2.0
    started = time.monotonic()
    assert _rewrite() == "OWASP ZAP what it can do for me"
    assert time.monotonic() - started < 1.5


def test_the_timeout_is_four_seconds():
    assert web_query.REWRITE_TIMEOUT_SECONDS == 4


def test_fallback_without_a_chat_name_is_the_message(model):
    model.error = RuntimeError("x")
    assert _rewrite(chat_name=None) == "what it can do for me"
    assert _rewrite(chat_name="  ") == "what it can do for me"


def test_no_turn_key_or_fake_mode_means_no_model_call(model, monkeypatch):
    assert _rewrite(config={"configurable": {"model": GPT}}) == "OWASP ZAP what it can do for me"
    from app.core.config import settings
    monkeypatch.setattr(settings, "ENGINE_FAKE_LLM", True)
    assert _rewrite() == "OWASP ZAP what it can do for me"
    assert model.built == []
