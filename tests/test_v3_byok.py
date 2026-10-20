"""V3 key-only model resolver (app-shell plan Task 6).

V3 board turns build their chat model from the user's own stored key for the
model's provider, never from a server env key. No DB and no network here: the
key lookup is monkeypatched and the LangChain clients are only constructed.
"""

import re
from pathlib import Path

import pytest

from app.agent.helpers import v3_llm
from app.core.config import settings

# Every env var a provider SDK or the V2 router could read a key or endpoint from.
# Each is set to a sentinel; a V3 client must never pick one up.
ENV_KEYS = (
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
    "GROQ_API_KEY", "DEEPSEEK_API_KEY", "MISTRAL_API_KEY", "MOONSHOT_API_KEY",
    "NVIDIA_API_KEY", "LITELLM_API_KEY",
)
ENV_BASES = (
    "OPENAI_API_BASE", "OPENAI_BASE_URL", "ANTHROPIC_API_URL", "ANTHROPIC_BASE_URL",
    "GROQ_API_BASE",
)
ENV_SENTINEL = "env-key"
ENV_BASE_SENTINEL = "https://env-base.invalid/v1"

# Must equal the engineId values of MODELS in components/board/lib/catalog.ts
# (ContextTree-src, after app-shell plan Task 2). Change both files together.
BOARD_MODEL_IDS = {
    "claude": "anthropic/claude-sonnet-4-20250514",
    "gpt": "openai/gpt-5-mini",
    "gemini": "gemini/gemini-3-flash-preview",
    "llama": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "deepseek": "deepseek/deepseek-chat",
    "mistral": "mistral/mistral-large-latest",
    "kimi": "moonshot/kimi-k2-0905-preview",
}

EXPECTED = {
    # id: (provider, client class name, secret attr, wire model attr, wire model)
    "anthropic/claude-sonnet-4-20250514":
        ("anthropic", "ChatAnthropic", "anthropic_api_key", "model", "claude-sonnet-4-20250514"),
    "openai/gpt-5-mini":
        ("openai", "ChatOpenAI", "openai_api_key", "model_name", "gpt-5-mini"),
    "gemini/gemini-3-flash-preview":
        ("google", "ChatGoogleGenerativeAI", "google_api_key", "model", "models/gemini-3-flash-preview"),
    "meta-llama/llama-4-maverick-17b-128e-instruct":
        ("groq", "ChatGroq", "groq_api_key", "model_name", "meta-llama/llama-4-maverick-17b-128e-instruct"),
    "deepseek/deepseek-chat":
        ("deepseek", "ChatOpenAI", "openai_api_key", "model_name", "deepseek-chat"),
    "mistral/mistral-large-latest":
        ("mistral", "ChatOpenAI", "openai_api_key", "model_name", "mistral-large-latest"),
    "moonshot/kimi-k2-0905-preview":
        ("moonshot", "ChatOpenAI", "openai_api_key", "model_name", "kimi-k2-0905-preview"),
}


class _Keys(set):
    calls: list


@pytest.fixture
def keys(monkeypatch):
    """The providers the user has a stored key for; lookups return 'user-<provider>'."""
    stored = _Keys(v3_llm.V3_PROVIDERS)
    stored.calls = calls = []

    def fake_lookup(user_id, provider):
        calls.append((user_id, provider))
        return f"user-{provider}" if provider in stored else None

    monkeypatch.setattr(v3_llm, "get_user_provider_api_key", fake_lookup)
    monkeypatch.setattr(settings, "ENGINE_FAKE_LLM", False)
    for name in ENV_KEYS:
        monkeypatch.setenv(name, ENV_SENTINEL)
    for name in ENV_BASES:
        monkeypatch.setenv(name, ENV_BASE_SENTINEL)
    return stored


def test_routes_are_exactly_the_board_models():
    assert set(v3_llm.V3_ROUTES) == set(BOARD_MODEL_IDS.values())
    assert set(v3_llm.V3_ROUTES) == set(EXPECTED)
    assert {r.provider for r in v3_llm.V3_ROUTES.values()} == set(v3_llm.V3_PROVIDERS)
    assert v3_llm.V3_PROVIDERS == (
        "anthropic", "openai", "google", "groq", "deepseek", "mistral", "moonshot",
    )
    assert v3_llm.OPENAI_COMPAT_BASE == {
        "deepseek": "https://api.deepseek.com/v1",
        "mistral": "https://api.mistral.ai/v1",
        "moonshot": "https://api.moonshot.ai/v1",
    }


@pytest.mark.parametrize("model_id", sorted(EXPECTED))
def test_each_board_model_builds_its_client_on_the_user_key(keys, model_id):
    provider, cls, secret_attr, model_attr, wire = EXPECTED[model_id]
    llm = v3_llm.get_v3_llm(model_id, "me@example.test")

    assert type(llm).__name__ == cls
    secret = getattr(llm, secret_attr).get_secret_value()
    assert secret == f"user-{provider}"
    assert secret != ENV_SENTINEL
    assert getattr(llm, model_attr) == wire
    assert llm.max_retries == 1
    assert keys.calls == [("me@example.test", provider)]


@pytest.mark.parametrize("provider", ["deepseek", "mistral", "moonshot"])
def test_openai_compatible_providers_use_their_own_base_url(keys, provider):
    model_id = next(m for m, r in v3_llm.V3_ROUTES.items() if r.provider == provider)
    llm = v3_llm.get_v3_llm(model_id, "me@example.test")
    assert llm.openai_api_base == v3_llm.OPENAI_COMPAT_BASE[provider]


def test_first_party_clients_ignore_base_url_env(keys):
    openai = v3_llm.get_v3_llm("openai/gpt-5-mini", "me@example.test")
    assert openai.openai_api_base == "https://api.openai.com/v1"
    claude = v3_llm.get_v3_llm("anthropic/claude-sonnet-4-20250514", "me@example.test")
    assert claude.anthropic_api_url == "https://api.anthropic.com"
    llama = v3_llm.get_v3_llm("meta-llama/llama-4-maverick-17b-128e-instruct", "me@example.test")
    assert llama.groq_api_base == "https://api.groq.com"


def test_no_stored_key_is_missing_key_even_with_env_key(keys, monkeypatch):
    keys.clear()
    monkeypatch.setenv("ANTHROPIC_API_KEY", ENV_SENTINEL)
    with pytest.raises(v3_llm.MissingKeyError) as err:
        v3_llm.get_v3_llm("anthropic/claude-sonnet-4-20250514", "me@example.test")
    assert err.value.provider == "anthropic"
    with pytest.raises(v3_llm.MissingKeyError) as err:
        v3_llm.v3_require_key("anthropic/claude-sonnet-4-20250514", "me@example.test")
    assert err.value.provider == "anthropic"


def test_missing_key_is_per_provider(keys):
    keys.clear()
    keys.add("openai")
    route, key = v3_llm.v3_require_key("openai/gpt-5-mini", "me@example.test")
    assert route == v3_llm.Route("openai", "gpt-5-mini")
    assert key == "user-openai"
    with pytest.raises(v3_llm.MissingKeyError) as err:
        v3_llm.v3_require_key("gemini/gemini-3-flash-preview", "me@example.test")
    assert err.value.provider == "google"


def test_missing_key_does_not_build_a_client(keys, monkeypatch):
    keys.clear()

    def boom(*a, **k):
        raise AssertionError("client must not be built without a user key")

    monkeypatch.setattr("langchain_anthropic.ChatAnthropic", boom)
    with pytest.raises(v3_llm.MissingKeyError):
        v3_llm.get_v3_llm("anthropic/claude-sonnet-4-20250514", "me@example.test")


@pytest.mark.parametrize(
    "legacy, provider, wire",
    [
        ("openai/gpt-oss-120b", "groq", "openai/gpt-oss-120b"),
        ("openai/gpt-oss-20b", "groq", "openai/gpt-oss-20b"),
        ("deepseek-ai/deepseek-v4-pro", "deepseek", "deepseek-chat"),
        ("deepseek-ai/deepseek-v3.1", "deepseek", "deepseek-chat"),
        ("mistralai/mistral-large-3-675b-instruct-2512", "mistral", "mistral-large-latest"),
        ("moonshotai/kimi-k2-instruct", "moonshot", "kimi-k2-0905-preview"),
        ("moonshotai/kimi-k2.6", "moonshot", "kimi-k2-0905-preview"),
        ("anthropic/claude-sonnet-4-5", "anthropic", "claude-sonnet-4-20250514"),
        ("anthropic/claude-opus-4-1", "anthropic", "claude-opus-4-1-20250805"),
        ("openai/gpt-5", "openai", "gpt-5"),
        ("openai/gpt-5.4-mini", "openai", "gpt-5-mini"),
        ("gemini/gemini-2.5-pro", "google", "models/gemini-2.5-pro"),
        ("gemini/gemini-1.5-flash", "google", "models/gemini-2.5-flash"),
    ],
)
def test_legacy_ids_route_to_the_matching_provider(legacy, provider, wire):
    assert v3_llm.v3_route(legacy) == v3_llm.Route(provider, wire)


def test_legacy_id_builds_on_the_new_provider_key(keys):
    llm = v3_llm.get_v3_llm("moonshotai/kimi-k2-instruct", "me@example.test")
    assert llm.openai_api_base == "https://api.moonshot.ai/v1"
    assert llm.openai_api_key.get_secret_value() == "user-moonshot"
    assert llm.model_name == "kimi-k2-0905-preview"


@pytest.mark.parametrize(
    "model_id",
    [
        "z-ai/glm-5.2", "z-ai/glm-4.7", "groq/compound", "groq/compound-mini",
        "litellm/x", "litellm/custom", "nvidia/whatever", "llama-3.3-70b-versatile",
        "anthropic/claude-unknown", "openai/gpt-4o", "gemini/not-a-model",
        "", None,
    ],
)
def test_unsupported_ids(model_id):
    with pytest.raises(v3_llm.UnsupportedModelError) as err:
        v3_llm.v3_route(model_id)
    assert err.value.model == model_id


def test_unsupported_id_never_looks_up_a_key(keys):
    with pytest.raises(v3_llm.UnsupportedModelError):
        v3_llm.v3_require_key("z-ai/glm-5.2", "me@example.test")
    assert keys.calls == []


def test_reasoning_model_gets_no_temperature(keys):
    llm = v3_llm.get_v3_llm("openai/gpt-5-mini", "me@example.test", temperature=0.3,
                            max_output_tokens=500)
    assert llm.temperature is None
    assert llm.max_tokens == 500  # ChatOpenAI stores max_completion_tokens here


def test_chat_model_gets_clamped_settings(keys):
    llm = v3_llm.get_v3_llm("deepseek/deepseek-chat", "me@example.test", temperature=5,
                            max_output_tokens=999999)
    assert llm.temperature == 2.0
    assert llm.max_tokens == 32000


@pytest.mark.parametrize(
    "model_id",
    ["anthropic/claude-sonnet-4-20250514", "moonshot/kimi-k2-0905-preview"],
)
def test_providers_with_a_0_to_1_temperature_are_capped(keys, model_id):
    llm = v3_llm.get_v3_llm(model_id, "me@example.test", temperature=1.7)
    assert llm.temperature == 1.0


def test_client_build_error_does_not_carry_the_key(keys, monkeypatch):
    def broken(**kwargs):
        raise ValueError(f"bad input {kwargs}")

    monkeypatch.setattr("langchain_openai.ChatOpenAI", broken)
    with pytest.raises(RuntimeError) as err:
        v3_llm.get_v3_llm("openai/gpt-5-mini", "me@example.test")
    assert "user-openai" not in str(err.value)
    assert err.value.__cause__ is None
    assert err.value.__suppress_context__ is True


class _Status:
    def __init__(self, code):
        self.status_code = code


class _Resp(Exception):
    def __init__(self, code):
        super().__init__("http error")
        self.response = _Status(code)


class PermissionDenied(Exception):
    pass


class Unauthenticated(Exception):
    pass


@pytest.mark.parametrize(
    "exc, expected",
    [
        (_Status(401), True),
        (_Status(403), True),
        (_Status(429), False),
        (_Status(500), False),
        (_Resp(401), True),
        (_Resp(404), False),
        (Exception("400 API_KEY_INVALID"), True),
        (PermissionDenied("nope"), True),
        (Unauthenticated("nope"), True),
        (Exception("timeout"), False),
    ],
)
def test_is_auth_error(exc, expected):
    assert v3_llm.is_auth_error(exc) is expected


def test_is_auth_error_follows_the_cause_chain():
    try:
        try:
            raise _Resp(401)
        except _Resp as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as outer:
        assert v3_llm.is_auth_error(outer) is True


def test_module_reads_no_env_and_no_v2_byok():
    text = Path(v3_llm.__file__).read_text(encoding="utf-8")
    assert "getenv" not in text
    assert "os.environ" not in text
    assert "import os" not in text
    # V2's env-first helpers, as whole words (the test file name itself is fine).
    assert not re.search(r"\b_byok\b", text)
    assert not re.search(r"\b_env\b", text)


def test_fake_mode_returns_the_fake_model_without_a_key(keys, monkeypatch):
    keys.clear()
    monkeypatch.setattr(settings, "ENGINE_FAKE_LLM", True)
    llm = v3_llm.get_v3_llm("anthropic/claude-sonnet-4-20250514", "me@example.test")
    assert llm.invoke("hi").content == "Fake answer to: hi"
    route, _ = v3_llm.v3_require_key("anthropic/claude-sonnet-4-20250514", "me@example.test")
    assert route.provider == "anthropic"
    assert keys.calls == []


def test_fake_mode_still_rejects_unsupported_models(monkeypatch):
    monkeypatch.setattr(settings, "ENGINE_FAKE_LLM", True)
    with pytest.raises(v3_llm.UnsupportedModelError):
        v3_llm.v3_require_key("z-ai/glm-5.2", "me@example.test")
