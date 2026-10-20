"""
Key-only chat models for V3 board turns.

A V3 turn runs on the user's own stored key for the chosen model's provider
(owner decision 2026-10-20, app-shell plan Task 6). There is no server-key
fallback:

- This module reads no environment variable. Every client gets its key and
  endpoint passed in explicitly.
- Provider SDKs still read some server env on their own (OPENAI_ORG_ID,
  OPENAI_PROJECT_ID, OPENAI_CUSTOM_HEADERS, ANTHROPIC_CUSTOM_HEADERS). Every
  httpx-based client (Anthropic, OpenAI, Groq, DeepSeek, Mistral, Moonshot) gets
  its own httpx clients with a request hook that strips those headers and sets
  the auth header to the user's key, so nothing from the server env rides along.
- Gemini (google-genai, which builds its own HTTP clients) fails closed when the
  server env switches it to Vertex AI.
- Proxies still apply: httpx trust_env, plus OPENAI_PROXY / ANTHROPIC_PROXY as
  LangChain reads them. A proxy routes traffic; it adds no credential.

V2 turns keep `get_llm`, `validate_model_access` and its key helpers unchanged.

Keys never appear in logs, exception text or exception chains raised from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any

from app.agent.helpers.v3_keys import fetch_user_provider_api_key
from app.agent.helpers.get_llm import (
    _ANTHROPIC_ALIASES,
    _GEMINI_ALIASES,
    _OPENAI_ALIASES,
    _build_fake_llm,
    _clamp_max_output_tokens,
    _clamp_temperature,
    _is_reasoning_model,
)
from app.core.config import settings
from app.core.logger import logger

V3_PROVIDERS = ("anthropic", "openai", "google", "groq", "deepseek", "mistral", "moonshot")


@dataclass(frozen=True)
class Route:
    provider: str
    wire_model: str


# Exactly one route per `MODELS` engineId in components/board/lib/catalog.ts
# (ContextTree-src). Change both files together; tests/test_v3_byok.py pins it.
V3_ROUTES: dict[str, Route] = {
    "anthropic/claude-sonnet-4-20250514": Route("anthropic", "claude-sonnet-4-20250514"),
    "openai/gpt-5-mini": Route("openai", "gpt-5-mini"),
    "gemini/gemini-3-flash-preview": Route("google", "models/gemini-3-flash-preview"),
    "meta-llama/llama-4-maverick-17b-128e-instruct": Route(
        "groq", "meta-llama/llama-4-maverick-17b-128e-instruct"
    ),
    "deepseek/deepseek-chat": Route("deepseek", "deepseek-chat"),
    "mistral/mistral-large-latest": Route("mistral", "mistral-large-latest"),
    "moonshot/kimi-k2-0905-preview": Route("moonshot", "kimi-k2-0905-preview"),
}

OPENAI_COMPAT_BASE = {
    "deepseek": "https://api.deepseek.com/v1",
    "mistral": "https://api.mistral.ai/v1",
    "moonshot": "https://api.moonshot.ai/v1",  # international platform
}

# First-party endpoints, passed explicitly so OPENAI_BASE_URL / ANTHROPIC_BASE_URL /
# GROQ_API_BASE in the server env can never redirect a user's key elsewhere.
_FIRST_PARTY_BASE = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "groq": "https://api.groq.com",
}

# Providers whose API rejects temperature above 1.0.
_TEMPERATURE_CEILING = {"anthropic": 1.0, "moonshot": 1.0}

_LEGACY_GROQ = {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}


class MissingKeyError(Exception):
    """The user has no stored key for the provider this model needs."""

    def __init__(self, provider: str):
        super().__init__(provider)
        self.provider = provider


class UnsupportedModelError(Exception):
    """The model id has no V3 route."""

    def __init__(self, model: str | None):
        super().__init__(model)
        self.model = model


class KeyRejectedError(Exception):
    """The provider turned the user's key down (401/403 or an invalid-key error)."""

    def __init__(self, provider: str):
        super().__init__(provider)
        self.provider = provider


class KeyStoreError(Exception):
    """The key store could not answer (DB down, row won't decrypt). Not "no key"."""

    def __init__(self, provider: str):
        super().__init__(provider)
        self.provider = provider


class V3ConfigError(RuntimeError):
    """The server environment rules out a user-key-only client (an operator fix)."""


def v3_route(model_id: str | None) -> Route:
    """The route for a V3 model id: the board's seven, then legacy ids on old chats."""
    if not model_id:
        raise UnsupportedModelError(model_id)
    route = V3_ROUTES.get(model_id)
    if route is not None:
        return route

    if model_id in _LEGACY_GROQ:
        return Route("groq", model_id)
    if model_id.startswith("deepseek-ai/"):
        return Route("deepseek", "deepseek-chat")
    if model_id.startswith("mistralai/"):
        return Route("mistral", "mistral-large-latest")
    if model_id.startswith("moonshotai/"):
        return Route("moonshot", "kimi-k2-0905-preview")
    if model_id.startswith("anthropic/") and model_id in _ANTHROPIC_ALIASES:
        return Route("anthropic", _ANTHROPIC_ALIASES[model_id])
    if model_id.startswith("openai/gpt-5") and model_id in _OPENAI_ALIASES:
        return Route("openai", _OPENAI_ALIASES[model_id])
    if model_id.startswith("gemini/"):
        wire = _GEMINI_ALIASES.get(model_id[len("gemini/"):])
        if wire:
            return Route("google", wire)

    raise UnsupportedModelError(model_id)


def v3_require_key(model_id: str | None, user_email: str | None) -> tuple[Route, str]:
    """The route and the user's stored key for its provider.

    Raises UnsupportedModelError before any key lookup, MissingKeyError when the
    user has no stored key, and KeyStoreError when the store cannot answer.
    Fake mode routes the model but skips the key check (as `validate_model_access`
    does) and returns an empty key.
    """
    route = v3_route(model_id)
    if settings.ENGINE_FAKE_LLM:
        return route, ""
    store_failed = False
    key = None
    try:
        key = fetch_user_provider_api_key(user_email, route.provider)
    except Exception:
        store_failed = True
    if store_failed:  # raised outside the except block: no chain to the cause
        raise KeyStoreError(route.provider)
    if not key:
        raise MissingKeyError(route.provider)
    return route, key


class _HeaderGuard:
    """httpx request hook: the request carries the user's key and no server header.

    `strip` starts with the OpenAI org/project headers and learns, per SDK client,
    the header names that SDK took from env custom-header variables.
    """

    def __init__(self, auth_header: str, auth_value: str, drop: tuple[str, ...] = ()):
        self._auth_header = auth_header
        self._auth_value = auth_value
        self.strip: set[str] = {"openai-organization", "openai-project", *drop}

    def __repr__(self) -> str:  # never show the key
        return "<V3 header guard>"

    def learn(self, sdk_client: Any, intended: Any) -> None:
        custom = getattr(sdk_client, "_custom_headers", None) or {}
        keep = {str(k).lower() for k in (intended or {})}
        self.strip |= {str(k).lower() for k in custom} - keep

    def apply(self, request) -> None:
        for name in self.strip:
            if name in request.headers:
                del request.headers[name]
        request.headers[self._auth_header] = self._auth_value

    async def apply_async(self, request) -> None:
        self.apply(request)


def _hooked_clients(sdk: Any, guard: _HeaderGuard, proxy: str | None):
    """The SDK's own default httpx clients (its timeouts and limits) plus our hook."""
    extra = {"proxy": proxy} if proxy else {}
    return (
        sdk.DefaultHttpxClient(event_hooks={"request": [guard.apply]}, **extra),
        sdk.DefaultAsyncHttpxClient(event_hooks={"request": [guard.apply_async]}, **extra),
    )


@lru_cache(maxsize=1)
def _anthropic_class():
    """ChatAnthropic whose SDK clients use our hooked httpx clients.

    LangChain builds Anthropic's httpx clients itself (cached and shared across
    instances), so the hook goes in through a subclass, not a constructor arg.
    """
    import anthropic
    from langchain_anthropic import ChatAnthropic
    from pydantic import PrivateAttr

    class V3ChatAnthropic(ChatAnthropic):
        _v3_guard: Any = PrivateAttr(default=None)

        @cached_property
        def _client(self) -> anthropic.Client:
            params = self._client_params
            http_client, _ = _hooked_clients(anthropic, self._v3_guard, self.anthropic_proxy)
            client = anthropic.Client(**params, http_client=http_client)
            self._v3_guard.learn(client, params.get("default_headers"))
            return client

        @cached_property
        def _async_client(self) -> anthropic.AsyncClient:
            params = self._client_params
            _, http_client = _hooked_clients(anthropic, self._v3_guard, self.anthropic_proxy)
            client = anthropic.AsyncClient(**params, http_client=http_client)
            self._v3_guard.learn(client, params.get("default_headers"))
            return client

    return V3ChatAnthropic


def _build_client(route: Route, key: str, temperature, max_output_tokens):
    provider = route.provider
    temp = _clamp_temperature(temperature)
    ceiling = _TEMPERATURE_CEILING.get(provider)
    if ceiling is not None:
        temp = min(temp, ceiling)
    max_tokens = _clamp_max_output_tokens(max_output_tokens)

    if provider == "anthropic":
        kwargs = {
            "model": route.wire_model,
            "api_key": key,
            "base_url": _FIRST_PARTY_BASE["anthropic"],
            "temperature": temp,
            "max_retries": 1,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        llm = _anthropic_class()(**kwargs)
        llm._v3_guard = _HeaderGuard("x-api-key", key, drop=("authorization",))
        return llm

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs = {
            "model": route.wire_model,
            "google_api_key": key,
            "vertexai": False,
            "temperature": temp,
            "max_retries": 1,
        }
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens
        llm = ChatGoogleGenerativeAI(**kwargs)
        api_client = getattr(getattr(llm, "client", None), "_api_client", None)
        if api_client is None or getattr(api_client, "vertexai", None):
            # GOOGLE_GENAI_USE_VERTEXAI / _ENTERPRISE in the server env beat vertexai=False.
            raise V3ConfigError(
                "Gemini is switched to Vertex AI by the server environment; "
                "V3 turns need the Gemini API on the user's key"
            )
        return llm

    if provider == "groq":
        import groq
        from langchain_groq import ChatGroq

        guard = _HeaderGuard("authorization", f"Bearer {key}")
        http_client, http_async_client = _hooked_clients(groq, guard, None)
        kwargs = {
            "model_name": route.wire_model,
            "api_key": key,
            "base_url": _FIRST_PARTY_BASE["groq"],
            "temperature": temp,
            "max_retries": 1,
            "http_client": http_client,
            "http_async_client": http_async_client,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        llm = ChatGroq(**kwargs)
        for resource in (llm.client, llm.async_client):
            guard.learn(getattr(resource, "_client", None), llm.default_headers)
        return llm

    # openai, deepseek, mistral, moonshot: OpenAI-compatible chat completions.
    import openai
    from langchain_openai import ChatOpenAI

    guard = _HeaderGuard("authorization", f"Bearer {key}")
    # Keep the proxy LangChain would have used (OPENAI_PROXY). LangChain refuses it
    # next to our own httpx clients, so it moves onto them.
    proxy = ChatOpenAI.model_fields["openai_proxy"].get_default(call_default_factory=True)
    http_client, http_async_client = _hooked_clients(openai, guard, proxy)

    base_url = OPENAI_COMPAT_BASE.get(provider) or _FIRST_PARTY_BASE["openai"]
    is_reasoning = _is_reasoning_model(route.wire_model)
    kwargs = {
        "model": route.wire_model,
        "api_key": key,
        "base_url": base_url,
        "max_retries": 1,
        "http_client": http_client,
        "http_async_client": http_async_client,
        "openai_proxy": None,
    }
    if not is_reasoning:
        kwargs["temperature"] = temp
    if max_tokens is not None:
        kwargs["max_completion_tokens" if is_reasoning else "max_tokens"] = max_tokens
    llm = ChatOpenAI(**kwargs)
    for sdk_client in (llm.root_client, llm.root_async_client):
        guard.learn(sdk_client, llm.default_headers)
    return llm


def get_v3_llm(
    model_id: str | None,
    user_email: str | None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    user_key: str | None = None,
):
    """A chat model for a V3 turn, built on the user's own key only.

    Pass `user_key` when the caller already resolved it with `v3_require_key`
    (one store lookup per turn); otherwise it is looked up here.
    """
    route = v3_route(model_id)
    if settings.ENGINE_FAKE_LLM:
        return _build_fake_llm()
    if not user_key:
        _, user_key = v3_require_key(model_id, user_email)
    failure: str | None = None
    try:
        llm = _build_client(route, user_key, temperature, max_output_tokens)
    except V3ConfigError:
        raise
    except Exception as exc:
        # Validation errors can echo constructor input (the key); keep only the type.
        failure = type(exc).__name__
    if failure is not None:  # raised outside the except block: no chain to the cause
        raise RuntimeError(f"Could not build the {route.provider} chat model ({failure})")
    logger.info("V3 model %s on the user's %s key", route.wire_model, route.provider)
    return llm


PROVIDER_NAME = {
    "anthropic": "Anthropic", "openai": "OpenAI", "google": "Google", "groq": "Groq",
    "deepseek": "DeepSeek", "mistral": "Mistral", "moonshot": "Moonshot",
}

# The run-config entry (config["configurable"][TURN_KEY_CONFIG]) that carries a TurnKey.
TURN_KEY_CONFIG = "v3_turn_key"


def _redacted_turn_key(provider: str) -> "TurnKey":
    return TurnKey(provider, "")


class TurnKey:
    """The user's key for one V3 turn.

    It is looked up once per turn (`v3_require_key`, in the route) and reaches every LLM call of
    that turn (answer, rolling summary, branch-start summary) through the run config, never
    through graph state, which is checkpointed. It is deliberately not a str: LangGraph copies
    str/int/float/bool `configurable` values into stored checkpoint metadata and trace metadata,
    and skips other objects. It prints as `<TurnKey provider>`, and pickling it drops the secret.
    """

    __slots__ = ("provider", "_secret")

    def __init__(self, provider: str, secret: str | None):
        self.provider = provider
        self._secret = secret or ""

    def reveal(self) -> str:
        return self._secret

    def __repr__(self) -> str:
        return f"<TurnKey {self.provider}>"

    __str__ = __repr__

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (_redacted_turn_key, (self.provider,))


def turn_key_from(config) -> TurnKey | None:
    """The TurnKey a V3 run config carries, or None."""
    configurable = (config or {}).get("configurable") or {}
    found = configurable.get(TURN_KEY_CONFIG)
    return found if isinstance(found, TurnKey) else None


def key_rejected_frame(provider: str) -> dict:
    """The SSE payload for a provider that refused the user's key (contract amendment 2026-10-20)."""
    name = PROVIDER_NAME.get(provider, provider)
    return {"error": f"{name} turned this key down.", "code": "key_rejected", "provider": provider}


def invoke_on_user_key(runnable, prompt, provider: str):
    """`runnable.invoke(prompt)` for a V3 LLM call; a refused key becomes KeyRejectedError.

    Only an error raised by this call itself is read as "key refused". Web search, tools,
    embeddings and store calls never pass through here, so their 401s stay ordinary errors.
    """
    refused = False
    try:
        return runnable.invoke(prompt)
    except Exception as exc:
        if not is_auth_error(exc):
            raise
        refused = True
    if refused:  # raised outside the except block: no chain to the provider's error text
        raise KeyRejectedError(provider)


_AUTH_STATUS = (401, 403)
_AUTH_EXC_NAMES = {"PermissionDenied", "Unauthenticated", "AuthenticationError", "PermissionDeniedError"}


def _single_is_auth_error(exc) -> bool:
    for holder in (exc, getattr(exc, "response", None)):
        status = getattr(holder, "status_code", None)
        if isinstance(status, int) and status in _AUTH_STATUS:
            return True
    # google-genai APIError carries the HTTP status as `code`.
    code = getattr(exc, "code", None)
    if isinstance(code, int) and not isinstance(code, bool) and code in _AUTH_STATUS:
        return True
    if type(exc).__name__ in _AUTH_EXC_NAMES:
        return True
    return "API_KEY_INVALID" in str(exc)


def is_auth_error(exc) -> bool:
    """True when a provider error means the key was refused (401/403, invalid key).

    Follows `__cause__` / `__context__` because LangChain wraps SDK errors.
    """
    seen: set[int] = set()
    current = exc
    while current is not None and id(current) not in seen and len(seen) < 8:
        seen.add(id(current))
        if _single_is_auth_error(current):
            return True
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    return False
