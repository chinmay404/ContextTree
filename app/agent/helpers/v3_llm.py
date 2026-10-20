"""
Key-only chat models for V3 board turns.

A V3 turn runs on the user's own stored key for the chosen model's provider
(owner decision 2026-10-20, app-shell plan Task 6). There is no server-key
fallback: this module reads no environment variable, and every client gets its
key and endpoint passed in explicitly so an SDK default cannot pick one up.

V2 turns keep `get_llm`, `validate_model_access` and its key helpers unchanged.

Keys never appear in logs, exception text or exception chains raised from here.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.agent.helpers.byok import get_user_provider_api_key
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
    user has no stored key. Fake mode routes the model but skips the key check
    (as `validate_model_access` does) and returns an empty key.
    """
    route = v3_route(model_id)
    if settings.ENGINE_FAKE_LLM:
        return route, ""
    key = get_user_provider_api_key(user_email, route.provider)
    if not key:
        raise MissingKeyError(route.provider)
    return route, key


def _build_client(route: Route, key: str, temperature, max_output_tokens):
    provider = route.provider
    temp = _clamp_temperature(temperature)
    ceiling = _TEMPERATURE_CEILING.get(provider)
    if ceiling is not None:
        temp = min(temp, ceiling)
    max_tokens = _clamp_max_output_tokens(max_output_tokens)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {
            "model": route.wire_model,
            "api_key": key,
            "base_url": _FIRST_PARTY_BASE["anthropic"],
            "temperature": temp,
            "max_retries": 1,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatAnthropic(**kwargs)

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
        return ChatGoogleGenerativeAI(**kwargs)

    if provider == "groq":
        from langchain_groq import ChatGroq

        kwargs = {
            "model_name": route.wire_model,
            "api_key": key,
            "base_url": _FIRST_PARTY_BASE["groq"],
            "temperature": temp,
            "max_retries": 1,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatGroq(**kwargs)

    # openai, deepseek, mistral, moonshot: OpenAI-compatible chat completions.
    from langchain_openai import ChatOpenAI

    base_url = OPENAI_COMPAT_BASE.get(provider) or _FIRST_PARTY_BASE["openai"]
    is_reasoning = _is_reasoning_model(route.wire_model)
    kwargs = {
        "model": route.wire_model,
        "api_key": key,
        "base_url": base_url,
        "max_retries": 1,
    }
    if not is_reasoning:
        kwargs["temperature"] = temp
    if max_tokens is not None:
        kwargs["max_completion_tokens" if is_reasoning else "max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


def get_v3_llm(
    model_id: str | None,
    user_email: str | None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
):
    """A chat model for a V3 turn, built on the user's own key only."""
    route = v3_route(model_id)
    if settings.ENGINE_FAKE_LLM:
        return _build_fake_llm()
    _, key = v3_require_key(model_id, user_email)
    try:
        llm = _build_client(route, key, temperature, max_output_tokens)
    except Exception as exc:
        # Validation errors can echo constructor input (the key); drop the chain.
        raise RuntimeError(
            f"Could not build the {route.provider} chat model ({type(exc).__name__})"
        ) from None
    logger.info("V3 model %s on the user's %s key", route.wire_model, route.provider)
    return llm


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
