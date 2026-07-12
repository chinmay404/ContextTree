"""
LLM provider routing.

Model naming convention (sent from frontend):
  - gemini/<model>        → Google Gemini via langchain_google_genai
  - NVIDIA-hosted ids     → NVIDIA NIM via langchain_openai (OpenAI-compatible)
  - everything else       → Groq via langchain_groq
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

from app.agent.helpers.byok import get_user_provider_api_key, get_user_provider_credentials
from app.core.logger import logger

load_dotenv()

# ── Groq defaults ──────────────────────────────────────────────────────────────
_GROQ_DEFAULT = os.getenv("DEFAULT_GROQ_MODEL") or "openai/gpt-oss-120b"
_GROQ_SENTINEL = {"None", "null", "default-model-name", "", None}
_OPENAI_DEFAULT = os.getenv("DEFAULT_OPENAI_MODEL") or "gpt-5"
_OPENAI_SENTINEL = {"None", "null", "default-model-name", "", None}
_NVIDIA_DEFAULT = os.getenv("DEFAULT_NVIDIA_MODEL") or "z-ai/glm-5.2"
_NVIDIA_SENTINEL = {"None", "null", "default-model-name", "", None}
_ANTHROPIC_DEFAULT = os.getenv("DEFAULT_ANTHROPIC_MODEL") or "claude-sonnet-4-20250514"
_ANTHROPIC_SENTINEL = {"None", "null", "default-model-name", "", None}
_LITELLM_DEFAULT = os.getenv("DEFAULT_LITELLM_MODEL") or "openrouter/openai/gpt-oss-120b"
_LITELLM_SENTINEL = {"None", "null", "default-model-name", "", "litellm/custom", None}


def _clamp_temperature(value: float | int | None) -> float:
    if value is None:
        return 0.8
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.8
    return max(0.0, min(2.0, numeric))


def _clamp_max_output_tokens(value: int | None) -> int | None:
    if value is None:
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return max(1, min(32000, numeric))


# ── Reasoning-model detection ──────────────────────────────────────────────────
# Reasoning models (GPT-5, o1/o3/o4 mini, R1) reject custom temperature and the
# `max_tokens` field — they internalise sampling and use `max_completion_tokens`.
# We strip those kwargs before initialising the LLM.
_REASONING_PREFIXES = (
    "gpt-5",
    "openai/gpt-5",
    "o1",
    "openai/o1",
    "o3",
    "openai/o3",
    "o4",
    "openai/o4",
    "deepseek-r1",
    "deepseek-ai/deepseek-r1",
)


def _is_reasoning_model(model_name: str | None) -> bool:
    if not model_name:
        return False
    lowered = model_name.lower()
    return any(lowered.startswith(prefix) for prefix in _REASONING_PREFIXES)

# ── Gemini model aliases ───────────────────────────────────────────────────────
# Google currently returns `models/<id>` names from the official models list.
_GEMINI_ALIASES: dict[str, str] = {
    "gemini-3-flash-preview": "models/gemini-3-flash-preview",
    "gemini-3-pro-preview": "models/gemini-3-pro-preview",
    "gemini-3.1-pro-preview": "models/gemini-3.1-pro-preview",
    "gemini-2.5-flash": "models/gemini-2.5-flash",
    "gemini-2.5-pro": "models/gemini-2.5-pro",
    # Legacy IDs are promoted to currently-available models so older nodes keep working.
    "gemini-2.0-flash": "models/gemini-2.5-flash",
    "gemini-1.5-pro": "models/gemini-2.5-pro",
    "gemini-1.5-flash": "models/gemini-2.5-flash",
    "gemini-flash-latest": "models/gemini-flash-latest",
    "gemini-flash-lite-latest": "models/gemini-flash-lite-latest",
    "gemini-pro-latest": "models/gemini-pro-latest",
}

_OPENAI_ALIASES: dict[str, str] = {
    "gpt-5": "gpt-5",
    "gpt-5-mini": "gpt-5-mini",
    "gpt-5.4": "gpt-5",
    "gpt-5.4-mini": "gpt-5-mini",
    "openai/gpt-5": "gpt-5",
    "openai/gpt-5-mini": "gpt-5-mini",
    "openai/gpt-5.4": "gpt-5",
    "openai/gpt-5.4-mini": "gpt-5-mini",
}

_ANTHROPIC_ALIASES: dict[str, str] = {
    "claude-sonnet-4-20250514": "claude-sonnet-4-20250514",
    "claude-sonnet-4-0": "claude-sonnet-4-20250514",
    "claude-sonnet-4-5": "claude-sonnet-4-20250514",
    "anthropic/claude-sonnet-4-20250514": "claude-sonnet-4-20250514",
    "anthropic/claude-sonnet-4-0": "claude-sonnet-4-20250514",
    "anthropic/claude-sonnet-4-5": "claude-sonnet-4-20250514",
    "claude-opus-4-1-20250805": "claude-opus-4-1-20250805",
    "claude-opus-4-1": "claude-opus-4-1-20250805",
    "anthropic/claude-opus-4-1-20250805": "claude-opus-4-1-20250805",
    "anthropic/claude-opus-4-1": "claude-opus-4-1-20250805",
}

# NIM rotated its catalog (July 2026): deepseek v3.x -> v4, glm 4.7 -> 5.2;
# kimi ids are listed but not invocable on our key (404), so they alias to
# GLM-5.2 — old nodes keep responding instead of dying. Verified working:
# z-ai/glm-5.2, mistralai/mistral-large-3-675b-instruct-2512.
_NVIDIA_ALIASES: dict[str, str] = {
    "moonshotai/kimi-k2-instruct": "z-ai/glm-5.2",
    "moonshotai/kimi-k2-instruct-0905": "z-ai/glm-5.2",
    "moonshotai/kimi-k2.6": "z-ai/glm-5.2",
    "z-ai/glm-4.7": "z-ai/glm-5.2",
    "z-ai/glm4.7": "z-ai/glm-5.2",
    "z-ai/glm4_7": "z-ai/glm-5.2",
    "z-ai/glm-5.2": "z-ai/glm-5.2",
    "deepseek-ai/deepseek-v3.1": "deepseek-ai/deepseek-v4-flash",
    "deepseek-ai/deepseek-v3_1": "deepseek-ai/deepseek-v4-flash",
    "deepseek-ai/deepseek-v3.2": "deepseek-ai/deepseek-v4-flash",
    "deepseek-ai/deepseek-v3_2": "deepseek-ai/deepseek-v4-flash",
    "deepseek-ai/deepseek-v4-flash": "deepseek-ai/deepseek-v4-flash",
    "deepseek-ai/deepseek-v4-pro": "deepseek-ai/deepseek-v4-pro",
    "mistralai/mistral-large-3-675b-instruct-2512": "mistralai/mistral-large-3-675b-instruct-2512",
}

_NVIDIA_PREFIXES = (
    "moonshotai/",
    "z-ai/",
    "deepseek-ai/",
    "mistralai/",
    "nvidia/",
)


def _is_gemini(model_name: str | None) -> bool:
    if not model_name:
        return False
    return model_name.startswith("gemini/") or model_name.startswith("gemini-")


def _is_openai_direct(model_name: str | None) -> bool:
    if not model_name:
        return False
    return model_name.startswith("openai/") and not model_name.startswith("openai/gpt-oss")


def _is_anthropic(model_name: str | None) -> bool:
    if not model_name:
        return False
    return model_name.startswith("anthropic/")


def _is_nvidia(model_name: str | None) -> bool:
    if not model_name:
        return False
    return model_name.startswith(_NVIDIA_PREFIXES)


def _is_litellm(model_name: str | None) -> bool:
    if not model_name:
        return False
    return model_name.startswith("litellm/")


def _normalize_openai_model_name(model_name: str | None) -> str:
    if model_name in _OPENAI_SENTINEL:
        return _OPENAI_DEFAULT

    bare = model_name or _OPENAI_DEFAULT
    if bare.startswith("openai/"):
        bare = bare[len("openai/"):]

    return _OPENAI_ALIASES.get(model_name or bare, _OPENAI_ALIASES.get(bare, bare))


def _normalize_gemini_model_name(model_name: str | None) -> str:
    bare = model_name or os.getenv("DEFAULT_GEMINI_MODEL") or "gemini-3-flash-preview"
    if bare.startswith("gemini/"):
        bare = bare[len("gemini/"):]
    if bare.startswith("models/"):
        return bare

    resolved = _GEMINI_ALIASES.get(bare, bare)
    return resolved if resolved.startswith("models/") else f"models/{resolved}"


def _normalize_anthropic_model_name(model_name: str | None) -> str:
    if model_name in _ANTHROPIC_SENTINEL:
        return _ANTHROPIC_DEFAULT
    return _ANTHROPIC_ALIASES.get(model_name or _ANTHROPIC_DEFAULT, model_name or _ANTHROPIC_DEFAULT)


def _normalize_nvidia_model_name(model_name: str | None) -> str:
    if model_name in _NVIDIA_SENTINEL:
        return _NVIDIA_DEFAULT
    return _NVIDIA_ALIASES.get(model_name or _NVIDIA_DEFAULT, model_name or _NVIDIA_DEFAULT)


def _normalize_litellm_model_name(model_name: str | None) -> str:
    if model_name in _LITELLM_SENTINEL:
        return _LITELLM_DEFAULT

    bare = model_name or _LITELLM_DEFAULT
    if bare.startswith("litellm/"):
        bare = bare[len("litellm/"):]

    return bare or _LITELLM_DEFAULT


def _resolve_provider_key(
    provider: str,
    env_key_name: str,
    user_id: str | None = None,
):
    env_key = os.getenv(env_key_name)
    if env_key:
        return env_key

    return get_user_provider_api_key(user_id, provider)


def validate_model_access(model_name: str | None, user_id: str | None = None) -> str | None:
    if _is_litellm(model_name):
        credentials = get_user_provider_credentials(user_id, "litellm")
        env_key = os.getenv("LITELLM_API_KEY")
        env_api_base = os.getenv("LITELLM_API_BASE")
        api_key = env_key or (credentials or {}).get("api_key")
        api_base = env_api_base or ((credentials or {}).get("metadata") or {}).get("apiBase")
        if api_key or api_base:
            return None
        return "Connect your LiteLLM credential or private endpoint to use custom models."

    if _is_openai_direct(model_name):
        if _resolve_provider_key("openai", "OPENAI_API_KEY", user_id):
            return None
        return "Connect your OpenAI API key to use GPT models."

    if _is_anthropic(model_name):
        if _resolve_provider_key("anthropic", "ANTHROPIC_API_KEY", user_id):
            return None
        return "Connect your Anthropic API key to use Claude models."

    return None


def get_gemini_llm(
    model_name: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
):
    """
    Returns a ChatGoogleGenerativeAI instance.
    model_name may be:
      - "gemini/gemini-3-flash-preview"  (frontend format)
      - "gemini-3-flash-preview"         (bare)
      - None → falls back to DEFAULT_GEMINI_MODEL or Gemini 3 Flash Preview
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not set — cannot initialise Gemini LLM")
        return None

    resolved = _normalize_gemini_model_name(model_name)

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs = {
            "model": resolved,
            "google_api_key": api_key,
            "temperature": _clamp_temperature(temperature),
            "max_retries": 2,
        }
        max_tokens = _clamp_max_output_tokens(max_output_tokens)
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens
        llm = ChatGoogleGenerativeAI(**kwargs)
        logger.info(f"Initialised Gemini LLM: {resolved}")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialise Gemini LLM ({resolved}): {e}")
        return None


def get_openai_llm(
    model_name: str | None = None,
    user_id: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
):
    resolved = _normalize_openai_model_name(model_name)
    api_key = _resolve_provider_key("openai", "OPENAI_API_KEY", user_id)
    if not api_key:
        logger.error("No OpenAI credentials available for %s", resolved)
        return None

    is_reasoning = _is_reasoning_model(resolved) or _is_reasoning_model(model_name)

    try:
        from langchain_openai import ChatOpenAI

        kwargs: dict = {
            "model": resolved,
            "api_key": api_key,
            "max_retries": 2,
        }
        if not is_reasoning:
            kwargs["temperature"] = _clamp_temperature(temperature)
        max_tokens = _clamp_max_output_tokens(max_output_tokens)
        if max_tokens is not None:
            # Reasoning models use `max_completion_tokens`; chat models use `max_tokens`.
            kwargs["max_completion_tokens" if is_reasoning else "max_tokens"] = max_tokens
        llm = ChatOpenAI(**kwargs)
        logger.info(
            "Initialised OpenAI LLM: %s (reasoning=%s)", resolved, is_reasoning
        )
        return llm
    except Exception as e:
        logger.error(f"Failed to initialise OpenAI LLM ({resolved}): {e}")
        return None


def get_groq_llm(
    name: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
):
    """Returns a ChatGroq instance, falling back to the default Groq model."""
    if name in _GROQ_SENTINEL:
        name = _GROQ_DEFAULT

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY not set — cannot initialise Groq LLM")
        return None

    try:
        from langchain_groq import ChatGroq
        kwargs = {
            "temperature": _clamp_temperature(temperature),
            "api_key": str(api_key),
            "model_name": name,
        }
        max_tokens = _clamp_max_output_tokens(max_output_tokens)
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        llm = ChatGroq(**kwargs)
        return llm
    except Exception as e:
        logger.error(f"Failed to initialise Groq LLM ({name}): {e}")
        return None


def get_anthropic_llm(
    model_name: str | None = None,
    user_id: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
):
    resolved = _normalize_anthropic_model_name(model_name)
    api_key = _resolve_provider_key("anthropic", "ANTHROPIC_API_KEY", user_id)
    if not api_key:
        logger.error("No Anthropic credentials available for %s", resolved)
        return None

    try:
        from langchain_anthropic import ChatAnthropic

        kwargs = {
            "model": resolved,
            "api_key": api_key,
            "temperature": _clamp_temperature(temperature),
            "max_retries": 2,
        }
        max_tokens = _clamp_max_output_tokens(max_output_tokens)
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        llm = ChatAnthropic(**kwargs)
        logger.info("Initialised Anthropic LLM: %s", resolved)
        return llm
    except Exception as e:
        logger.error(f"Failed to initialise Anthropic LLM ({resolved}): {e}")
        return None


def get_nvidia_llm(
    model_name: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
):
    """Returns a ChatOpenAI instance pointed at NVIDIA's OpenAI-compatible API."""
    model_name = _normalize_nvidia_model_name(model_name)

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        logger.error("NVIDIA_API_KEY not set — cannot initialise NVIDIA LLM")
        return None

    try:
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": model_name,
            "api_key": api_key,
            "base_url": "https://integrate.api.nvidia.com/v1",
            "temperature": _clamp_temperature(temperature),
            "max_retries": 2,
        }
        max_tokens = _clamp_max_output_tokens(max_output_tokens)
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        llm = ChatOpenAI(**kwargs)
        logger.info(f"Initialised NVIDIA LLM: {model_name}")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialise NVIDIA LLM ({model_name}): {e}")
        return None


def get_litellm_llm(
    model_name: str | None = None,
    user_id: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
):
    """
    Returns a ChatLiteLLM instance for user-supplied LiteLLM model strings.

    LiteLLM forwards calls to whatever upstream provider the user configured.
    Some providers (e.g. OpenAI reasoning models, certain Bedrock/Vertex routes)
    reject `temperature` or `max_tokens`. We try the full kwargs first, then
    progressively drop the optional sampling fields if construction fails so
    the route still works for restrictive providers.
    """
    resolved = _normalize_litellm_model_name(model_name)
    credentials = get_user_provider_credentials(user_id, "litellm") or {}
    metadata = credentials.get("metadata") or {}
    api_key = os.getenv("LITELLM_API_KEY") or credentials.get("api_key")
    api_base = os.getenv("LITELLM_API_BASE") or metadata.get("apiBase")

    if not api_key and not api_base:
        logger.error("No LiteLLM credentials or API base available for %s", resolved)
        return None

    try:
        from langchain_litellm import ChatLiteLLM
    except Exception as e:
        logger.error("Failed to import langchain_litellm: %s", e)
        return None

    base_kwargs: dict = {"model": resolved, "max_retries": 2}
    if api_key:
        base_kwargs["api_key"] = api_key
    if api_base:
        base_kwargs["api_base"] = api_base

    max_tokens = _clamp_max_output_tokens(max_output_tokens)

    # Try in descending strictness so a permissive upstream gets full controls,
    # while a restrictive one still gets a working LLM with provider defaults.
    attempts: list[tuple[str, dict]] = [
        (
            "with temperature + max_tokens",
            {
                **base_kwargs,
                "temperature": _clamp_temperature(temperature),
                **({"max_tokens": max_tokens} if max_tokens is not None else {}),
            },
        ),
        (
            "without max_tokens",
            {**base_kwargs, "temperature": _clamp_temperature(temperature)},
        ),
        ("without sampling controls", base_kwargs),
    ]

    last_error: Exception | None = None
    for label, kwargs in attempts:
        try:
            llm = ChatLiteLLM(**kwargs)
            logger.info("Initialised LiteLLM model: %s (%s)", resolved, label)
            return llm
        except Exception as e:
            last_error = e
            logger.warning(
                "LiteLLM init failed for %s %s: %s — retrying with fewer params",
                resolved,
                label,
                e,
            )

    logger.error("Failed to initialise LiteLLM (%s): %s", resolved, last_error)
    return None


def _fallback_to_groq_default(
    original_model_name: str | None,
    provider_name: str,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
):
    logger.warning(
        "%s init failed for '%s', falling back to Groq default '%s'",
        provider_name,
        original_model_name,
        _GROQ_DEFAULT,
    )
    return get_groq_llm(
        name=_GROQ_DEFAULT,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def get_llm(
    model_name: str | None = None,
    user_id: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
):
    """
    Unified LLM dispatcher.

    Routes:
      - openai/* direct GPT ids  → OpenAI
      - anthropic/* direct ids   → Anthropic
      - litellm/* custom ids     → LiteLLM
      - gemini/* or gemini-*  → Gemini
      - nvidia-hosted model ids → NVIDIA NIM
      - everything else        → Groq

    Falls back to the Groq default model if Gemini/NVIDIA init fails.
    """
    # No model chosen → NVIDIA NIM default (widest catalog, larger context
    # window than free-tier Groq). Groq stays as the cross-provider fallback.
    if model_name in _NVIDIA_SENTINEL:
        llm = get_nvidia_llm(
            None,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        if llm is not None:
            return llm
        return _fallback_to_groq_default(
            model_name,
            "NVIDIA",
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    if _is_litellm(model_name):
        llm = get_litellm_llm(
            model_name,
            user_id=user_id,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        if llm is not None:
            return llm
        return None

    if _is_openai_direct(model_name):
        llm = get_openai_llm(
            model_name,
            user_id=user_id,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        if llm is not None:
            return llm
        return None

    if _is_anthropic(model_name):
        llm = get_anthropic_llm(
            model_name,
            user_id=user_id,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        if llm is not None:
            return llm
        return None

    if _is_gemini(model_name):
        llm = get_gemini_llm(
            model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        if llm is not None:
            return llm
        return _fallback_to_groq_default(
            model_name,
            "Gemini",
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    if _is_nvidia(model_name):
        llm = get_nvidia_llm(
            model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        if llm is not None:
            return llm
        return _fallback_to_groq_default(
            model_name,
            "NVIDIA",
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    return get_groq_llm(
        name=model_name,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
