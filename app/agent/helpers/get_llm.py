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

from app.core.logger import logger

load_dotenv()

# ── Groq defaults ──────────────────────────────────────────────────────────────
_GROQ_DEFAULT = os.getenv("DEFAULT_GROQ_MODEL") or "openai/gpt-oss-120b"
_GROQ_SENTINEL = {"None", "null", "default-model-name", "", None}
_NVIDIA_DEFAULT = os.getenv("DEFAULT_NVIDIA_MODEL") or "moonshotai/kimi-k2-instruct-0905"
_NVIDIA_SENTINEL = {"None", "null", "default-model-name", "", None}

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

_NVIDIA_ALIASES: dict[str, str] = {
    "moonshotai/kimi-k2-instruct": "moonshotai/kimi-k2-instruct",
    "moonshotai/kimi-k2-instruct-0905": "moonshotai/kimi-k2-instruct-0905",
    "z-ai/glm-4.7": "z-ai/glm4.7",
    "z-ai/glm4.7": "z-ai/glm4.7",
    "z-ai/glm4_7": "z-ai/glm4.7",
    "deepseek-ai/deepseek-v3.1": "deepseek-ai/deepseek-v3.1",
    "deepseek-ai/deepseek-v3_1": "deepseek-ai/deepseek-v3.1",
    "deepseek-ai/deepseek-v3.2": "deepseek-ai/deepseek-v3.2",
    "deepseek-ai/deepseek-v3_2": "deepseek-ai/deepseek-v3.2",
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


def _is_nvidia(model_name: str | None) -> bool:
    if not model_name:
        return False
    return model_name.startswith(_NVIDIA_PREFIXES)


def _normalize_gemini_model_name(model_name: str | None) -> str:
    bare = model_name or os.getenv("DEFAULT_GEMINI_MODEL") or "gemini-3-flash-preview"
    if bare.startswith("gemini/"):
        bare = bare[len("gemini/"):]
    if bare.startswith("models/"):
        return bare

    resolved = _GEMINI_ALIASES.get(bare, bare)
    return resolved if resolved.startswith("models/") else f"models/{resolved}"


def _normalize_nvidia_model_name(model_name: str | None) -> str:
    if model_name in _NVIDIA_SENTINEL:
        return _NVIDIA_DEFAULT
    return _NVIDIA_ALIASES.get(model_name or _NVIDIA_DEFAULT, model_name or _NVIDIA_DEFAULT)


def get_gemini_llm(model_name: str | None = None):
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
        llm = ChatGoogleGenerativeAI(
            model=resolved,
            google_api_key=api_key,
            temperature=0.8,
            max_retries=2,
        )
        logger.info(f"Initialised Gemini LLM: {resolved}")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialise Gemini LLM ({resolved}): {e}")
        return None


def get_groq_llm(name: str | None = None):
    """Returns a ChatGroq instance, falling back to the default Groq model."""
    if name in _GROQ_SENTINEL:
        name = _GROQ_DEFAULT

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY not set — cannot initialise Groq LLM")
        return None

    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            temperature=0.8,
            api_key=str(api_key),
            model_name=name,
        )
        return llm
    except Exception as e:
        logger.error(f"Failed to initialise Groq LLM ({name}): {e}")
        return None


def get_nvidia_llm(model_name: str | None = None):
    """Returns a ChatOpenAI instance pointed at NVIDIA's OpenAI-compatible API."""
    model_name = _normalize_nvidia_model_name(model_name)

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        logger.error("NVIDIA_API_KEY not set — cannot initialise NVIDIA LLM")
        return None

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://integrate.api.nvidia.com/v1",
            temperature=0.8,
            max_retries=2,
        )
        logger.info(f"Initialised NVIDIA LLM: {model_name}")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialise NVIDIA LLM ({model_name}): {e}")
        return None


def _fallback_to_groq_default(original_model_name: str | None, provider_name: str):
    logger.warning(
        "%s init failed for '%s', falling back to Groq default '%s'",
        provider_name,
        original_model_name,
        _GROQ_DEFAULT,
    )
    return get_groq_llm(name=_GROQ_DEFAULT)


def get_llm(model_name: str | None = None):
    """
    Unified LLM dispatcher.

    Routes:
      - gemini/* or gemini-*  → Gemini
      - nvidia-hosted model ids → NVIDIA NIM
      - everything else        → Groq

    Falls back to the Groq default model if Gemini/NVIDIA init fails.
    """
    if _is_gemini(model_name):
        llm = get_gemini_llm(model_name)
        if llm is not None:
            return llm
        return _fallback_to_groq_default(model_name, "Gemini")

    if _is_nvidia(model_name):
        llm = get_nvidia_llm(model_name)
        if llm is not None:
            return llm
        return _fallback_to_groq_default(model_name, "NVIDIA")

    return get_groq_llm(name=model_name)
