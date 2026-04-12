"""
LLM provider routing.

Model naming convention (sent from frontend):
  - gemini/<model>   → Google Gemini via langchain_google_genai
  - nvidia-hosted ids → NVIDIA NIM via langchain_openai (OpenAI-compatible)
  - everything else  → Groq via langchain_groq

Supported Gemini model IDs (the part after "gemini/"):
  gemini-3-flash-preview, gemini-2.5-pro, gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

from app.core.logger import logger

load_dotenv()

# ── Groq defaults ──────────────────────────────────────────────────────────────
_GROQ_DEFAULT = "meta-llama/llama-4-scout-17b-16e-instruct"
_GROQ_SENTINEL = {"None", "null", "default-model-name", "", None}
_NVIDIA_DEFAULT = "moonshotai/kimi-k2-instruct-0905"
_NVIDIA_SENTINEL = {"None", "null", "default-model-name", "", None}

# ── Gemini model aliases ───────────────────────────────────────────────────────
# Maps the short model string (after "gemini/") to its full Gemini model ID.
_GEMINI_ALIASES: dict[str, str] = {
    "gemini-3-flash-preview": "gemini-3-flash-preview",
    "gemini-2.5-pro":    "gemini-2.5-pro-preview-06-05",
    "gemini-2.0-flash":  "gemini-2.0-flash",
    "gemini-1.5-pro":    "gemini-1.5-pro",
    "gemini-1.5-flash":  "gemini-1.5-flash",
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


def get_gemini_llm(model_name: str | None = None):
    """
    Returns a ChatGoogleGenerativeAI instance.
    model_name may be:
      - "gemini/gemini-2.0-flash"  (frontend format)
      - "gemini-2.0-flash"         (bare)
      - None → falls back to gemini-2.0-flash
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not set — cannot initialise Gemini LLM")
        return None

    # Normalise: strip the "gemini/" prefix if present
    bare = model_name or "gemini-2.0-flash"
    if bare.startswith("gemini/"):
        bare = bare[len("gemini/"):]

    # Resolve alias to full model ID accepted by the Gemini API
    resolved = _GEMINI_ALIASES.get(bare, bare)

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
    if model_name in _NVIDIA_SENTINEL:
        model_name = _NVIDIA_DEFAULT

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


def get_llm(model_name: str | None = None):
    """
    Unified LLM dispatcher.

    Routes:
      - gemini/* or gemini-*  → Gemini
      - nvidia-hosted model ids → NVIDIA NIM
      - everything else        → Groq

    Falls back to Groq default if Gemini init fails.
    """
    if _is_gemini(model_name):
        llm = get_gemini_llm(model_name)
        if llm is not None:
            return llm
        logger.warning(f"Gemini init failed for '{model_name}', falling back to Groq default")

    if _is_nvidia(model_name):
        llm = get_nvidia_llm(model_name)
        if llm is not None:
            return llm
        logger.warning(f"NVIDIA init failed for '{model_name}', falling back to Groq default")

    return get_groq_llm(name=model_name)
