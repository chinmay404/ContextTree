import logging
import os
from urllib.parse import urlparse
from typing import Any, List, Optional

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class Settings(BaseSettings):
    # ── API ────────────────────────────────────────────────────────────────────
    version_str: str = "v1"
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "ContextTree API"
    VERSION: str = "0.2.0"
    DESCRIPTION: str = "ContextTree Backend — scoped, branching LLM conversations"

    # ── Server ─────────────────────────────────────────────────────────────────
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000
    DEBUG_MODE: bool = False
    LOG_LEVEL: str = "INFO"

    # ── CORS ───────────────────────────────────────────────────────────────────
    BACKEND_CORS_ORIGINS: List[str] = ["*"]

    # ── Database ───────────────────────────────────────────────────────────────
    DATABASE_URL: Optional[str] = None

    # ── LLM: Groq ──────────────────────────────────────────────────────────────
    GROQ_API_KEY: Optional[str] = None
    DEFAULT_GROQ_MODEL: str = "openai/gpt-oss-120b"

    # ── LLM: OpenAI direct / BYOK ─────────────────────────────────────────────
    OPENAI_API_KEY: Optional[str] = None
    DEFAULT_OPENAI_MODEL: str = "gpt-5"

    # ── LLM: Google Gemini ─────────────────────────────────────────────────────
    GOOGLE_API_KEY: Optional[str] = None
    DEFAULT_GEMINI_MODEL: str = "gemini/gemini-3-flash-preview"

    # ── LLM: Anthropic direct / BYOK ──────────────────────────────────────────
    ANTHROPIC_API_KEY: Optional[str] = None
    DEFAULT_ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"

    # ── LLM: NVIDIA NIM ────────────────────────────────────────────────────────
    NVIDIA_API_KEY: Optional[str] = None
    DEFAULT_NVIDIA_MODEL: str = "z-ai/glm-5.2"
    SUMMARY_MODEL_NAME: str = "openai/gpt-oss-120b"

    # ── Observability: LangSmith ──────────────────────────────────────────────
    LANGSMITH_TRACING: bool = False
    LANGSMITH_ENDPOINT: Optional[str] = None
    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_PROJECT: Optional[str] = None

    # ── Secure BYOK storage ────────────────────────────────────────────────────
    BYOK_ENCRYPTION_SECRET: Optional[str] = None

    # ── Service-to-service auth ────────────────────────────────────────────────
    # Shared secret with the Next.js proxy; it mints a short-lived HS256 JWT
    # per request (sub = user id). No secret configured → all requests refused.
    BACKEND_JWT_SECRET: Optional[str] = None

    # ── Quotas ─────────────────────────────────────────────────────────────────
    # Durable daily message cap per verified user (Postgres-backed, survives
    # restarts/workers). Generous default; plan-based tiers come with billing.
    DAILY_MESSAGE_LIMIT: int = 500

    # ── Conversation memory ────────────────────────────────────────────────────
    # Trigger summarisation after this many assistant turns per node
    MAX_MESSAGES_BEFORE_SUMMARY: int = 10
    # Keep this many recent messages verbatim after summarisation
    KEEP_LAST_MESSAGES: int = 6
    # Carry forward this many messages as verbatim buffer when forking
    FORK_BUFFER_MESSAGES: int = 2

    # ── Web search (free, DuckDuckGo via ddgs — no API key) ────────────────────
    # Kill switch; per-turn opt-in still comes from ChatMessage.web_search
    WEB_SEARCH_ENABLED: bool = True
    # Webpage parser: fetch user-pasted URLs into external context
    # (SSRF-guarded + char-capped; see app/agent/helpers/web_fetch.py).
    WEB_FETCH_ENABLED: bool = True

    # ── Context assembly ───────────────────────────────────────────────────────
    # Max tokens reserved for rolling summary
    MAX_SUMMARY_TOKENS: int = 500
    # Max tokens reserved for the most recent messages
    MAX_PREVIOUS_MESSAGES_TOKENS: int = 2000
    # Token budget kept free for the LLM's own response
    RESERVE_TOKENS: int = 1000
    # Minimum semantic similarity score for message-to-message retrieval.
    SIMILARITY_MIN_SCORE: float = 0.2

    # ── Local end-to-end runs ──────────────────────────────────────────────────
    # Deterministic fake LLM + embeddings (no network). Local only: refused
    # when running on Railway.
    ENGINE_FAKE_LLM: bool = False

    @model_validator(mode="after")
    def _fake_llm_local_only(self):
        if not self.ENGINE_FAKE_LLM:
            return self
        # Broad refusal: presence (even empty) of any RAILWAY_* variable.
        if any(k.upper().startswith("RAILWAY_") for k in os.environ):
            raise RuntimeError("ENGINE_FAKE_LLM is for local runs only")
        # Positive local opt-in: a configured DB must be on this machine.
        if self.DATABASE_URL:
            host = (urlparse(self.DATABASE_URL).hostname or "").lower()
            if host not in _LOCAL_HOSTS:
                raise RuntimeError(
                    "ENGINE_FAKE_LLM is for local runs only (DATABASE_URL host is not local)"
                )
        logging.getLogger(__name__).warning(
            "*** ENGINE_FAKE_LLM is ON: fake LLM answers and fake embeddings. "
            "Local runs only; never use with real data. ***"
        )
        return self

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        if isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    model_config = {
        "case_sensitive": True,
        "env_file": ".env",
        "extra": "ignore",
    }


settings = Settings()
