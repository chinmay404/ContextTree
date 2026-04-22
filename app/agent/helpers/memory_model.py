from __future__ import annotations

from app.core.config import settings


def get_summary_model_name(active_model: str | None = None) -> str | None:
    configured = (settings.SUMMARY_MODEL_NAME or "").strip()
    if configured:
        return configured
    return active_model
