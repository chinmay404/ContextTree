"""
TypeSafe Jev: typed yes/no answers with calibrated probabilities, for small context decisions.

Optional by design. With no TYPESAFE_API_KEY nothing is sent anywhere, and any error or a slow reply
returns None so the turn carries on with the search order. Board text goes to TypeSafe when this is
on, on the server's key, so turning it on is a privacy-policy decision as well as a config one.
"""
from __future__ import annotations

import json
from typing import Any

import httpx

from app.core.config import settings
from app.core.logger import logger

MAX_STATE_CHARS = 80_000  # Jev reads 32k tokens; stay well inside it, questions travel too


def enabled() -> bool:
    return bool(settings.TYPESAFE_API_KEY)


def _transport() -> httpx.BaseTransport | None:
    return None  # httpx's default; tests replace it


def ask_nouls(state: dict, questions: dict[str, str]) -> dict[str, float] | None:
    """{question id: chance of yes}, or None when Jev is off, failed or answered oddly."""
    if not enabled() or not questions:
        return None
    payload = {
        "model": settings.JEV_MODEL,
        "state": json.dumps(state, ensure_ascii=False),
        "questions": {qid: {"type": "noul", "instructions": text} for qid, text in questions.items()},
    }
    failure: str | None = None
    try:
        with httpx.Client(transport=_transport(), timeout=settings.JEV_TIMEOUT_SECONDS) as client:
            reply = client.post(
                settings.TYPESAFE_BASE_URL.rstrip("/") + "/v1/systemone",
                headers={"Authorization": f"Bearer {settings.TYPESAFE_API_KEY}"},
                json=payload,
            )
        if reply.status_code != 200:
            failure = f"HTTP {reply.status_code}"
        else:
            return _nouls(reply.json(), questions)
    except Exception as exc:  # noqa: BLE001 - Jev never fails a turn
        failure = type(exc).__name__  # the type only: exception text can echo request headers
    logger.warning("Jev unavailable: %s", failure)
    return None


def _nouls(body: Any, questions: dict[str, str]) -> dict[str, float] | None:
    answers = body.get("answers") if isinstance(body, dict) else None
    if not isinstance(answers, dict):
        logger.warning("Jev unavailable: reply has no answers")
        return None
    out: dict[str, float] = {}
    for qid in questions:
        answer = answers.get(qid)
        value = answer.get("noul") if isinstance(answer, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            logger.warning("Jev unavailable: answer %s is not a probability", qid)
            return None
        out[qid] = float(value)
    return out
