"""
Strict key-store lookup for V3 turns.

V2's `byok.get_user_provider_api_key` turns every failure into "no key". On a V3
turn that would tell a user to re-add a key they already saved, so this variant
keeps the two apart: no row → None; store failure → KeyStoreUnavailableError.
V2 behaviour is unchanged.
"""

from __future__ import annotations

import os
from typing import Optional

import psycopg2

from app.agent.helpers.byok import decrypt_api_key
from app.core.logger import logger


class KeyStoreUnavailableError(Exception):
    """The key store could not answer (no DB, DB error, or a row that won't decrypt)."""


def fetch_user_provider_api_key(user_email: Optional[str], provider: str) -> Optional[str]:
    """The user's stored key for a provider, or None when there is no row.

    Raises KeyStoreUnavailableError when the store cannot answer. The error
    carries the failure type only, never the cause chain.
    """
    if not user_email or "@" not in user_email:
        return None  # V3 callers pass the resolved email (the tenancy key)
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise KeyStoreUnavailableError("DATABASE_URL is not configured")
    failure: Optional[str] = None
    try:
        conn = psycopg2.connect(db_url)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT encrypted_key
                    FROM user_api_keys
                    WHERE user_email = %s AND provider = %s
                    LIMIT 1
                    """,
                    (user_email, provider),
                )
                row = cur.fetchone()
        finally:
            conn.close()
        if not row or not row[0]:
            return None
        return decrypt_api_key(row[0]) or None
    except Exception as exc:
        failure = type(exc).__name__
    logger.warning("V3 key store unavailable for %s (%s)", provider, failure)
    raise KeyStoreUnavailableError(f"key store unavailable ({failure})")
