from __future__ import annotations

import base64
import hashlib
import os
from typing import Optional

import psycopg2
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.core.logger import logger


def _get_master_secret() -> str:
    secret = os.getenv("BYOK_ENCRYPTION_SECRET") or os.getenv("NEXTAUTH_SECRET")
    if not secret:
        raise RuntimeError("BYOK_ENCRYPTION_SECRET (or NEXTAUTH_SECRET fallback) is not configured")
    return secret


def _derive_key() -> bytes:
    return hashlib.sha256(_get_master_secret().encode("utf-8")).digest()


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("utf-8"))


def decrypt_api_key(payload: str) -> str:
    parts = str(payload or "").split(".")
    if len(parts) != 3:
        raise ValueError("Stored API key payload is invalid")

    iv = _b64url_decode(parts[0])
    tag = _b64url_decode(parts[1])
    ciphertext = _b64url_decode(parts[2])

    aesgcm = AESGCM(_derive_key())
    decrypted = aesgcm.decrypt(iv, ciphertext + tag, None)
    return decrypted.decode("utf-8")


def _resolve_user_email(user_id: Optional[str]) -> Optional[str]:
    if not user_id:
        return None
    if "@" in user_id:
        return user_id

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return user_id

    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT email FROM users WHERE id = %s", (user_id,))
                row = cur.fetchone()
                return row[0] if row else user_id
    except Exception as exc:
        logger.warning("Failed to resolve user email for BYOK lookup: %s", exc)
        return user_id


def get_user_provider_api_key(user_id: Optional[str], provider: str) -> Optional[str]:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return None

    user_email = _resolve_user_email(user_id)
    if not user_email:
        return None

    try:
        with psycopg2.connect(db_url) as conn:
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
                if not row or not row[0]:
                    return None

                return decrypt_api_key(row[0])
    except Exception as exc:
        logger.warning("Failed to load BYOK credentials for %s: %s", provider, exc)
        return None
