"""
Trust-boundary tests (V2 architecture §2, §7 row 1).

The API must take identity ONLY from a verified service JWT:
- no token            → 401
- garbage token       → 401
- expired token       → 401
- wrong-secret token  → 401
- valid token         → passes auth (may fail later for other reasons,
                        but never 401/403)
- body user_id        → ignored by schema (identity cannot be spoofed)
"""

import time

import jwt
import pytest
from fastapi.testclient import TestClient

from app.api.server import app
from app.core.config import settings
from app.schemas.item import ChatMessage

client = TestClient(app)

CHAT = f"{settings.API_V1_STR}/chat/"
SUMMARY = f"{settings.API_V1_STR}/fork/summary/some-thread"

BODY = {
    "message": "hello",
    "message_id": "m1",
    "nodeId": "node_test_1",
    "model": "openai/gpt-oss-120b",
}


def _token(sub="user-a", secret=None, exp_offset=60):
    secret = secret or settings.BACKEND_JWT_SECRET
    now = int(time.time())
    return jwt.encode({"sub": sub, "iat": now, "exp": now + exp_offset}, secret, algorithm="HS256")


@pytest.fixture(autouse=True)
def require_secret():
    if not settings.BACKEND_JWT_SECRET:
        pytest.skip("BACKEND_JWT_SECRET not configured in test env")


def test_chat_without_token_is_401():
    r = client.post(CHAT, json=BODY)
    assert r.status_code == 401


def test_chat_with_garbage_token_is_401():
    r = client.post(CHAT, json=BODY, headers={"Authorization": "Bearer not.a.jwt"})
    assert r.status_code == 401


def test_chat_with_expired_token_is_401():
    r = client.post(
        CHAT, json=BODY,
        headers={"Authorization": f"Bearer {_token(exp_offset=-120)}"},
    )
    assert r.status_code == 401


def test_chat_with_wrong_secret_is_401():
    r = client.post(
        CHAT, json=BODY,
        headers={"Authorization": f"Bearer {_token(secret='attacker-secret-attacker-secret')}"},
    )
    assert r.status_code == 401


def test_summary_without_token_is_401():
    r = client.get(SUMMARY)
    assert r.status_code == 401


def test_valid_token_passes_the_auth_layer():
    r = client.get(SUMMARY, headers={"Authorization": f"Bearer {_token()}"})
    # Auth must not be the failure; downstream may 200 or 500 (no DB in CI).
    assert r.status_code not in (401, 403)


def test_body_user_id_is_ignored_by_schema():
    parsed = ChatMessage(**{**BODY, "user_id": "victim@example.com"})
    assert not hasattr(parsed, "user_id")
