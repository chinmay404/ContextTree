"""
Service-to-service trust boundary (V2 architecture §2).

The Next.js proxy is the only legitimate caller of this API. For every
request it mints a short-lived HS256 JWT (shared secret BACKEND_JWT_SECRET)
whose `sub` claim is the authenticated user's id. FastAPI verifies the
token here and exposes the identity as a dependency — request bodies are
never trusted for identity.

Fail-closed: if BACKEND_JWT_SECRET is not configured, every request is
rejected rather than falling back to trusting the body.
"""

from fastapi import HTTPException, Request, status

import jwt

from app.core.config import settings

ALGORITHM = "HS256"
# Tolerate small clock drift between the Next.js host and this service.
LEEWAY_SECONDS = 30


def get_current_user_id(request: Request) -> str:
    """FastAPI dependency: verify the bearer token, return the user id."""
    secret = settings.BACKEND_JWT_SECRET
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth is not configured on the server (BACKEND_JWT_SECRET missing)",
        )

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )

    token = auth_header.removeprefix("Bearer ").strip()
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=[ALGORITHM],
            leeway=LEEWAY_SECONDS,
            options={"require": ["sub", "exp"]},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

    user_id = str(claims["sub"]).strip()
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has an empty subject",
        )

    # Rate limiting and logging read the verified identity from here.
    request.state.user_id = user_id
    return user_id
