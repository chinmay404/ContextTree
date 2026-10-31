"""
Health check endpoints.
"""
import os

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from app.agent.helpers import jev
from app.agent.store.board_store import store
from app.agent.turn_issues import alert
from app.agent.utils.embeddings import get_embedding
from app.api.limiter import limiter
from app.core.config import settings
from app.core.deploy_info import get_deploy_info

from fastapi import Request

# The newest file in db/migrations (a test keeps this true): a deploy ahead of its migration shows here.
EXPECTED_MIGRATION = "012_board_marks.sql"


router = APIRouter()


@router.get("/", status_code=status.HTTP_200_OK)
@limiter.limit("5/minute")
async def health_check(request: Request):
    """
    Health check endpoint.

    Returns:
        dict: Health status of the application.
    """
    return {
        "status": "healthy",
        "message": "The service is up and running.",
        "pid": os.getpid(),
        "deploy": get_deploy_info(),
    }


def _check_database() -> str:
    with store.tx() as cur:
        cur.execute("SELECT 1 AS ok FROM schema_migrations WHERE filename = %s", (EXPECTED_MIGRATION,))
        return "ok" if cur.fetchone() else "behind"


@router.get("/deep")
@limiter.limit("5/minute")
def deep_health_check(request: Request):
    """Does the engine really work? 503 when not, so an uptime monitor can raise the alarm.

    Says which part failed and never why in detail: this route is public.
    """
    checks = {"database": "ok", "migrations": "ok", "embeddings": "ok"}
    try:
        checks["migrations"] = _check_database()
    except Exception as e:  # noqa: BLE001
        alert("health_database", "health", e)
        checks["database"] = checks["migrations"] = "failing"
    try:
        checks["embeddings"] = "ok" if any(get_embedding("health check")) else "failing"
    except Exception as e:  # noqa: BLE001
        alert("health_embeddings", "health", e)
        checks["embeddings"] = "failing"
    healthy = all(v == "ok" for v in checks.values())
    if not healthy:
        alert("health_failing", ",".join(k for k, v in checks.items() if v != "ok"))
    return JSONResponse(
        {"status": "ok" if healthy else "failing", "checks": checks,
         "embeddingProvider": settings.EMBEDDING_PROVIDER, "jevEnabled": jev.enabled(),
         "deploy": get_deploy_info()},
        status_code=200 if healthy else 503,
    )
