"""
Health check endpoints.
"""
import os

from fastapi import APIRouter, status
from app.api.limiter import limiter
from app.core.deploy_info import get_deploy_info

from fastapi import Request


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
