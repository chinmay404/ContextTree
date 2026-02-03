"""
API router that includes all API endpoints.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import chat, health, fork, files


api_router = APIRouter()


api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat endpoints"])
api_router.include_router(fork.router, prefix="/fork", tags=["thread forking"])
api_router.include_router(files.router, prefix="/files", tags=["files ingestion"])
