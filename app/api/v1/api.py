"""
API router that includes all API endpoints.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import chat, health


api_router = APIRouter()


api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat endpoints"])
