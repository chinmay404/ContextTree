"""
API router that includes all API endpoints.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import chat, health, fork, files, board, cards, edges, reads, sources


api_router = APIRouter()


api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat endpoints"])
api_router.include_router(fork.router, prefix="/fork", tags=["fork utilities"])
api_router.include_router(files.router, prefix="/files", tags=["files ingestion"])
api_router.include_router(board.router, prefix="/boards", tags=["boards"])
api_router.include_router(cards.router, prefix="/boards", tags=["cards"])
api_router.include_router(edges.router, prefix="/boards", tags=["edges"])
api_router.include_router(reads.router, prefix="/boards", tags=["reads"])
api_router.include_router(sources.router, tags=["sources"])
