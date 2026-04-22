"""
FastAPI application creation and configuration.
"""
import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.v1.api import api_router


from .limiter import limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler


from app.core.logger import logger
from app.core.deploy_info import get_deploy_info

# TODO : add_event_handler for Checking Dbs Connections


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.DESCRIPTION,
        version=settings.VERSION,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url=f"{settings.API_V1_STR}/docs",
        redoc_url=f"{settings.API_V1_STR}/redoc",
    )

    application.state.limiter = limiter
    application.add_exception_handler(
        RateLimitExceeded, _rate_limit_exceeded_handler)
    application.add_middleware(SlowAPIMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin)for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api_prefix = settings.API_V1_STR or ""
    application.include_router(api_router, prefix=api_prefix)
    # Also serve unprefixed routes for clients that call "/chat/..." or "/files/..."
    if api_prefix:
        application.include_router(api_router, prefix="")

    @application.on_event("startup")
    async def log_startup_context():
        deploy = get_deploy_info()
        logger.info(
            "Startup context: cwd=%s python=%s api_prefix=%s version=%s deploy=%s",
            os.getcwd(),
            sys.executable,
            api_prefix,
            settings.VERSION,
            deploy,
        )
        logger.info(
            "Provider key availability: groq=%s google=%s nvidia=%s",
            bool(settings.GROQ_API_KEY),
            bool(settings.GOOGLE_API_KEY),
            bool(settings.NVIDIA_API_KEY),
        )
        logger.info(
            "LangSmith tracing: enabled=%s project=%s endpoint=%s api_key_configured=%s",
            settings.LANGSMITH_TRACING,
            settings.LANGSMITH_PROJECT or "default",
            settings.LANGSMITH_ENDPOINT or "https://api.smith.langchain.com",
            bool(settings.LANGSMITH_API_KEY),
        )

    @application.get("/")
    async def root():
        logger.info("API health Check Call.")
        """Health check endpoint."""
        return {
            "message": f"Welcome to {settings.PROJECT_NAME}",
            "version": settings.VERSION,
            "status": "healthy",
        }

    return application


app = create_application()
