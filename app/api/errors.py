"""
Uniform error envelope for the V3 board API: {"error": "<code>", "detail": "<text>"}.
"""
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

CODE_BY_STATUS = {
    400: "invalid",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    409: "conflict",
    413: "too_large",
    415: "unsupported_type",
    429: "rate_limited",
    503: "unavailable",
}

# Board-API paths get the 400 envelope for body/param validation errors; every
# other route (V2 /files/ingest, chat) keeps FastAPI's default 422.
_BOARD_PREFIXES = ("/boards", "/api/v1/boards")


class ApiError(HTTPException):
    """HTTP error with an explicit machine code. Subclasses HTTPException so broad
    `except HTTPException` paths (and Starlette) treat it as a normal HTTP error."""

    def __init__(self, status: int, code: str, detail: Optional[str] = None):
        super().__init__(status_code=status, detail=detail)
        self.code = code
        self.api_detail = detail


def _envelope(status: int, code: str, detail, headers=None) -> JSONResponse:
    return JSONResponse({"error": code, "detail": detail}, status_code=status, headers=headers)


def _is_board_path(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in _BOARD_PREFIXES)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def _http_exc(request: Request, exc: StarletteHTTPException):
        if isinstance(exc, ApiError):
            return _envelope(exc.status_code, exc.code, exc.api_detail, exc.headers)
        code = CODE_BY_STATUS.get(exc.status_code, "error")
        return _envelope(exc.status_code, code, exc.detail, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        if _is_board_path(request.url.path):
            errors = exc.errors()
            msg = str(errors[0].get("msg", "invalid request")) if errors else "invalid request"
            return _envelope(400, "invalid", msg)
        from fastapi.exception_handlers import request_validation_exception_handler
        return await request_validation_exception_handler(request, exc)

    @app.exception_handler(RateLimitExceeded)
    async def _rate_limited(request: Request, exc: RateLimitExceeded):
        response = _envelope(429, "rate_limited", f"Rate limit exceeded: {exc.detail}")
        try:
            limit = getattr(request.state, "view_rate_limit", None)
            if limit is not None:
                response = request.app.state.limiter._inject_headers(response, limit)
        except Exception:
            pass
        return response
