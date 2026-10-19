"""ASGI guard: reject oversized source uploads before the body is spooled to disk/memory."""
from __future__ import annotations

import json

MAX_UPLOAD_BYTES = 4 * 1024 * 1024
MULTIPART_OVERHEAD = 64 * 1024
LIMIT = MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD


async def _reject(send):
    body = json.dumps({"error": "too_large", "detail": "files are limited to 4 MB"}).encode()
    await send({"type": "http.response.start", "status": 413,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


class _TooLarge(Exception):
    pass


class UploadSizeLimitMiddleware:
    def __init__(self, app):
        self.app = app

    @staticmethod
    def _applies(scope) -> bool:
        return (scope["type"] == "http" and scope["method"] == "POST"
                and scope["path"].endswith("/sources/file"))

    async def __call__(self, scope, receive, send):
        if not self._applies(scope):
            return await self.app(scope, receive, send)

        for k, v in scope["headers"]:
            if k == b"content-length":
                try:
                    if int(v) > LIMIT:
                        return await _reject(send)
                except ValueError:
                    pass

        seen = 0
        started = False
        overflow = False

        async def counting_receive():
            nonlocal seen, overflow
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > LIMIT:
                    overflow = True
                    raise _TooLarge()
            return msg

        async def tracking_send(msg):
            nonlocal started
            if overflow:
                # Whatever the app answers after an over-limit read (FastAPI turns the
                # aborted body read into a 400) is replaced by one 413.
                if not started:
                    started = True
                    await _reject(send)
                return
            if msg["type"] == "http.response.start":
                started = True
            await send(msg)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except Exception:
            if overflow and not started:
                return await _reject(send)
            raise
