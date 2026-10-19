"""Error-envelope handlers (no database needed)."""
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from app.api.errors import ApiError, install_error_handlers


class Body(BaseModel):
    n: int


def _app():
    app = FastAPI()
    limiter = Limiter(key_func=get_remote_address, default_limits=[])
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)
    install_error_handlers(app)

    @app.post("/boards/x")
    def a(b: Body):
        return {}

    @app.post("/files/ingest")
    def c(b: Body):
        return {}

    @app.get("/api/v1/boards/e")
    def d():
        raise ApiError(409, "duplicate", "dup")

    @app.get("/h")
    def e():
        raise HTTPException(401, "nope", headers={"WWW-Authenticate": "Bearer"})

    @app.get("/limited")
    @limiter.limit("1/minute")
    def f(request: Request):
        return {}

    return TestClient(app)


def test_api_error_envelope():
    r = _app().get("/api/v1/boards/e")
    assert r.status_code == 409 and r.json() == {"error": "duplicate", "detail": "dup"}


def test_http_exception_envelope_keeps_headers():
    r = _app().get("/h")
    assert r.status_code == 401 and r.json() == {"error": "unauthorized", "detail": "nope"}
    assert r.headers["www-authenticate"] == "Bearer"


def test_validation_400_only_on_board_paths():
    c = _app()
    r = c.post("/boards/x", json={"n": "zz"})
    assert r.status_code == 400 and r.json()["error"] == "invalid"
    assert c.post("/files/ingest", json={"n": "zz"}).status_code == 422


def test_unknown_route_is_not_found_envelope():
    r = _app().get("/nothing")
    assert r.status_code == 404 and r.json()["error"] == "not_found"


def test_rate_limit_429_envelope():
    c = _app()
    assert c.get("/limited").status_code == 200
    r = c.get("/limited")
    assert r.status_code == 429 and r.json()["error"] == "rate_limited" and r.json()["detail"]
