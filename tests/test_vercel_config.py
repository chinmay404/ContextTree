"""The engine deploys as one Vercel Function (contract amendment 2026-10-21; Hobby: 300 s)."""
import json
import re
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _vercel() -> dict:
    return json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))


def _brace_globs(pattern: str) -> list[str]:
    m = re.fullmatch(r"\{(.*)\}", pattern)
    return m.group(1).split(",") if m else [pattern]


def test_the_entrypoint_is_main_py_with_a_fastapi_app():
    from fastapi import FastAPI

    import main  # Vercel's zero-config FastAPI entrypoint: a top-level `app` in main.py
    from app.api.server import app

    assert isinstance(main.app, FastAPI) and main.app is app


def test_the_function_runs_for_the_hobby_maximum():
    fn = _vercel()["functions"]["main.py"]
    assert fn["maxDuration"] == 300


def test_tests_and_dev_files_stay_out_of_the_bundle():
    globs = _brace_globs(_vercel()["functions"]["main.py"]["excludeFiles"])
    for path in ("tests/test_v3_boards.py", "tests/v3_support.py", "env/lib/x.py", ".venv/x.py",
                 "scripts/migrate.py", "db/migrations/009_v3_board.sql", ".superpowers/a.md",
                 "app.log"):
        assert any(fnmatch(path, g) for g in globs), path
    for path in ("main.py", "app/api/server.py", "app/agent/prompts/prompts.yaml"):
        assert not any(fnmatch(path, g) for g in globs), path


def test_cli_uploads_never_carry_secrets_or_venvs():
    lines = {ln.strip() for ln in (ROOT / ".vercelignore").read_text(encoding="utf-8").splitlines()}
    assert {".env", ".env.*", "env/", ".venv/", "tests/"} <= lines


def test_python_version_is_pinned():
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12"


def test_test_only_packages_are_not_runtime_dependencies():
    def names(path):
        out = set()
        for ln in (ROOT / path).read_text(encoding="utf-8").splitlines():
            ln = ln.split("#")[0].strip()
            if ln and not ln.startswith("-r"):
                out.add(re.split(r"[<>=\[ ]", ln)[0].lower())
        return out

    runtime = names("requirements.txt")
    assert not {"pytest", "pytest-cov"} & runtime
    assert {"fastapi", "uvicorn", "httpx", "psycopg2-binary"} <= runtime  # httpx is used at runtime
    dev = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert "-r requirements.txt" in dev and {"pytest", "pytest-cov"} <= names("requirements-dev.txt")


def test_startup_runs_no_migrations():
    server = (ROOT / "app" / "api" / "server.py").read_text(encoding="utf-8")
    assert "migrate" not in server
    for path in ("main.py",):
        assert "migrate" not in (ROOT / path).read_text(encoding="utf-8")


def test_logger_imports_on_a_read_only_filesystem(monkeypatch):
    import importlib
    import logging

    import app.core.logger as logger_mod

    def read_only(*_a, **_k):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(logging, "FileHandler", read_only)
    saved = logger_mod.logger.handlers[:]
    logger_mod.logger.handlers.clear()
    try:
        importlib.reload(logger_mod)  # Vercel: /var/task is read-only
        assert [type(h) for h in logger_mod.logger.handlers] == [logging.StreamHandler]
    finally:
        logger_mod.logger.handlers[:] = saved


def test_health_reports_the_vercel_commit(monkeypatch):
    from app.core.deploy_info import get_deploy_info

    monkeypatch.delenv("APP_BUILD_SHA", raising=False)
    monkeypatch.setenv("VERCEL_GIT_COMMIT_SHA", "abc1234")
    assert get_deploy_info()["sha"] == "abc1234"
    monkeypatch.setenv("APP_BUILD_SHA", "fromci")
    assert get_deploy_info()["sha"] == "fromci"
