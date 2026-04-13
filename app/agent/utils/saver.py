import os
from dotenv import load_dotenv
import atexit

load_dotenv()

_pg_pool = None
_async_pg_pool = None


def postgres_saver():
    """
    Return a synchronous PostgresSaver for LangGraph checkpointing.

    Uses a module-level ConnectionPool with autocommit=True and
    prepare_threshold=None to satisfy Supabase PgBouncer's transaction-level
    pooling mode (no prepared statements, no DDL inside transactions).

    saver.setup() is called once on first use; we swallow errors caused by
    concurrent DDL restrictions (CREATE INDEX CONCURRENTLY) because those
    tables/indexes were already created by a previous run.
    """
    try:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg_pool import ConnectionPool
        except ImportError as import_err:
            print(
                "Postgres checkpoint support is unavailable: "
                f"{import_err}. Install psycopg[binary] in the backend environment."
            )
            return None

        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL must be set")

        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": None,
        }

        global _pg_pool
        if _pg_pool is None:
            pool = ConnectionPool(
                conninfo=db_url,
                max_size=20,
                kwargs=connection_kwargs,
                open=False,  # Don't connect on creation; open below with error handling
            )
            try:
                pool.open(wait=True, timeout=5)
            except Exception as conn_err:
                print(f"PostgreSQL connection failed (is Supabase paused?): {conn_err}")
                return None
            _pg_pool = pool
            atexit.register(lambda: _safe_close_pool(_pg_pool))

        saver = PostgresSaver(_pg_pool)

        try:
            saver.setup()
        except Exception as setup_err:
            err_str = str(setup_err).lower()
            # Supabase/PgBouncer raises errors for concurrent DDL or when
            # tables already exist — safe to ignore.
            if any(k in err_str for k in (
                "create index concurrently",
                "already exists",
                "cannot run inside a transaction",
                "relation",
            )):
                pass  # Tables/indexes already exist — fine
            else:
                print(f"PostgresSaver.setup() failed: {setup_err}")
                return None

        return saver

    except Exception as e:
        print("Error initializing PostgresSaver:", e)
        return None


def async_postgres_saver():
    """
    Return an AsyncPostgresSaver for streaming graph execution.

    We deliberately skip calling saver.setup() here because:
    1. The synchronous postgres_saver() (called first in main.py __init__)
       already ran setup() and created all necessary tables/indexes.
    2. Calling async setup() during FastAPI startup (before the event loop
       is running) would crash with RuntimeError or schedule a task on a
       loop that is later abandoned.
    """
    try:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            from psycopg_pool import AsyncConnectionPool
        except ImportError as import_err:
            print(
                "Async Postgres checkpoint support is unavailable: "
                f"{import_err}. Install psycopg[binary] in the backend environment."
            )
            return None

        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL must be set")

        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": None,
        }

        global _async_pg_pool
        if _async_pg_pool is None:
            _async_pg_pool = AsyncConnectionPool(
                conninfo=db_url,
                max_size=20,
                kwargs=connection_kwargs,
                open=False,  # Don't open immediately; let it open lazily
            )
            atexit.register(lambda: _safe_close_pool(_async_pg_pool))

        return AsyncPostgresSaver(_async_pg_pool)

    except Exception as e:
        print("Error initializing AsyncPostgresSaver:", e)
        return None


def _safe_close_pool(pool):
    try:
        if pool:
            pool.close()
    except Exception:
        pass
