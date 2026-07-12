"""
Forward-only SQL migration runner (V2 architecture §4).

- Migrations live in db/migrations/NNN_name.sql, applied in filename order.
- Each file runs in ONE transaction; a schema_migrations row is inserted in
  the same transaction, so a failed migration leaves no partial state.
- A pg_advisory_xact_lock serializes concurrent deploys.

Usage:
    python scripts/migrate.py            # apply pending migrations
    python scripts/migrate.py --dry-run  # show what would run, change nothing
"""

import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = ROOT / "db" / "migrations"
LOCK_KEY = 728_431_902  # arbitrary constant, unique to this app's migrations

load_dotenv(ROOT / ".env")


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL is not set")
        return 1

    files = sorted(p for p in MIGRATIONS_DIR.glob("*.sql"))
    if not files:
        print(f"No migrations found in {MIGRATIONS_DIR}")
        return 1

    conn = psycopg2.connect(database_url)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        filename    text PRIMARY KEY,
                        applied_at  timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )

        with conn.cursor() as cur:
            cur.execute("SELECT filename FROM schema_migrations")
            applied = {r[0] for r in cur.fetchall()}
        conn.commit()

        pending = [f for f in files if f.name not in applied]
        if not pending:
            print(f"Up to date ({len(applied)} applied).")
            return 0

        for f in pending:
            if dry_run:
                print(f"WOULD APPLY: {f.name}")
                continue
            sql = f.read_text(encoding="utf-8")
            with conn:  # one transaction per migration file
                with conn.cursor() as cur:
                    cur.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_KEY,))
                    cur.execute(sql)
                    cur.execute(
                        "INSERT INTO schema_migrations (filename) VALUES (%s)",
                        (f.name,),
                    )
            print(f"APPLIED: {f.name}")

        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
