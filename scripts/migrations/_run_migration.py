"""
One-shot runner for a single SQL migration file. Reads DATABASE_URL from the
project's .env, executes the file as one server-side transaction, prints the
result, and exits.

Usage:
    python scripts/migrations/_run_migration.py scripts/migrations/2026_04_26_context_tree_core.sql
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: _run_migration.py <path-to-sql-file>", file=sys.stderr)
        return 2

    sql_path = Path(sys.argv[1]).resolve()
    if not sql_path.is_file():
        print(f"file not found: {sql_path}", file=sys.stderr)
        return 2

    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL is not set in .env", file=sys.stderr)
        return 2

    sql = sql_path.read_text()
    print(f"running migration: {sql_path.name}")
    print(f"target host: {db_url.split('@', 1)[-1].split('/', 1)[0]}")

    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()

    # Re-open a fresh connection for verification (the file's own COMMIT
    # already finalized the changes).
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name, data_type, column_default
                FROM information_schema.columns
                WHERE table_name = 'nodes'
                  AND column_name IN ('ancestor_ids', 'summarized_up_to_position', 'is_initialized')
                ORDER BY column_name
            """)
            rows = cur.fetchall()
            print("\nnew columns on nodes:")
            for name, dtype, default in rows:
                print(f"  {name:30s} {dtype:20s} default={default}")

            cur.execute("SELECT count(*) FROM nodes")
            n_nodes = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM messages")
            n_messages = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM canvases")
            n_canvases = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM users")
            n_users = cur.fetchone()[0]
            print("\nrow counts after migration:")
            print(f"  canvases  = {n_canvases}")
            print(f"  nodes     = {n_nodes}")
            print(f"  messages  = {n_messages}")
            print(f"  users     = {n_users}  (preserved)")

    print("\nmigration complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
