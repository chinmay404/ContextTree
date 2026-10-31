"""
Re-embed stored file chunks and messages after changing the embedding provider.

Vectors from two models can't be compared, so search is unreliable until this finishes. Keyword
search keeps working meanwhile. It stops at the first failed batch rather than store zero vectors,
and it can be run again: every row just gets its vector rewritten.

Usage (DATABASE_URL and the provider settings come from the environment):
    python scripts/reembed.py          # count the rows, change nothing
    python scripts/reembed.py --apply  # do it
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.utils.embeddings import get_embeddings  # noqa: E402

BATCH = 64
PAUSE_SECONDS = 1.6  # stays under NIM's reported 40 requests a minute
_TABLES = {
    "file_chunks": ("SELECT id, chunk_text FROM file_chunks WHERE id > %s AND chunk_text IS NOT NULL ORDER BY id LIMIT %s", 0),
    "messages": ("SELECT id, content FROM messages WHERE id > %s AND content IS NOT NULL ORDER BY id LIMIT %s", ""),
}


def _vec(values) -> str:
    return "[" + ",".join(repr(float(x)) for x in values) + "]"


def reembed(conn, apply: bool = False, batch: int = BATCH, pause: float = 0.0) -> dict:
    counts = {}
    for table, (select, start) in _TABLES.items():
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {table}")
            counts[table] = cur.fetchone()[0]
        if not apply:
            continue
        last, done = start, 0
        while True:
            with conn.cursor() as cur:
                cur.execute(select, (last, batch))
                rows = cur.fetchall()
            if not rows:
                break
            vectors = get_embeddings([r[1] for r in rows])
            # A zero vector is how a failed call looks; an oversized single text may stay as it was.
            if not any(any(v) for v in vectors):
                raise SystemExit(f"Provider failed in {table} after {done} rows. Nothing more was written.")
            with conn.cursor() as cur:
                for row, vector in zip(rows, vectors):
                    if any(vector):
                        cur.execute(f"UPDATE {table} SET embedding = %s::vector WHERE id = %s", (_vec(vector), row[0]))
            conn.commit()
            last, done = rows[-1][0], done + len(rows)
            print(f"{table}: {done}/{counts[table]}", flush=True)
            time.sleep(pause)
    return counts


def main() -> int:
    import psycopg2

    url = os.environ.get("DATABASE_URL")
    if not url:
        print("ERROR: DATABASE_URL is not set")
        return 1
    apply = "--apply" in sys.argv
    conn = psycopg2.connect(url)
    try:
        counts = reembed(conn, apply=apply, pause=PAUSE_SECONDS)
    finally:
        conn.close()
    print(("Re-embedded: " if apply else "Would re-embed: ") + ", ".join(f"{k} {v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
