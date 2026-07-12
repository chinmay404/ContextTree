"""
Read-only schema introspection: prints every table's columns, defaults,
PKs/uniques/FKs, and indexes from the live database. Used once to author
db/migrations/001_baseline.sql truthfully; kept for future drift checks.

Usage:  python scripts/dump_schema.py
"""

import os

import psycopg2
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

cur.execute(
    """
    SELECT table_name FROM information_schema.tables
    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
    ORDER BY table_name
    """
)
tables = [r[0] for r in cur.fetchall()]

for t in tables:
    print(f"\n== {t} ==")
    cur.execute(
        """
        SELECT column_name,
               CASE WHEN data_type = 'USER-DEFINED' THEN udt_name
                    WHEN data_type = 'ARRAY' THEN udt_name
                    ELSE data_type END,
               is_nullable, column_default,
               character_maximum_length
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position
        """,
        (t,),
    )
    for name, dtype, nullable, default, maxlen in cur.fetchall():
        bits = [f"  {name}  {dtype}"]
        if maxlen:
            bits.append(f"({maxlen})")
        if nullable == "NO":
            bits.append(" NOT NULL")
        if default:
            bits.append(f" DEFAULT {default}")
        print("".join(bits))

    cur.execute(
        """
        SELECT conname, pg_get_constraintdef(oid)
        FROM pg_constraint
        WHERE conrelid = %s::regclass
        ORDER BY contype
        """,
        (t,),
    )
    for cname, cdef in cur.fetchall():
        print(f"  CONSTRAINT {cname}: {cdef}")

    cur.execute(
        """
        SELECT indexname, indexdef FROM pg_indexes
        WHERE schemaname = 'public' AND tablename = %s
        ORDER BY indexname
        """,
        (t,),
    )
    for iname, idef in cur.fetchall():
        print(f"  INDEX: {idef}")

# vector columns get reported as 'vector' without dims via info_schema; get real dims
print("\n== vector column dimensions ==")
cur.execute(
    """
    SELECT c.relname, a.attname, format_type(a.atttypid, a.atttypmod)
    FROM pg_attribute a
    JOIN pg_class c ON c.oid = a.attrelid
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public' AND format_type(a.atttypid, a.atttypmod) LIKE 'vector%'
      AND a.attnum > 0
    """
)
for rel, col, typ in cur.fetchall():
    print(f"  {rel}.{col}: {typ}")

conn.close()
