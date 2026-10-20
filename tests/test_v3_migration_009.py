import uuid
from pathlib import Path

from tests.v3_support import requires_db, db, user, seed_board, seed_card, seed_edge  # noqa: F401

pytestmark = requires_db
SQL = (Path(__file__).resolve().parent.parent / "db/migrations/009_v3_board.sql").read_text(encoding="utf-8")

EXPECTED = {
    ("nodes", "kind"): "text", ("nodes", "source_type"): "text",
    ("nodes", "x"): "double precision", ("nodes", "y"): "double precision",
    ("nodes", "desk_id"): "text", ("nodes", "pile_id"): "text",
    ("nodes", "status"): "text", ("nodes", "deleted_at"): "timestamp with time zone",
    ("edges", "kind"): "text", ("edges", "deleted_at"): "timestamp with time zone",
    ("edges", "meta"): "jsonb", ("edges", "data"): "jsonb",
    ("canvases", "title"): "text", ("canvases", "board_version"): "smallint",
    ("canvases", "deleted_at"): "timestamp with time zone",
}


def _columns(cur):
    cur.execute(
        "SELECT table_name, column_name, data_type FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name IN ('nodes','edges','canvases') "
        "ORDER BY table_name, column_name")
    return cur.fetchall()


def _indexes(cur):
    cur.execute(
        "SELECT tablename, indexname, indexdef FROM pg_indexes WHERE schemaname = 'public' "
        "AND tablename IN ('nodes','edges','canvases') ORDER BY indexname")
    return cur.fetchall()


def test_009_recorded_by_migrate(db):
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM schema_migrations WHERE filename = '009_v3_board.sql'")
        assert cur.fetchone()[0] == 1


def test_009_columns_have_expected_types(db):
    with db.cursor() as cur:
        cols = {(t, c): d for t, c, d in _columns(cur)}
    for key, typ in EXPECTED.items():
        assert cols.get(key) == typ, key


def test_009_indexes_exist(db):
    with db.cursor() as cur:
        names = {r[1] for r in _indexes(cur)}
    assert {"nodes_desk_idx", "nodes_pile_idx", "edges_to_kind_idx"} <= names


def test_009_desk_fk_sets_null_on_hard_delete(db, user):
    board = seed_board(db, user)
    chat = seed_card(db, user, board)
    src = seed_card(db, user, board, kind="source", source_type="note", desk_id=chat)
    with db.cursor() as cur:
        cur.execute("DELETE FROM nodes WHERE id = %s", (chat,))
        cur.execute("SELECT desk_id FROM nodes WHERE id = %s", (src,))
        assert cur.fetchone()[0] is None


def test_009_backfill_kinds(db, user):
    board = seed_board(db, user)
    a, b, c = (seed_card(db, user, board) for _ in range(3))
    f = seed_card(db, user, board, kind="source", source_type="note")
    e_meta = seed_edge(db, user, board, a, b, None, meta={"condition": "Fork"})
    e_data = seed_edge(db, user, board, a, c, None, data={"meta": {"condition": "Fork"}})
    e_ctx = seed_edge(db, user, board, a, f, None, meta={"condition": "Context"})
    e_ans = seed_edge(db, user, board, b, c, "answer")
    with db.cursor() as cur:
        cur.execute(SQL)
        cur.execute("SELECT id, kind, from_node, to_node FROM edges WHERE canvas_id = %s", (board,))
        rows = {r[0]: r[1:] for r in cur.fetchall()}
    assert rows[e_meta][0] == "lineage"
    assert rows[e_data][0] == "lineage"
    assert rows[e_ctx] == ("wire", a, f)  # direction untouched
    assert rows[e_ans] == ("answer", b, c)


def test_009_twice_is_noop(db, user):
    board = seed_board(db, user)
    a, b = seed_card(db, user, board), seed_card(db, user, board)
    seed_edge(db, user, board, a, b, None, meta={"condition": "Fork"})

    def snapshot(cur):
        cur.execute("SELECT id, kind, from_node, to_node FROM edges ORDER BY id")
        return _columns(cur), _indexes(cur), cur.fetchall()

    with db.cursor() as cur:
        cur.execute(SQL)
        before = snapshot(cur)
        cur.execute(SQL)
        assert snapshot(cur) == before


def test_009_applies_on_v2_frontend_shaped_tables(db):
    schema = f"v2shape_{uuid.uuid4().hex[:8]}"
    with db.cursor() as cur:
        cur.execute(f"CREATE SCHEMA {schema}")
        try:
            cur.execute(f"SET search_path TO {schema}")
            # Mirrors lib/mongodb.ts:70-131 (edges has no `meta` column).
            cur.execute("""
                create table users (id text primary key, email text unique not null, name text,
                  created_at timestamptz not null default now());
                create table canvases (id text primary key,
                  user_email text not null references users(email) on delete cascade,
                  data jsonb not null, created_at timestamptz not null default now(),
                  updated_at timestamptz not null default now());
                create table nodes (id text primary key,
                  canvas_id text not null references canvases(id) on delete cascade,
                  user_email text not null references users(email) on delete cascade,
                  data jsonb not null, parent_node_id text, forked_from_message_id text,
                  is_primary boolean, type text,
                  created_at timestamptz not null default now(),
                  updated_at timestamptz not null default now());
                create table edges (id text primary key,
                  canvas_id text not null references canvases(id) on delete cascade,
                  user_email text not null references users(email) on delete cascade,
                  from_node text not null, to_node text not null, data jsonb,
                  created_at timestamptz not null default now(),
                  updated_at timestamptz not null default now());
                insert into users (id, email) values ('u1', 'v2@example.com');
                insert into canvases (id, user_email, data) values ('cv', 'v2@example.com', '{}');
                insert into edges (id, canvas_id, user_email, from_node, to_node, data)
                  values ('e1', 'cv', 'v2@example.com', 'a', 'b', '{"meta":{"condition":"Fork"}}');
            """)
            cur.execute(SQL)
            cur.execute(
                "SELECT count(*) FROM information_schema.columns WHERE table_schema = %s "
                "AND table_name = 'edges' AND column_name = 'meta'", (schema,))
            assert cur.fetchone()[0] == 1
            cur.execute("SELECT kind FROM edges WHERE id = 'e1'")
            assert cur.fetchone()[0] == "lineage"
        finally:
            cur.execute("RESET search_path")
            cur.execute(f"DROP SCHEMA {schema} CASCADE")


def test_v2_shaped_writes_still_work_after_009(db, user):
    board = seed_board(db, user, version=None)
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO nodes (id, canvas_id, user_email, data, parent_node_id, "
            "forked_from_message_id, is_primary, type) "
            "VALUES (%s, %s, %s, '{}'::jsonb, NULL, NULL, true, 'chat')",
            (f"n_{uuid.uuid4().hex[:8]}", board, user))
        cur.execute(
            "INSERT INTO edges (id, canvas_id, user_email, from_node, to_node, data) "
            "VALUES (%s, %s, %s, 'a', 'b', '{}'::jsonb)",
            (f"e_{uuid.uuid4().hex[:8]}", board, user))
        cur.execute("SELECT kind FROM edges WHERE canvas_id = %s", (board,))
        assert cur.fetchone()[0] is None
        cur.execute("SELECT kind FROM nodes WHERE canvas_id = %s", (board,))
        assert cur.fetchone()[0] is None


def test_009_sets_local_lock_timeout_inside_its_transaction():
    import psycopg2
    from tests.v3_support import TEST_DB
    conn = psycopg2.connect(TEST_DB)  # autocommit off: same shape as migrate.py's per-file transaction
    try:
        with conn.cursor() as cur:
            cur.execute(SQL)
            cur.execute("SHOW lock_timeout")
            assert cur.fetchone()[0] == "5s"
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute("SHOW lock_timeout")
            assert cur.fetchone()[0] == "0"  # SET LOCAL does not leak past the transaction
    finally:
        conn.close()
