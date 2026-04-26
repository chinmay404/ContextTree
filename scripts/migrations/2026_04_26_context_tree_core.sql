-- Context Tree core schema migration (2026-04-26)
--
-- Adds the columns the new branching/context code relies on:
--   nodes.ancestor_ids               text[]      materialized ancestry chain
--   nodes.summarized_up_to_position  bigint      watermark; messages with
--                                                position <= watermark are
--                                                folded into nodes.summary
--   nodes.is_initialized             boolean     fork-init idempotency flag
--
-- This script is split into two phases:
--   PHASE A — clear conversation/canvas data (you've authorized this)
--   PHASE B — schema changes (idempotent, safe to re-run)
--
-- Tables PRESERVED across both phases:
--   users, user_api_keys, bug_reports, waitlist
--
-- Tables CLEARED in Phase A:
--   canvases, nodes, messages, edges, external_files, file_chunks,
--   plus LangGraph checkpoint tables (they reference now-deleted threads)
--
-- Run as a single transaction so a failure mid-way leaves nothing partial:
--
--     psql "$DATABASE_URL" -1 -f 2026_04_26_context_tree_core.sql

BEGIN;

-- ─────────────────────────────────────────────────────────────────────────────
-- PHASE A — clear conversation data
-- ─────────────────────────────────────────────────────────────────────────────

-- Order doesn't matter with TRUNCATE ... CASCADE, but listing all of them
-- explicitly makes intent obvious in code review and avoids cascading into
-- anything we didn't list.
TRUNCATE TABLE
    file_chunks,
    external_files,
    edges,
    messages,
    nodes,
    canvases
RESTART IDENTITY CASCADE;

-- LangGraph PostgresSaver tables. These keep per-thread checkpoints, so any
-- stale entries would point at thread_ids that no longer exist. Drop only if
-- they are present (they're created lazily on first use).
DO $$
DECLARE
    t text;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'checkpoints',
        'checkpoint_writes',
        'checkpoint_blobs',
        'checkpoint_migrations'
    ]
    LOOP
        IF EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = current_schema() AND table_name = t
        ) THEN
            EXECUTE format('TRUNCATE TABLE %I RESTART IDENTITY CASCADE', t);
        END IF;
    END LOOP;
END $$;

-- ─────────────────────────────────────────────────────────────────────────────
-- PHASE B — schema additions on `nodes`
-- ─────────────────────────────────────────────────────────────────────────────

ALTER TABLE nodes
    ADD COLUMN IF NOT EXISTS ancestor_ids              text[]  NOT NULL DEFAULT '{}'::text[],
    ADD COLUMN IF NOT EXISTS summarized_up_to_position bigint  NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS is_initialized            boolean NOT NULL DEFAULT true;

-- GIN index lets us answer "all descendants of node X" via WHERE ancestor_ids @> ARRAY[$1]
-- without a recursive CTE. Cheap to maintain, valuable later for analytics and
-- cascading soft-deletes.
CREATE INDEX IF NOT EXISTS nodes_ancestor_ids_gin
    ON nodes USING gin (ancestor_ids);

-- ─────────────────────────────────────────────────────────────────────────────
-- Sanity checks (will raise if something's off)
-- ─────────────────────────────────────────────────────────────────────────────
DO $$
BEGIN
    -- All three new columns exist
    PERFORM 1 FROM information_schema.columns
        WHERE table_name = 'nodes' AND column_name = 'ancestor_ids';
    IF NOT FOUND THEN RAISE EXCEPTION 'ancestor_ids column missing'; END IF;

    PERFORM 1 FROM information_schema.columns
        WHERE table_name = 'nodes' AND column_name = 'summarized_up_to_position';
    IF NOT FOUND THEN RAISE EXCEPTION 'summarized_up_to_position column missing'; END IF;

    PERFORM 1 FROM information_schema.columns
        WHERE table_name = 'nodes' AND column_name = 'is_initialized';
    IF NOT FOUND THEN RAISE EXCEPTION 'is_initialized column missing'; END IF;

    -- Conversation tables empty
    IF (SELECT count(*) FROM nodes) <> 0 THEN
        RAISE EXCEPTION 'nodes not empty after Phase A';
    END IF;
    IF (SELECT count(*) FROM messages) <> 0 THEN
        RAISE EXCEPTION 'messages not empty after Phase A';
    END IF;
END $$;

COMMIT;
