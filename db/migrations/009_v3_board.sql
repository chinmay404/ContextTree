-- 009_v3_board.sql — V3 board columns. Additive, idempotent, safe for the live V2 frontend.
-- Spec: ContextTree-src/docs/superpowers/specs/2026-10-19-v3-board-design.md §5.6, §8.
-- Nothing is renamed, dropped, retyped or made NOT NULL. Re-running is a no-op
-- (IF NOT EXISTS everywhere; the backfills only touch kind IS NULL rows).

ALTER TABLE nodes ADD COLUMN IF NOT EXISTS kind text;            -- 'chat' | 'source'
ALTER TABLE nodes ADD COLUMN IF NOT EXISTS source_type text;     -- note|pdf|doc|web|video|audio|csv|image|pile
ALTER TABLE nodes ADD COLUMN IF NOT EXISTS x double precision;
ALTER TABLE nodes ADD COLUMN IF NOT EXISTS y double precision;
ALTER TABLE nodes ADD COLUMN IF NOT EXISTS desk_id text REFERENCES nodes(id) ON DELETE SET NULL;
ALTER TABLE nodes ADD COLUMN IF NOT EXISTS pile_id text REFERENCES nodes(id) ON DELETE SET NULL;
ALTER TABLE nodes ADD COLUMN IF NOT EXISTS status text;          -- processing|ready|failed|soon (sources)
ALTER TABLE nodes ADD COLUMN IF NOT EXISTS deleted_at timestamptz;
ALTER TABLE edges ADD COLUMN IF NOT EXISTS kind text;            -- lineage|wire|answer
ALTER TABLE edges ADD COLUMN IF NOT EXISTS deleted_at timestamptz;
-- Prod's edges table was created by the V2 frontend init, which has no `meta`.
-- The backfill below reads both, so make sure both exist.
ALTER TABLE edges ADD COLUMN IF NOT EXISTS meta jsonb;
ALTER TABLE edges ADD COLUMN IF NOT EXISTS data jsonb;
ALTER TABLE canvases ADD COLUMN IF NOT EXISTS title text;
ALTER TABLE canvases ADD COLUMN IF NOT EXISTS board_version smallint;  -- 3 = V3 board; NULL = V2 canvas
ALTER TABLE canvases ADD COLUMN IF NOT EXISTS deleted_at timestamptz;

-- ADD COLUMN IF NOT EXISTS never fixes a wrong column type: abort on drift.
DO $$
DECLARE drift text;
BEGIN
  SELECT string_agg(format('%s.%s is %s, want %s', c.table_name, c.column_name, c.data_type, w.typ), '; ')
    INTO drift
  FROM (VALUES
    ('nodes','kind','text'), ('nodes','source_type','text'),
    ('nodes','x','double precision'), ('nodes','y','double precision'),
    ('nodes','desk_id','text'), ('nodes','pile_id','text'),
    ('nodes','status','text'), ('nodes','deleted_at','timestamp with time zone'),
    ('edges','kind','text'), ('edges','deleted_at','timestamp with time zone'),
    ('edges','meta','jsonb'), ('edges','data','jsonb'),
    ('canvases','title','text'), ('canvases','board_version','smallint'),
    ('canvases','deleted_at','timestamp with time zone')
  ) AS w(tbl, col, typ)
  JOIN information_schema.columns c
    ON c.table_schema = current_schema() AND c.table_name = w.tbl AND c.column_name = w.col
  WHERE c.data_type <> w.typ;
  IF drift IS NOT NULL THEN
    RAISE EXCEPTION '009_v3_board: column type drift: %', drift;
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS nodes_desk_idx ON nodes(desk_id);
CREATE INDEX IF NOT EXISTS nodes_pile_idx ON nodes(pile_id);
CREATE INDEX IF NOT EXISTS edges_to_kind_idx ON edges(to_node, kind);

-- §8 backfill: edges.kind only. V2 canvases are not converted.
UPDATE edges SET kind = 'lineage'
WHERE kind IS NULL
  AND COALESCE(meta->>'condition', data->'meta'->>'condition') = 'Fork';

-- V2 stored chat→file. Direction is left as V2 wrote it: V3 never reads V2 canvases.
UPDATE edges SET kind = 'wire'
WHERE kind IS NULL
  AND COALESCE(meta->>'condition', data->'meta'->>'condition') = 'Context';
