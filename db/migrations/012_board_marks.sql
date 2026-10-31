-- 012_board_marks.sql — pen, arrow and highlight marks on the board and on note pages.
-- Additive and idempotent. Each mark is its own row, so saving one can never overwrite another, and
-- a later version can tell a model what was circled. The AI reads none of this yet.
--   anchor_id  the card this board mark moves with; points are then relative to that card
--   note_id    the note page this mark is on; NULL means the board itself
--   touches    the cards the mark lay over when drawn

SET LOCAL lock_timeout = '5s';

CREATE TABLE IF NOT EXISTS board_marks (
    id          text PRIMARY KEY,
    canvas_id   text NOT NULL REFERENCES canvases(id) ON DELETE CASCADE,
    user_email  text NOT NULL,
    kind        text NOT NULL CHECK (kind IN ('pen', 'arrow', 'highlight')),
    color       text NOT NULL,
    size        real NOT NULL DEFAULT 3,
    points      jsonb NOT NULL,
    anchor_id   text REFERENCES nodes(id) ON DELETE CASCADE,
    note_id     text REFERENCES nodes(id) ON DELETE CASCADE,
    touches     text[] NOT NULL DEFAULT '{}',
    created_at  timestamptz NOT NULL DEFAULT now(),
    deleted_at  timestamptz
);
CREATE INDEX IF NOT EXISTS board_marks_canvas_idx ON board_marks(canvas_id) WHERE deleted_at IS NULL;

-- Supabase can expose public tables through its anon key. The engine connects as the owner, which
-- these rules don't apply to. 011 shipped without this for its two tables, so it is set here too.
ALTER TABLE board_marks ENABLE ROW LEVEL SECURITY;
ALTER TABLE summary_blocks ENABLE ROW LEVEL SECURITY;
ALTER TABLE context_traces ENABLE ROW LEVEL SECURITY;
