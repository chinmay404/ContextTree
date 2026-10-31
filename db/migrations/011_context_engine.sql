-- 011_context_engine.sql — the context compiler's stored pieces. See context-eng.excalidraw.
-- Additive and idempotent: one nullable column and two new tables. V2 writes name none of them.
--   nodes.lineage_summary  what a branch inherited, frozen at its fork. Siblings forking from the
--                          same messages get the same text, so their prompts share a cacheable prefix.
--   summary_blocks         immutable summaries of a chat's messages from_position..to_position,
--                          reused by every later fork instead of a fresh LLM call.
--   context_traces         what one turn sent and left out, and why.

SET LOCAL lock_timeout = '5s';

ALTER TABLE nodes ADD COLUMN IF NOT EXISTS lineage_summary text;

CREATE TABLE IF NOT EXISTS summary_blocks (
    node_id        text NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    user_email     text NOT NULL,
    from_position  bigint NOT NULL,
    to_position    bigint NOT NULL,
    text           text NOT NULL,
    created_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (node_id, from_position, to_position)
);

CREATE TABLE IF NOT EXISTS context_traces (
    id          bigserial PRIMARY KEY,
    node_id     text NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    user_email  text NOT NULL,
    message_id  text,
    data        jsonb NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS context_traces_node_idx ON context_traces(node_id, id DESC);
