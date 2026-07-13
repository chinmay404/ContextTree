-- 004_message_positions.sql — backfill NULL message positions.
--
-- messages.position was never populated on insert (add_message and the fork
-- buffer both omitted it), leaving every row NULL. The hydration query
-- (get_thread_messages_after) filters `position > watermark`; NULL > 0 is
-- never true in SQL, so forked branches inherited nothing and the summary
-- watermark could never advance. add_message / _insert_buffer_messages now
-- assign position = MAX(position)+1 per node; this migration heals the
-- pre-existing rows so old canvases inherit correctly too.
--
-- DEPLOY ORDER: apply this migration BEFORE (or together with) deploying the
-- add_message/_insert_buffer_messages position change. If the code ships
-- first, a new message on a still-all-NULL node computes MAX(NULL)+1 = 1
-- (colliding with legacy siblings) until this migration renumbers the
-- partition. It is self-healing once applied (the UPDATE below renumbers the
-- whole node by timestamp), but running the migration first avoids the
-- transient window entirely.

WITH ordered AS (
    SELECT id,
           ROW_NUMBER() OVER (PARTITION BY node_id ORDER BY timestamp, id) AS rn
    FROM messages
)
UPDATE messages m
SET position = o.rn
FROM ordered o
WHERE m.id = o.id
  AND m.position IS DISTINCT FROM o.rn;
