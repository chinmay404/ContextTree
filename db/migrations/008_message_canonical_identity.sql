-- 008_message_canonical_identity.sql
--
-- Message placement durability (2026-07-24). Three coupled fixes:
--
-- 1. `inherited` flag — fork-buffer copies (parent messages seeded into a
--    branch as LLM context) were distinguished from the branch's own history
--    by `timestamp < node.created_at`: a comparison across three different
--    clocks (browser, backend process, Postgres). Provenance is now explicit.
--
-- 2. Canonical message ids — the backend persisted "<base>" / "<base>_ai"
--    while the frontend persisted "<base>_u" / "<base>_a" for the SAME
--    logical message, so ON CONFLICT (id) never deduplicated and every turn
--    accumulated up to four rows (a 2-turn root chat showed "6 msgs").
--    All non-inherited rows converge on "<base>_u" (user) / "<base>_a"
--    (assistant); both writers now generate these forms.
--
-- 3. Inherited rows move to negative positions so they always order before
--    the branch's own messages (the first native message can be written
--    before fork init runs, which used to push inherited context after it).

ALTER TABLE messages ADD COLUMN IF NOT EXISTS inherited boolean NOT NULL DEFAULT false;
CREATE INDEX IF NOT EXISTS idx_messages_node_position ON messages(node_id, position);

-- 1) One-time backfill via the old heuristic: rows on a branch that are older
--    than the branch itself are fork-buffer copies.
UPDATE messages m
SET inherited = true
FROM nodes n
WHERE m.node_id = n.id
  AND n.is_primary IS FALSE
  AND m.timestamp < n.created_at
  AND NOT m.inherited;

-- 2) Shift inherited rows below all native positions, preserving their
--    relative order. Guarded so a manual re-run cannot double-shift.
UPDATE messages
SET position = position - 1000000
WHERE inherited
  AND position IS NOT NULL
  AND position > -900000;

-- 3) Converge duplicate twin rows: delete a non-canonical row when its
--    canonical twin already exists on the same node. Deleted rows are copied
--    to a backup table first (same shape as messages) for reversibility.
CREATE TABLE IF NOT EXISTS messages_canonical_backup_008
  (LIKE messages INCLUDING DEFAULTS);

WITH canon AS (
  SELECT id, node_id,
         CASE
           WHEN role = 'assistant'
             THEN regexp_replace(id, '(_assistant|-assistant|_ai|_a|-a)$', '') || '_a'
           ELSE regexp_replace(id, '(_user|-user|_u|-u)$', '') || '_u'
         END AS cid
  FROM messages
  WHERE NOT inherited
),
doomed AS (
  SELECT c.id
  FROM canon c
  JOIN messages t ON t.id = c.cid AND t.node_id = c.node_id
  WHERE c.cid <> c.id
),
backed_up AS (
  INSERT INTO messages_canonical_backup_008
  SELECT m.* FROM messages m JOIN doomed d ON d.id = m.id
)
DELETE FROM messages m USING doomed d WHERE m.id = d.id;

-- 4) Rename surviving non-canonical rows to their canonical id so future
--    upserts from either writer land on the same row. Skip any id whose
--    canonical form is claimed by another row (paranoia guards).
WITH canon AS (
  SELECT id,
         CASE
           WHEN role = 'assistant'
             THEN regexp_replace(id, '(_assistant|-assistant|_ai|_a|-a)$', '') || '_a'
           ELSE regexp_replace(id, '(_user|-user|_u|-u)$', '') || '_u'
         END AS cid
  FROM messages
  WHERE NOT inherited
),
uniq AS (
  SELECT c.id, c.cid
  FROM canon c
  WHERE c.cid <> c.id
    AND NOT EXISTS (SELECT 1 FROM messages t WHERE t.id = c.cid)
    AND c.cid IN (SELECT cid FROM canon GROUP BY cid HAVING count(*) = 1)
)
UPDATE messages m SET id = u.cid FROM uniq u WHERE m.id = u.id;
