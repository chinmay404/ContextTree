-- 010_message_citations.sql — the citations a V3 answer used, stored with the assistant message.
-- Contract: ContextTree-src/docs/superpowers/plans/2026-10-19-v3-api-contract.md,
-- "Amendment 2026-10-20: citations and source passages". Shape: [{n, cardId, loc, page, quote}].
-- Additive and idempotent: one nullable column, no default (a metadata-only change, no table
-- rewrite). V2 writes never name it and keep working; NULL = not a V3 answer / written before 010.

-- Fail fast instead of queueing behind (and blocking) live traffic on the messages table.
-- migrate.py runs each file in one transaction, so SET LOCAL covers every statement below.
SET LOCAL lock_timeout = '5s';

ALTER TABLE messages ADD COLUMN IF NOT EXISTS citations jsonb;

-- ADD COLUMN IF NOT EXISTS never fixes a wrong column type: abort on drift.
DO $$
DECLARE typ text;
BEGIN
  SELECT c.data_type INTO typ
  FROM information_schema.columns c
  WHERE c.table_schema = current_schema() AND c.table_name = 'messages' AND c.column_name = 'citations';
  IF typ IS DISTINCT FROM 'jsonb' THEN
    RAISE EXCEPTION '010_message_citations: messages.citations is %, want jsonb', typ;
  END IF;
END $$;
