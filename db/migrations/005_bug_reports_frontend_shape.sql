-- 005_bug_reports_frontend_shape.sql — align bug_reports with the frontend.
--
-- The Next.js data layer's init DDL indexes bug_reports(status), but the
-- 001 baseline (V1 as-built) created the table without status/updated_at.
-- On the new prod DB that index statement threw "column does not exist",
-- aborting the frontend's entire init block — so canvas/node/message writes
-- 500'd silently, node rows were never persisted, and the backend saw forks
-- with no parent ("Fork init skipped: no parent resolvable"). The frontend
-- init also backfills these now; this migration is the schema's source of
-- truth so a fresh database never reproduces the mismatch.

ALTER TABLE bug_reports ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'open';
ALTER TABLE bug_reports ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT now();
