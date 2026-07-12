-- 002_uuid_identity.sql — identity becomes users.id (uuid), email demoted.
--
-- V2 architecture §2/§3: email is PII and mutable; the tenant key must be
-- a stable UUID. The NextAuth adapter already generates crypto.randomUUID()
-- for users.id, so the text→uuid cast is lossless.
--
-- This migration is ADDITIVE on the conversation tables: it adds user_id
-- uuid columns (backfilled from users by email) alongside user_email.
-- The user_email columns are dropped in a later migration, after the
-- application code sweep (PostgresStore + the interim Next.js data layer)
-- switches to user_id and the single-writer refactor (plan Day 6) lands.
--
-- Guard: if users.id contains any value that is not a valid UUID, the
-- whole migration aborts (transaction rolls back) with a clear error.

-- ── Guard: every existing users.id must be castable to uuid ──────────────
DO $$
DECLARE bad_count int;
BEGIN
    SELECT count(*) INTO bad_count FROM users
    WHERE id !~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$';
    IF bad_count > 0 THEN
        RAISE EXCEPTION
            'users.id contains % non-UUID value(s); resolve manually before 002',
            bad_count;
    END IF;
END $$;

-- ── users.id: text → uuid, server-side default ───────────────────────────
ALTER TABLE accounts DROP CONSTRAINT IF EXISTS accounts_user_id_fkey;
ALTER TABLE sessions DROP CONSTRAINT IF EXISTS sessions_user_id_fkey;

ALTER TABLE users
    ALTER COLUMN id TYPE uuid USING id::uuid,
    ALTER COLUMN id SET DEFAULT gen_random_uuid();

ALTER TABLE accounts ALTER COLUMN user_id TYPE uuid USING user_id::uuid;
ALTER TABLE sessions ALTER COLUMN user_id TYPE uuid USING user_id::uuid;

ALTER TABLE accounts
    ADD CONSTRAINT accounts_user_id_fkey
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE sessions
    ADD CONSTRAINT sessions_user_id_fkey
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;

-- ── Conversation tables: add user_id uuid alongside user_email ───────────
ALTER TABLE canvases       ADD COLUMN IF NOT EXISTS user_id uuid REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE nodes          ADD COLUMN IF NOT EXISTS user_id uuid REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE messages       ADD COLUMN IF NOT EXISTS user_id uuid REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE external_files ADD COLUMN IF NOT EXISTS user_id uuid REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE user_api_keys  ADD COLUMN IF NOT EXISTS user_id uuid REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE bug_reports    ADD COLUMN IF NOT EXISTS user_id uuid REFERENCES users(id) ON DELETE SET NULL;

-- Backfill from the email linkage where rows exist (no-op on a fresh DB).
UPDATE canvases       c SET user_id = u.id FROM users u WHERE c.user_id IS NULL AND c.user_email = u.email;
UPDATE nodes          n SET user_id = u.id FROM users u WHERE n.user_id IS NULL AND n.user_email = u.email;
UPDATE messages       m SET user_id = u.id FROM users u WHERE m.user_id IS NULL AND m.user_email = u.email;
UPDATE external_files f SET user_id = u.id FROM users u WHERE f.user_id IS NULL AND f.user_email = u.email;
UPDATE user_api_keys  k SET user_id = u.id FROM users u WHERE k.user_id IS NULL AND k.user_email = u.email;
UPDATE bug_reports    b SET user_id = u.id FROM users u WHERE b.user_id IS NULL AND b.user_email = u.email;

CREATE INDEX IF NOT EXISTS idx_canvases_user  ON canvases(user_id);
CREATE INDEX IF NOT EXISTS idx_nodes_user     ON nodes(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_user  ON messages(user_id);
