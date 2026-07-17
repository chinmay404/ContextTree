-- 006_frontend_init_shape_alignment.sql — make every table the frontend's
-- init block touches carry every column it indexes or alters.
--
-- Prod carried legacy-shaped tables (created by older frontend inits before
-- the V2 baseline; CREATE TABLE IF NOT EXISTS no-ops over them). The init's
-- `create index ... on file_chunks(file_id)` threw 42703, aborting the whole
-- init — every canvas/node/message write 500'd (same failure class as 005's
-- bug_reports.status). This aligns ALL of them in one pass; each statement
-- is an if-not-exists no-op where the column already exists.

-- file RAG tables
ALTER TABLE IF EXISTS file_chunks ADD COLUMN IF NOT EXISTS file_id text;
ALTER TABLE IF EXISTS file_chunks ADD COLUMN IF NOT EXISTS chunk_index integer;
ALTER TABLE IF EXISTS file_chunks ADD COLUMN IF NOT EXISTS chunk_text text;
ALTER TABLE IF EXISTS file_chunks ADD COLUMN IF NOT EXISTS embedding vector(768);
ALTER TABLE IF EXISTS file_chunks ADD COLUMN IF NOT EXISTS metadata jsonb;
ALTER TABLE IF EXISTS file_chunks ADD COLUMN IF NOT EXISTS created_at timestamptz DEFAULT now();

ALTER TABLE IF EXISTS context_chunks ADD COLUMN IF NOT EXISTS file_id text;
ALTER TABLE IF EXISTS context_chunks ADD COLUMN IF NOT EXISTS content text;
ALTER TABLE IF EXISTS context_chunks ADD COLUMN IF NOT EXISTS embedding vector(768);
ALTER TABLE IF EXISTS context_chunks ADD COLUMN IF NOT EXISTS chunk_index integer;
ALTER TABLE IF EXISTS context_chunks ADD COLUMN IF NOT EXISTS token_count integer;
ALTER TABLE IF EXISTS context_chunks ADD COLUMN IF NOT EXISTS created_at timestamptz DEFAULT now();

ALTER TABLE IF EXISTS external_files ADD COLUMN IF NOT EXISTS node_id text;
ALTER TABLE IF EXISTS external_files ADD COLUMN IF NOT EXISTS canvas_id text;
ALTER TABLE IF EXISTS external_files ADD COLUMN IF NOT EXISTS user_email text;
ALTER TABLE IF EXISTS external_files ADD COLUMN IF NOT EXISTS file_name text;
ALTER TABLE IF EXISTS external_files ADD COLUMN IF NOT EXISTS file_type text;
ALTER TABLE IF EXISTS external_files ADD COLUMN IF NOT EXISTS file_size integer;
ALTER TABLE IF EXISTS external_files ADD COLUMN IF NOT EXISTS content text;
ALTER TABLE IF EXISTS external_files ADD COLUMN IF NOT EXISTS data bytea;
ALTER TABLE IF EXISTS external_files ADD COLUMN IF NOT EXISTS mime_type text;
ALTER TABLE IF EXISTS external_files ADD COLUMN IF NOT EXISTS processed boolean DEFAULT false;
ALTER TABLE IF EXISTS external_files ADD COLUMN IF NOT EXISTS created_at timestamptz DEFAULT now();
ALTER TABLE IF EXISTS external_files ADD COLUMN IF NOT EXISTS updated_at timestamptz DEFAULT now();

-- edges (indexed on canvas_id/from_node/to_node by the init)
ALTER TABLE IF EXISTS edges ADD COLUMN IF NOT EXISTS canvas_id text;
ALTER TABLE IF EXISTS edges ADD COLUMN IF NOT EXISTS user_email text;
ALTER TABLE IF EXISTS edges ADD COLUMN IF NOT EXISTS from_node text;
ALTER TABLE IF EXISTS edges ADD COLUMN IF NOT EXISTS to_node text;
ALTER TABLE IF EXISTS edges ADD COLUMN IF NOT EXISTS data jsonb;
ALTER TABLE IF EXISTS edges ADD COLUMN IF NOT EXISTS created_at timestamptz DEFAULT now();
ALTER TABLE IF EXISTS edges ADD COLUMN IF NOT EXISTS updated_at timestamptz DEFAULT now();

-- BYOK keys (init alters encrypted_key and indexes provider)
ALTER TABLE IF EXISTS user_api_keys ADD COLUMN IF NOT EXISTS encrypted_key text;
ALTER TABLE IF EXISTS user_api_keys ADD COLUMN IF NOT EXISTS key_hint text;
ALTER TABLE IF EXISTS user_api_keys ADD COLUMN IF NOT EXISTS provider text;
ALTER TABLE IF EXISTS user_api_keys ADD COLUMN IF NOT EXISTS metadata jsonb DEFAULT '{}'::jsonb;
ALTER TABLE IF EXISTS user_api_keys ADD COLUMN IF NOT EXISTS created_at timestamptz DEFAULT now();
ALTER TABLE IF EXISTS user_api_keys ADD COLUMN IF NOT EXISTS updated_at timestamptz DEFAULT now();
