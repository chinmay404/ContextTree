-- 001_baseline.sql — V1 as-built schema, consolidated (2026-07-12).
--
-- Derived from: scripts/setup-production-db.js (Next.js), lib/auth.ts and
-- lib/mongodb.ts runtime DDL (Next.js), app/api/waitlist/route.ts,
-- scripts/migrations/2026_04_26_context_tree_core.sql, and the column sets
-- actually read/written by app/agent/store/PostgresStore.py.
-- The old Supabase project was unreachable at authoring time, so this file
-- is the code-truth baseline; every statement is idempotent (IF NOT EXISTS)
-- so it is safe on both a fresh database and a surviving one.
--
-- Vector columns are 768-dim (Gemini embeddings — V2 decision; see
-- V2/01-CURRENT-STATE-AUDIT.md item A7). LangGraph checkpoint tables are
-- NOT defined here: the PostgresSaver owns and creates them itself.

CREATE EXTENSION IF NOT EXISTS vector;

-- ── Identity (NextAuth v4, database sessions) ─────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    id              text PRIMARY KEY,
    email           text UNIQUE NOT NULL,
    name            text,
    image           text,
    email_verified  timestamptz,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),
    canvas_ids      text[] DEFAULT array[]::text[],
    canvas_count    integer DEFAULT 0,
    total_nodes     integer DEFAULT 0
);

CREATE TABLE IF NOT EXISTS accounts (
    id                   text PRIMARY KEY,
    user_id              text NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    type                 text NOT NULL,
    provider             text NOT NULL,
    provider_account_id  text NOT NULL,
    refresh_token        text,
    access_token         text,
    expires_at           bigint,
    token_type           text,
    scope                text,
    id_token             text,
    session_state        text,
    created_at           timestamptz NOT NULL DEFAULT now(),
    updated_at           timestamptz NOT NULL DEFAULT now(),
    UNIQUE (provider, provider_account_id)
);

CREATE TABLE IF NOT EXISTS sessions (
    id             text PRIMARY KEY,
    session_token  text NOT NULL UNIQUE,
    user_id        text NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires        timestamptz NOT NULL,
    created_at     timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS verification_tokens (
    identifier  text NOT NULL,
    token       text NOT NULL,
    expires     timestamptz NOT NULL,
    PRIMARY KEY (identifier, token)
);

-- ── Conversation data ─────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS canvases (
    id          text PRIMARY KEY,
    user_email  text NOT NULL,
    data        jsonb NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS nodes (
    id                         text PRIMARY KEY,
    canvas_id                  text NOT NULL,
    user_email                 text NOT NULL,
    data                       jsonb,
    type                       text,
    parent_node_id             text,
    forked_from_message_id     text,
    is_primary                 boolean,
    summary                    text,
    summary_embedding          vector(768),
    ancestor_ids               text[]  NOT NULL DEFAULT '{}'::text[],
    summarized_up_to_position  bigint  NOT NULL DEFAULT 0,
    is_initialized             boolean NOT NULL DEFAULT true,
    created_at                 timestamptz NOT NULL DEFAULT now(),
    updated_at                 timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id          text PRIMARY KEY,
    node_id     text NOT NULL,
    canvas_id   text,
    user_email  text,
    role        text NOT NULL,
    content     text NOT NULL,
    embedding   vector(768),
    position    bigint,
    timestamp   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS edges (
    id          text PRIMARY KEY,
    canvas_id   text,
    from_node   text NOT NULL,
    to_node     text NOT NULL,
    meta        jsonb,
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- ── External context / RAG (dormant in V2 launch; FILES_ENABLED=false) ───

CREATE TABLE IF NOT EXISTS external_files (
    id          text PRIMARY KEY,
    user_email  text,
    node_id     text,
    canvas_id   text,
    file_name   text,
    file_type   text,
    file_size   bigint,
    content     text,
    data        jsonb,
    mime_type   text,
    processed   boolean DEFAULT false,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS file_chunks (
    id          bigserial PRIMARY KEY,
    file_id     text REFERENCES external_files(id) ON DELETE CASCADE,
    chunk_index integer,
    chunk_text  text,
    embedding   vector(768),
    metadata    jsonb
);

CREATE TABLE IF NOT EXISTS context_chunks (
    id          bigserial PRIMARY KEY,
    canvas_id   text,
    node_id     text,
    content     text,
    embedding   vector(768),
    metadata    jsonb,
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- ── Account features ──────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS user_api_keys (
    user_email     text NOT NULL,
    provider       text NOT NULL,
    encrypted_key  text NOT NULL,
    key_hint       text,
    metadata       jsonb,
    created_at     timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_email, provider)
);

CREATE TABLE IF NOT EXISTS bug_reports (
    id                  text PRIMARY KEY,
    user_email          text,
    user_name           text,
    title               text,
    description         text,
    severity            text,
    steps_to_reproduce  text,
    expected_behavior   text,
    actual_behavior     text,
    browser_info        jsonb,
    additional_info     text,
    created_at          timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS waitlist (
    id          serial PRIMARY KEY,
    name        varchar(255) NOT NULL,
    email       varchar(255) UNIQUE NOT NULL,
    created_at  timestamptz DEFAULT now(),
    updated_at  timestamptz DEFAULT now(),
    ip_address  inet,
    user_agent  text,
    status      varchar(50) DEFAULT 'pending'
);

-- ── Indexes ───────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_nodes_canvas        ON nodes(canvas_id);
CREATE INDEX IF NOT EXISTS idx_nodes_parent        ON nodes(parent_node_id);
CREATE INDEX IF NOT EXISTS idx_nodes_forked_from   ON nodes(forked_from_message_id);
CREATE INDEX IF NOT EXISTS nodes_ancestor_ids_gin  ON nodes USING gin (ancestor_ids);
CREATE INDEX IF NOT EXISTS idx_messages_node       ON messages(node_id);
CREATE INDEX IF NOT EXISTS idx_messages_canvas     ON messages(canvas_id);
CREATE INDEX IF NOT EXISTS idx_edges_canvas        ON edges(canvas_id);
CREATE INDEX IF NOT EXISTS idx_file_chunks_file    ON file_chunks(file_id);

-- ivfflat needs rows to build good lists; fine to create up front.
CREATE INDEX IF NOT EXISTS idx_messages_embedding
    ON messages USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
