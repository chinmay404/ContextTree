-- 003_quotas.sql — durable per-user daily quota buckets (V2 arch §2.5).
--
-- SlowAPI's in-process limiter remains as a burst guard (per-minute), but
-- the daily cap must survive restarts and count across workers, so it
-- lives here. user_key is the VERIFIED identity from the service JWT
-- (email today, uuid after the identity sweep) — never client input.

CREATE TABLE IF NOT EXISTS quotas (
    user_key   text NOT NULL,
    day        date NOT NULL,
    msg_count  integer NOT NULL DEFAULT 0,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_key, day)
);
