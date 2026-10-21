# Deploying the engine on Vercel

The FastAPI engine runs as its own Vercel project: one Python function on the Hobby plan,
with Fluid compute and up to 300 s per request. See the contract amendment of 2026-10-21 in
`ContextTree-src/docs/superpowers/plans/2026-10-19-v3-api-contract.md`.

`railway.json` and `render.yaml` are kept for the older hosts. They are not used on Vercel.

## What is in the repo

| File | Purpose |
|---|---|
| `main.py` | The entrypoint. Vercel's zero-config FastAPI detection loads the top-level `app` (`app.api.server:app`). |
| `vercel.json` | `functions["main.py"]`: `maxDuration: 300` (the Hobby maximum) and `excludeFiles` (tests, venvs, scripts, migrations, logs). `regions: ["dub1"]` puts the function in Dublin, next to Supabase `eu-west-1`. |
| `.python-version` | Python 3.12, Vercel's default and the version Render used. |
| `requirements.txt` | Runtime dependencies only. Test tools live in `requirements-dev.txt`. |
| `.vercelignore` | Keeps `.env*`, virtualenvs, tests and logs out of `vercel deploy` uploads. |

What the app does and does not do on Vercel:

- **No migrations at startup.** They run as a one-off command (see below).
- **Startup does one cheap step:** a best-effort `UPDATE` that marks sources stuck in `processing`
  for more than 10 minutes as failed.
- **The LangGraph checkpointer is built when the module is imported.**
  - `PostgresSaver.setup()` checks the checkpoint tables. On a database that is already set up,
    this is a single read.
  - If the database is unreachable, the import retries for up to about 35 s, then falls back to
    in-memory checkpoints.
- **The async checkpointer cannot start outside a running event loop.** The log shows
  `Error initializing AsyncPostgresSaver: no running event loop`. Streams then use the sync graph
  and send the whole answer in one frame. Local `uvicorn` behaves the same way.
- **Source ingest never runs after a response.** `POST /boards/{b}/sources/file|link` return 202.
  The client then calls `POST /boards/{b}/cards/{id}/ingest`, which runs ingest inside that request.
  A lease of 300 s stops two calls from ingesting the same source at once.

## Environment variables

Set these in Vercel → Project → Settings → Environment Variables, for Production (and Preview if
you use it). This list gives names only. The values are the ones the engine already uses.

### Required

| Name | Notes |
|---|---|
| `DATABASE_URL` | Supabase **transaction pooler** URL (port `6543`). Serverless instances come and go, and the session pooler (15 slots) runs out. |
| `BACKEND_JWT_SECRET` | Must equal the frontend's value. The frontend signs, and the engine verifies. |
| `BYOK_ENCRYPTION_SECRET` | Must equal the frontend's value. Without it, users' saved keys cannot be decrypted. |
| `GOOGLE_API_KEY` | Embeddings (768-dim) for ingest and retrieval. These run on the server key on every board. |
| `GROQ_API_KEY` | Source overviews (`SUMMARY_MODEL_NAME`, Groq-hosted by default) and V2 chat. |

### Only for V2 chat (server-key models)

`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `NVIDIA_API_KEY`. V3 board turns never use them: they run
on each user's own key.

### Optional

`DAILY_MESSAGE_LIMIT` (default 500), `LANGSMITH_TRACING` (set `false` unless you trace),
`LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `SUMMARY_MODEL_NAME`.

### Must NOT be set

| Name | Why |
|---|---|
| `OPENAI_ORG_ID`, `OPENAI_ORGANIZATION`, `OPENAI_PROJECT_ID` | The OpenAI SDK adds these to requests. V3 strips them, but they do not belong on a user's key. |
| `OPENAI_CUSTOM_HEADERS`, `ANTHROPIC_CUSTOM_HEADERS` | Same reason: they inject server headers. |
| `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_GENAI_USE_ENTERPRISE` | They switch Gemini to Vertex AI. V3 Gemini turns then fail with a 500. |
| `ENGINE_FAKE_LLM` | Fake answers and fake embeddings, for local runs only. The engine refuses it with a non-local `DATABASE_URL`. |

## One-off: run migrations

Run this from your machine, never at app startup, against the target database. Prefer the
**direct / session** connection (port `5432`) for DDL.

```bash
cd ContextTree
DATABASE_URL='<target database URL>' python scripts/migrate.py --dry-run   # what would run
DATABASE_URL='<target database URL>' python scripts/migrate.py             # apply
```

The migrations are forward-only and idempotent, and an advisory lock serialises concurrent runs.
`scripts/migrate.py` reads `.env` only for variables you have not set inline, so always pass
`DATABASE_URL` on the command line.

## Health check

`GET https://<engine-project>.vercel.app/api/v1/health/` returns `{"status": "healthy", ...}`
with `deploy.sha` set from `VERCEL_GIT_COMMIT_SHA`. `GET /` also answers 200.

## Running it locally the way Vercel does

`vercel dev` needs a linked project (`vercel link`). Without one, run the same ASGI app:

```bash
# TEST_DATABASE_URL: the URL of your local Docker test database (never the production one)
DATABASE_URL="$TEST_DATABASE_URL" .venv/Scripts/python.exe -m uvicorn main:app --port 8010
```

Use a spare port. Port 8000 is reserved for the owner's local engine.

Tests: `pip install -r requirements-dev.txt`, then run `python -m pytest -q` with
`TEST_DATABASE_URL` set to the local Docker test database.
