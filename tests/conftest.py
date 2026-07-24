"""
Test-suite safety net.

The store keeps a MODULE-LEVEL connection pool bound to whatever DATABASE_URL
is set when the first app module is imported — and chat.py builds the graph
(and thus the pool) at import time. Individual test files setting
os.environ["DATABASE_URL"] at their own import was too late whenever an
earlier-collected module imported app code first: the pool silently bound to
the PROD url from .env and test writes leaked into production
(@example.com users/canvases/messages observed on 2026-07-12, -13 and -24).

conftest.py is imported by pytest BEFORE any test module, so this runs first:

- If TEST_DATABASE_URL is set, force DATABASE_URL to it for the whole run
  (load_dotenv never overrides an existing env var, so .env cannot undo this).
- If it is not set, poison DATABASE_URL with an unreachable sentinel — same
  reasoning: load_dotenv skips already-set vars, so .env's prod url can never
  win. Tests guarded by skipif(TEST_DATABASE_URL) skip as before; unguarded
  DB tests fail fast on the sentinel instead of writing to prod.
"""

import os

TEST_DB = os.getenv("TEST_DATABASE_URL")

os.environ["DATABASE_URL"] = TEST_DB or (
    "postgresql://tests-require-TEST_DATABASE_URL:5432/blocked"
)
