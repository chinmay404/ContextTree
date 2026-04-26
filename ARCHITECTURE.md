# Context Tree — Backend Architecture

This document describes the **as-built** behavior of the backend after the
2026-04-26 core migration. It supersedes the older `core_arcitetcure.md` and
`CONTEXT_TREE_REPORT.md`. If you find a divergence between this document and
the code, treat the code as truth and open a PR to update this file.

---

## 1. Mental model

A canvas is a forest of chat **nodes**. Each node owns a linear conversation
and three pieces of memory:

```
                              CANVAS
                                │
                            ┌───┴───┐
                            │ Node A │   ◄──── primary (root) thread
                            │   ●    │
                            │   ●    │
                            │   ●  ──┼─── fork at message m3
                            └────────┘
                                │
                          ┌─────┴─────┐
                          │           │
                      ┌───┴───┐   ┌───┴───┐
                      │Node B │   │Node C │
                      │   ●   │   │   ●   │   ◄──── two siblings, independent
                      │   ●   │   │   ●   │
                      └───────┘   └───────┘
```

### Per-node memory layers

Every node carries four memory layers that combine to form the system prompt
on each turn:

| Layer        | Lives in              | Bounded by                         |
| ------------ | --------------------- | ---------------------------------- |
| Working      | LangGraph state       | `KEEP_LAST_MESSAGES` raw msgs      |
| Episodic     | `nodes.summary`       | folded summary of older turns      |
| Semantic     | `nodes.data.memoryFacts` | small JSON of durable facts     |
| Retrieved    | computed each turn    | `SIMILAR_CONTEXT_LIMIT` snippets + `FILE_CONTEXT_LIMIT` file chunks |

Working + Episodic + Semantic + Retrieved → assembled by
`build_memory_context_block` and prepended to the chat history.

### The honest invariant

> A child node never reads parent **live state**: no shared LangGraph state, no
> reads of the parent's current `summary`, no implicit inheritance after the
> fork moment. Each node evolves independently.
>
> **However**, ancestry-scoped vector search (`find_similar_by_message_id`) lets
> a child retrieve relevant snippets from any ancestor's history, capped at the
> point where the child branched off. This is intentional: branches inherit
> *knowledge*, not *state*. If you want hard isolation set
> `SIMILAR_CONTEXT_LIMIT=0` per-call or skip the ancestry walk.

---

## 2. Schema (post-2026-04-26)

```
users (id, email, …)                         ◄── preserved across migrations
canvases (id, user_email, data jsonb)
nodes (
  id text PK,
  canvas_id, user_email,
  parent_node_id text NULL,
  forked_from_message_id text NULL,
  ancestor_ids text[] NOT NULL DEFAULT '{}',         ◄── NEW: root → parent
  summary text,
  summary_embedding vector,
  summarized_up_to_position bigint NOT NULL DEFAULT 0, ◄── NEW: watermark
  is_initialized boolean NOT NULL DEFAULT true,        ◄── NEW: fork lock
  data jsonb,
  created_at, is_primary
)
messages (
  id text PK, node_id FK, role, content, embedding vector,
  position bigint, timestamp, user_email
)
edges (id, from_node, to_node, meta jsonb)   ◄── canvas topology
external_files (id, node_id, file_name, file_type, processed, …)
file_chunks (id, file_id FK, chunk_index, chunk_text, embedding vector, metadata)
```

GIN index on `nodes.ancestor_ids` lets us answer "all descendants of X" via
`WHERE ancestor_ids @> ARRAY[$X]` without recursion.

---

## 3. Sequence: fork initialisation

The first message sent to a forked node triggers
`_init_fork_if_needed` ([chat.py:118](app/api/v1/endpoints/chat.py#L118)).
The whole DB write phase is one transaction guarded by a transaction-scoped
advisory lock keyed on the new `thread_id`.

```
Client                  /api/v1/chat                fork_thread()           Postgres
  │                          │                          │                      │
  │── POST /chat ───────────►│                          │                      │
  │   parentNodeId           │                          │                      │
  │   forkedFromMessageId    │                          │                      │
  │                          │── get_fork_inheritance_payload (read parent)──► │
  │                          │  (summary, recent N msgs, memory facts)         │
  │                          │◄────────────────────────────────────────────────│
  │                          │── LLM resummarize (if K msgs > buffer) ──►      │
  │                          │  (no DB)                                        │
  │                          │── fork_thread(...) ─────►│                      │
  │                          │                          │── BEGIN ──────────► │
  │                          │                          │── advisory_xact_lock(thread_id) ─► │
  │                          │                          │── SELECT is_initialized ──►        │
  │                          │                          │  (if true) ── COMMIT, return early │
  │                          │                          │── INSERT nodes (ancestor_ids = parent.ancestor_ids ║ parent_id, is_initialized=true)
  │                          │                          │── INSERT buffer messages ──►       │
  │                          │                          │── jsonb merge memoryFacts ──►      │
  │                          │                          │── COMMIT ────────────────────────►│
  │                          │── graph.invoke() ───────►│  (normal turn)                     │
  │◄── response ─────────────│                          │                                    │
```

If two requests race, the second one acquires the lock after the first
commits, sees `is_initialized=true`, and exits without redoing the work.
LLM cost may double on rare collisions; DB consistency is guaranteed.

---

## 4. Sequence: per-turn (after first message)

```
        ┌───────────────────────────────────────────────────────────────┐
        │  prepare                                                       │
        │   └─► get_thread_summary_state          (summary + watermark)  │
        │   └─► get_thread_messages_after(watermark, K)  (hydrate tail)  │
        │   └─► get_thread_ancestry_scopes        (1+1 queries, indexed) │
        │   └─► find_similar_by_message_id        (pgvector cosine)      │
        │   └─► get_related_file_context(context_node_ids?)              │
        │       ├ if list provided → filter by external_files.node_id    │
        │       └ if None          → fall back to canvas edges           │
        ├───────────────────────────────────────────────────────────────┤
        │  graph.invoke                                                  │
        │   └─► assistant node                                           │
        │   └─► summury_decision: count > MAX_MESSAGES_BEFORE_SUMMARY?   │
        │       ├ no  → END                                              │
        │       └ yes → summurize                                        │
        │              ├ LLM merge old msgs into summary                 │
        │              ├ compute new watermark =                         │
        │              │     MAX(messages.position) of summarized msgs   │
        │              ├ UPDATE nodes SET summary, summarized_up_to_position │
        │              │     = GREATEST(existing, new)                   │
        │              └ RemoveMessage(...) for state (DB rows untouched)│
        ├───────────────────────────────────────────────────────────────┤
        │  persist                                                       │
        │   └─► add_message(user)        (with embedding)               │
        │   └─► add_message(assistant)   (with embedding)               │
        └───────────────────────────────────────────────────────────────┘
```

Key property: messages with `position <= summarized_up_to_position` are kept
in the DB forever (so the UI keeps rendering them) but never replayed into
LangGraph state. Their content lives in `nodes.summary`.

---

## 5. External context (RAG over uploaded files)

External-context nodes hold uploaded documents. Each file is chunked and
each chunk gets an embedding (`file_chunks.embedding`).

The user **attaches** a context node to a chat node by drawing an edge in
the UI. That edge persists. But the *runtime* attachment set is also sent on
every chat request:

```
LLMRequest {
  ...,
  contextNodeIds: ["ctx_abc", "ctx_def"]
}
```

Backend behavior:

| Frontend sends         | Backend does                                           |
| ---------------------- | ------------------------------------------------------ |
| `contextNodeIds: [a,b]`| RAG over only the files belonging to nodes a and b     |
| `contextNodeIds: []`   | No external context this turn                          |
| field omitted          | Fall back to canvas edges (legacy behavior)            |

This lets the UI implement runtime connect/disconnect semantics that match
what the user sees on screen — without round-tripping through edge mutations.

---

## 6. Configuration knobs

| Setting                       | Default | Effect                                    |
| ----------------------------- | ------- | ----------------------------------------- |
| `MAX_MESSAGES_BEFORE_SUMMARY` | 10      | Triggers summurize when count exceeds this |
| `KEEP_LAST_MESSAGES`          | 6       | Tail kept in working memory after summary  |
| `FORK_BUFFER_MESSAGES`        | 2       | Verbatim msgs copied at fork (rest summarized) |
| `SIMILARITY_MIN_SCORE`        | 0.2     | Floor on cosine score for retrieved snippets |
| `SIMILAR_CONTEXT_LIMIT`       | 3       | Max retrieved past-message snippets        |
| `FILE_CONTEXT_LIMIT`          | 3       | Max retrieved file chunks                  |

All are read from `app.core.config.settings`.

---

## 7. What changed on 2026-04-26

- New columns on `nodes`: `ancestor_ids`, `summarized_up_to_position`, `is_initialized`.
- `get_thread_ancestry_scopes` no longer walks `parent_node_id` recursively;
  it reads the materialized `ancestor_ids` array (constant queries vs N-deep walk).
- `fork_thread` now runs in one transaction under an advisory lock; concurrent
  first-message fork inits are safe.
- `fork_thread` writes memory facts in the same transaction (no second trip).
- `update_thread_summary` accepts an optional watermark argument; the summarize
  node uses it. Watermark is monotonic (`GREATEST`).
- `_hydrate_recent_messages` now skips messages at or below the watermark, so
  re-hydration after a cold start doesn't re-feed already-summarized turns to
  the LLM. The `messages` table itself is never pruned by the summarizer.
- Chat API now accepts `contextNodeIds`; backend filters file-context retrieval
  by it. UI sends the live attachment set on every request.
- `core_arcitetcure.md` and `CONTEXT_TREE_REPORT.md` are obsolete pointers to
  this document.
