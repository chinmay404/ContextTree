"""
PostgresConversationStore — persistence layer for ContextTree's FastAPI backend.

Key design decisions:
- Module-level ThreadedConnectionPool: connections are reused across requests instead
  of opening a fresh TCP connection per DB call (previous behaviour).
- All methods acquire a connection via the _get_conn() context manager which commits
  on success, rolls back on any exception, and returns the connection to the pool.
- Vector similarity search uses `= ANY(%s::text[])` instead of the old broken
  IN-clause construction that failed for single-element sets.
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, List, Optional, Tuple

import numpy as np
import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from psycopg2 import pool as pg_pool
from psycopg2.extras import DictCursor, execute_values

from app.core.logger import logger

load_dotenv()

# ── Module-level connection pool ───────────────────────────────────────────────
_pool: Optional[pg_pool.ThreadedConnectionPool] = None


def _get_pool(db_url: str) -> pg_pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = pg_pool.ThreadedConnectionPool(minconn=2, maxconn=20, dsn=db_url)
        logger.info("PostgreSQL connection pool created (min=2, max=20)")
    return _pool


class PostgresConversationStore:
    def __init__(self, db_url: Optional[str] = None, embedding_dim: int = 768):
        self.db_url = db_url or os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL not found in environment.")
        self.embedding_dim = embedding_dim
        # Pool is opened lazily on first DB call — allows the app to start even
        # when the database is temporarily unavailable (e.g. Supabase paused).

    # ── Connection management ──────────────────────────────────────────────────

    @contextmanager
    def _get_conn(self):
        """
        Borrow a connection from the pool.
        Commits on clean exit, rolls back + re-raises on exception.
        """
        p = _get_pool(self.db_url)
        conn = p.getconn()
        try:
            register_vector(conn)
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            p.putconn(conn)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _resolve_user_email(self, cur, user_id: str) -> str:
        if not user_id:
            return "anonymous"
        if "@" in user_id:
            return user_id
        try:
            cur.execute("SELECT email FROM users WHERE id = %s", (user_id,))
            row = cur.fetchone()
            if row:
                return row[0]
        except Exception:
            pass
        return user_id

    def resolve_user_email(self, user_id: str) -> str:
        """Public wrapper: map a user id (uuid or email) to the user_email key."""
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                return self._resolve_user_email(cur, user_id)
        except Exception:
            return user_id

    def _get_default_canvas(self, cur, user_email: str) -> str:
        cur.execute("SELECT id FROM canvases WHERE user_email = %s LIMIT 1", (user_email,))
        row = cur.fetchone()
        if row:
            return row[0]
        new_id = str(uuid.uuid4())
        cur.execute(
            "INSERT INTO canvases (id, user_email, data) VALUES (%s, %s, '{}'::jsonb)",
            (new_id, user_email),
        )
        return new_id

    def _normalize_embedding(self, emb) -> Optional[list]:
        """Return a plain Python list of floats, or None if empty/invalid."""
        if emb is None or emb == {}:
            return None
        if isinstance(emb, np.ndarray):
            emb = emb.tolist()
        if isinstance(emb, list) and len(emb) > 0:
            return emb
        return None

    # ── Messages ───────────────────────────────────────────────────────────────

    def add_message(
        self,
        user_id: str,
        thread_id: str,
        message_id: str,
        role: str,
        text: str,
        embedding: List[float],
        summary: str = None,
        summarize_fn=None,
        embed_summary_fn=None,
        context_fn=None,
    ):
        now = datetime.utcnow()
        with self._get_conn() as conn:
            cur = conn.cursor()
            user_email = self._resolve_user_email(cur, user_id)

            summary_embedding = None
            if summary and callable(embed_summary_fn):
                try:
                    summary_embedding = self._normalize_embedding(embed_summary_fn(summary))
                except Exception:
                    pass

            cur.execute("SELECT id FROM nodes WHERE id = %s", (thread_id,))
            if cur.fetchone():
                if summary:
                    if summary_embedding:
                        cur.execute(
                            "UPDATE nodes SET summary = %s, summary_embedding = %s WHERE id = %s",
                            (summary, summary_embedding, thread_id),
                        )
                    else:
                        cur.execute("UPDATE nodes SET summary = %s WHERE id = %s", (summary, thread_id))
            else:
                canvas_id = self._get_default_canvas(cur, user_email)
                cur.execute(
                    """
                    INSERT INTO nodes (id, canvas_id, user_email, summary, summary_embedding, created_at, is_primary)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (thread_id, canvas_id, user_email, summary, summary_embedding, now, True),
                )

            emb = self._normalize_embedding(embedding)
            cur.execute(
                """
                INSERT INTO messages (id, node_id, role, content, embedding, timestamp, user_email, position)
                SELECT %s, %s, %s, %s, %s, %s, %s,
                       COALESCE((SELECT MAX(position) FROM messages WHERE node_id = %s), 0) + 1
                ON CONFLICT (id) DO NOTHING
                """,
                (message_id, thread_id, role, text, emb, now, user_email, thread_id),
            )

    def get_thread_messages(self, user_id: str, thread_id: str) -> List[dict]:
        try:
            with self._get_conn() as conn:
                cur = conn.cursor(cursor_factory=DictCursor)
                user_email = self._resolve_user_email(cur, user_id)
                cur.execute(
                    """
                    SELECT id as message_id, role, content as text, timestamp, position, embedding
                    FROM messages
                    WHERE node_id = %s AND user_email = %s
                    ORDER BY timestamp, position
                    """,
                    (thread_id, user_email),
                )
                return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            logger.error(f"Error getting thread messages: {e}")
            return []

    def get_thread_recent_messages(self, user_id: str, thread_id: str, limit: int) -> List[dict]:
        try:
            with self._get_conn() as conn:
                cur = conn.cursor(cursor_factory=DictCursor)
                user_email = self._resolve_user_email(cur, user_id)
                cur.execute(
                    """
                    SELECT id as message_id, role, content as text, timestamp, position, embedding
                    FROM messages
                    WHERE node_id = %s AND user_email = %s
                    ORDER BY timestamp DESC, position DESC
                    LIMIT %s
                    """,
                    (thread_id, user_email, limit),
                )
                return list(reversed([dict(r) for r in cur.fetchall()]))
        except Exception as e:
            logger.error(f"Error getting recent thread messages: {e}")
            return []

    def get_max_position_for_messages(self, thread_id: str, message_ids: List[str]) -> int:
        """
        Return the largest `position` among the given message ids on this
        thread. Used by the summarizer to advance the watermark to the highest
        position actually summarized — not just the index in LangGraph state,
        which can drift from the DB.
        """
        if not message_ids:
            return 0
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT COALESCE(MAX(position), 0)
                    FROM messages
                    WHERE node_id = %s AND id = ANY(%s::text[])
                    """,
                    (thread_id, list(message_ids)),
                )
                row = cur.fetchone()
                return int(row[0]) if row and row[0] is not None else 0
        except Exception as e:
            logger.error(f"Error computing max position for messages: {e}")
            return 0

    def get_thread_message_count(self, thread_id: str) -> int:
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM messages WHERE node_id = %s", (thread_id,))
                row = cur.fetchone()
                return int(row[0]) if row else 0
        except Exception:
            return 0

    def get_thread_fork_metadata(self, thread_id: str) -> Optional[dict]:
        try:
            with self._get_conn() as conn:
                cur = conn.cursor(cursor_factory=DictCursor)
                cur.execute(
                    """
                    SELECT
                        parent_node_id,
                        forked_from_message_id,
                        is_initialized,
                        ancestor_ids
                    FROM nodes
                    WHERE id = %s
                    """,
                    (thread_id,),
                )
                row = cur.fetchone()
                meta = dict(row) if row else None

                # Racing canvas saves have been observed nulling the lineage
                # columns. The edge the canvas draws is ground truth — fall
                # back to it so forks can ALWAYS find their parent.
                if meta is None or not meta.get("parent_node_id"):
                    cur.execute(
                        "SELECT from_node FROM edges WHERE to_node = %s LIMIT 1",
                        (thread_id,),
                    )
                    edge = cur.fetchone()
                    if edge and edge[0]:
                        meta = meta or {
                            "forked_from_message_id": None,
                            "is_initialized": False,
                            "ancestor_ids": [],
                        }
                        meta["parent_node_id"] = edge[0]
                        logger.info(
                            "Fork metadata repaired from edges: %s -> parent %s",
                            thread_id,
                            edge[0],
                        )
                return meta
        except Exception as e:
            logger.error(f"Error fetching fork metadata: {e}")
            return None

    def get_thread_created_at(self, thread_id: str) -> Optional[datetime]:
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT created_at FROM nodes WHERE id = %s", (thread_id,))
                row = cur.fetchone()
                return row[0] if row else None
        except Exception:
            return None

    def get_message_by_id(self, message_id: str) -> Optional[dict]:
        try:
            with self._get_conn() as conn:
                cur = conn.cursor(cursor_factory=DictCursor)
                cur.execute(
                    """
                    SELECT id as message_id, node_id as thread_id, role, content as text, timestamp, embedding
                    FROM messages WHERE id = %s
                    """,
                    (message_id,),
                )
                row = cur.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error getting message by id: {e}")
            return None

    def get_recent_messages(self, limit: int = 10) -> List[dict]:
        try:
            with self._get_conn() as conn:
                cur = conn.cursor(cursor_factory=DictCursor)
                cur.execute(
                    """
                    SELECT id as message_id, node_id as thread_id, role, content as text, timestamp
                    FROM messages ORDER BY timestamp DESC LIMIT %s
                    """,
                    (limit,),
                )
                return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            logger.error(f"Error getting recent messages: {e}")
            return []

    def prune_thread_messages(self, thread_id: str, keep_last_n: int) -> int:
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT id FROM messages WHERE node_id = %s ORDER BY timestamp, position",
                    (thread_id,),
                )
                ids = [row[0] for row in cur.fetchall()]
                if keep_last_n <= 0 or len(ids) <= keep_last_n:
                    return 0
                ids_to_delete = ids[:-keep_last_n]
                cur.execute(
                    "DELETE FROM messages WHERE node_id = %s AND id = ANY(%s)",
                    (thread_id, ids_to_delete),
                )
                return cur.rowcount
        except Exception as e:
            logger.error(f"Error pruning thread messages: {e}")
            return 0

    # ── Summaries ──────────────────────────────────────────────────────────────

    def get_thread_summary(self, user_id: str, thread_id: str) -> str:
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                user_email = self._resolve_user_email(cur, user_id)
                cur.execute(
                    "SELECT summary FROM nodes WHERE id = %s AND user_email = %s",
                    (thread_id, user_email),
                )
                row = cur.fetchone()
                return row[0] if row else ""
        except Exception:
            return ""

    def update_thread_summary(
        self,
        user_id: str,
        thread_id: str,
        summary: str,
        summarized_up_to_position: Optional[int] = None,
    ) -> bool:
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                user_email = self._resolve_user_email(cur, user_id)
                if summarized_up_to_position is None:
                    cur.execute(
                        "UPDATE nodes SET summary = %s WHERE id = %s AND user_email = %s",
                        (summary, thread_id, user_email),
                    )
                else:
                    # Watermark monotonically advances: never move it backwards even
                    # if a stale summarize call arrives out of order.
                    cur.execute(
                        """
                        UPDATE nodes
                        SET summary = %s,
                            summarized_up_to_position = GREATEST(summarized_up_to_position, %s)
                        WHERE id = %s AND user_email = %s
                        """,
                        (summary, int(summarized_up_to_position), thread_id, user_email),
                    )
                return cur.rowcount > 0
        except Exception:
            return False

    def get_thread_summary_state(self, user_id: str, thread_id: str) -> dict:
        """Single-trip read of the bits hydration needs: summary text + watermark."""
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                user_email = self._resolve_user_email(cur, user_id)
                cur.execute(
                    "SELECT summary, summarized_up_to_position FROM nodes WHERE id = %s AND user_email = %s",
                    (thread_id, user_email),
                )
                row = cur.fetchone()
                if not row:
                    return {"summary": "", "summarized_up_to_position": 0}
                return {
                    "summary": row[0] or "",
                    "summarized_up_to_position": int(row[1] or 0),
                }
        except Exception:
            return {"summary": "", "summarized_up_to_position": 0}

    def get_thread_messages_after(
        self,
        user_id: str,
        thread_id: str,
        after_position: int,
        limit: int,
    ) -> List[dict]:
        """
        Return the most recent `limit` messages whose `position` is strictly
        greater than `after_position`. Used to hydrate LangGraph state with only
        the un-summarized tail of the conversation. Messages at or below the
        watermark stay in the DB (the UI still needs them) but are not loaded
        back into runtime state — their content lives in nodes.summary.
        """
        try:
            with self._get_conn() as conn:
                cur = conn.cursor(cursor_factory=DictCursor)
                user_email = self._resolve_user_email(cur, user_id)
                cur.execute(
                    """
                    SELECT id as message_id, role, content as text, timestamp, position, embedding
                    FROM messages
                    WHERE node_id = %s AND user_email = %s AND position > %s
                    ORDER BY timestamp DESC, position DESC
                    LIMIT %s
                    """,
                    (thread_id, user_email, int(after_position or 0), limit),
                )
                return list(reversed([dict(r) for r in cur.fetchall()]))
        except Exception as e:
            logger.error(f"Error getting messages after position {after_position}: {e}")
            return []

    def get_thread_memory_facts(self, user_id: str, thread_id: str) -> dict[str, Any]:
        try:
            with self._get_conn() as conn:
                cur = conn.cursor(cursor_factory=DictCursor)
                user_email = self._resolve_user_email(cur, user_id)
                cur.execute(
                    "SELECT data FROM nodes WHERE id = %s AND user_email = %s",
                    (thread_id, user_email),
                )
                row = cur.fetchone()
                data = row["data"] if row else {}
                if not isinstance(data, dict):
                    return {}
                if isinstance(data.get("memoryFacts"), dict):
                    return data.get("memoryFacts") or {}
                nested = data.get("data")
                if isinstance(nested, dict) and isinstance(nested.get("memoryFacts"), dict):
                    return nested.get("memoryFacts") or {}
                return {}
        except Exception:
            return {}

    def update_thread_memory_facts(self, user_id: str, thread_id: str, memory_facts: dict[str, Any]) -> bool:
        try:
            payload = json.dumps(memory_facts or {})
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    UPDATE nodes
                    SET data = jsonb_set(
                                jsonb_set(coalesce(data, '{}'::jsonb),
                                    '{memoryFacts}', %s::jsonb, true),
                                '{data,memoryFacts}', %s::jsonb, true)
                    WHERE id = %s
                    """,
                    (payload, payload, thread_id),
                )
                return cur.rowcount > 0
        except Exception:
            return False

    def get_fork_inheritance_payload(
        self,
        user_id: str,
        thread_id: str,
        message_id: str,
        recent_limit: int,
    ) -> dict[str, Any]:
        summary = self.get_thread_summary(user_id, thread_id)
        memory_facts = self.get_thread_memory_facts(user_id, thread_id)
        msgs = self.get_thread_messages(user_id, thread_id)
        ids = [m["message_id"] for m in msgs]
        resolved = self._resolve_message_id(message_id, ids)
        if resolved and resolved != message_id:
            logger.info(
                "Resolved fork message id for thread %s: requested=%s resolved=%s total_candidates=%s",
                thread_id,
                message_id,
                resolved,
                len(ids),
            )

        if not resolved:
            logger.warning(
                "Could not resolve fork message id for thread %s: requested=%s total_candidates=%s",
                thread_id,
                message_id,
                len(ids),
            )
            return {
                "summary": summary,
                "memory_facts": memory_facts,
                "messages": self.get_thread_recent_messages(user_id, thread_id, recent_limit),
                "mode": "latest-parent-state-fallback",
                "resolved_message_id": None,
                "skip_resummarize": True,
            }

        target_message = next((message for message in msgs if message["message_id"] == resolved), None)
        created_at = self.get_thread_created_at(thread_id)
        target_timestamp = target_message.get("timestamp") if isinstance(target_message, dict) else None
        is_inherited_buffer_message = bool(
            created_at and target_timestamp and isinstance(target_timestamp, datetime) and target_timestamp < created_at
        )
        if is_inherited_buffer_message:
            return {
                "summary": summary,
                "memory_facts": memory_facts,
                "messages": self.get_thread_recent_messages(user_id, thread_id, recent_limit),
                "mode": "latest-parent-state-from-buffer",
                "resolved_message_id": resolved,
                "skip_resummarize": True,
            }

        scoped_summary = summary if ids and resolved == ids[-1] else ""
        target_msgs: List[dict] = []
        for message in msgs:
            target_msgs.append(message)
            if message["message_id"] == resolved:
                break

        return {
            "summary": scoped_summary,
            "memory_facts": memory_facts if ids and resolved == ids[-1] else {},
            "messages": target_msgs,
            "mode": "scoped-fork-point",
            "resolved_message_id": resolved,
            "skip_resummarize": False,
        }

    # ── Thread / ancestry ──────────────────────────────────────────────────────

    def increment_daily_quota(self, user_key: str) -> int:
        """
        Atomically increment today's message count for a verified user and
        return the new count. One statement — safe across workers.
        """
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO quotas (user_key, day, msg_count)
                VALUES (%s, CURRENT_DATE, 1)
                ON CONFLICT (user_key, day)
                DO UPDATE SET msg_count = quotas.msg_count + 1,
                              updated_at = now()
                RETURNING msg_count
                """,
                (user_key,),
            )
            return int(cur.fetchone()[0])

    def get_thread_owner(self, thread_id: str) -> Optional[str]:
        """
        Return the owning user_email of a thread/node, or None if the thread
        does not exist. Used by the API layer's ownership gate: a request for
        a thread owned by someone else is rejected before any work happens.
        """
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT user_email FROM nodes WHERE id = %s", (thread_id,))
                row = cur.fetchone()
                return row[0] if row else None
        except Exception as e:
            logger.error(f"Error resolving thread owner: {e}")
            return None

    def thread_exists(self, thread_id: str) -> bool:
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT 1 FROM nodes WHERE id = %s", (thread_id,))
                return cur.fetchone() is not None
        except Exception:
            return False

    def get_thread_ancestry_scopes(self, thread_id: str) -> List[Tuple[str, Optional[str]]]:
        """
        Returns [(thread_id, limit_message_id)] from current thread to root.

        The current thread has no limit. Each ancestor is scoped by the child
        node's forked_from_message_id so retrieval cannot see messages that were
        added to the parent after the branch was created.

        Uses the materialized `ancestor_ids` column populated at fork time, so
        this is a fixed two-query operation regardless of tree depth — no
        recursive parent_node_id walk.
        """
        try:
            with self._get_conn() as conn:
                cur = conn.cursor(cursor_factory=DictCursor)

                cur.execute(
                    "SELECT ancestor_ids FROM nodes WHERE id = %s",
                    (thread_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return [(thread_id, None)]

                ancestor_ids = list(row["ancestor_ids"] or [])
                # Order in `ancestor_ids` is root → ... → immediate-parent.
                # Caller expects current → root, so reverse and prepend self.
                if not ancestor_ids:
                    return [(thread_id, None)]

                cur.execute(
                    "SELECT id, forked_from_message_id FROM nodes WHERE id = ANY(%s::text[])",
                    (ancestor_ids,),
                )
                fork_points = {r["id"]: r["forked_from_message_id"] for r in cur.fetchall()}

                # Build (node, limit_msg_id) pairs current → root.
                # `limit_message_id` for ancestor X is the fork point of X's
                # immediate child on this lineage — i.e. the next node walking
                # toward `thread_id`.
                lineage = list(reversed(ancestor_ids)) + []  # immediate-parent → root
                ancestry: List[Tuple[str, Optional[str]]] = [(thread_id, None)]
                # Limit on the immediate parent is the fork point recorded on
                # `thread_id` itself (already represents "where we branched off
                # the parent").
                cur.execute(
                    "SELECT forked_from_message_id FROM nodes WHERE id = %s",
                    (thread_id,),
                )
                self_row = cur.fetchone()
                child_fork_point: Optional[str] = self_row["forked_from_message_id"] if self_row else None

                for ancestor_id in lineage:
                    ancestry.append((ancestor_id, child_fork_point))
                    child_fork_point = fork_points.get(ancestor_id) or None

                return ancestry
        except Exception as e:
            logger.error(f"Error getting thread ancestry scopes: {e}")
            return [(thread_id, None)]

    def get_thread_ancestry(self, thread_id: str) -> List[str]:
        """Compatibility helper returning only the thread IDs."""
        return [thread for thread, _ in self.get_thread_ancestry_scopes(thread_id)]

    def get_messages_until(
        self, user_id: str, thread_id: str, message_id: str
    ) -> Tuple[str, List[dict]]:
        summary = self.get_thread_summary(user_id, thread_id)
        msgs = self.get_thread_messages(user_id, thread_id)
        ids = [m["message_id"] for m in msgs]
        resolved = self._resolve_message_id(message_id, ids)
        if resolved and resolved != message_id:
            logger.info(
                "Resolved fork message id for thread %s: requested=%s resolved=%s total_candidates=%s",
                thread_id,
                message_id,
                resolved,
                len(ids),
            )
        if not resolved:
            logger.warning(
                "Could not resolve fork message id for thread %s: requested=%s total_candidates=%s",
                thread_id,
                message_id,
                len(ids),
            )
        scoped_summary = summary if resolved and ids and resolved == ids[-1] else ""
        target_msgs: List[dict] = []
        for m in msgs:
            target_msgs.append(m)
            if m["message_id"] == resolved:
                return scoped_summary, target_msgs
        return summary, []

    # ── Similarity search ──────────────────────────────────────────────────────

    def find_similar_by_message_id(
        self,
        user_id: str,
        thread_queries: List[Tuple[str, Optional[str]]],
        query_embeddings: List[float],
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> List[dict]:
        """
        Vector similarity search scoped to the given thread ancestry.
        Uses `= ANY(%s::text[])` to avoid the old broken IN-clause construction.
        """
        try:
            with self._get_conn() as conn:
                cur = conn.cursor(cursor_factory=DictCursor)

                is_global = not thread_queries
                allowed_ids: List[str] = []

                if not is_global:
                    for thread_id, limit_msg_id in thread_queries:
                        msgs = self.get_thread_messages(user_id, thread_id)
                        ids = [m["message_id"] for m in msgs]
                        if limit_msg_id:
                            resolved = self._resolve_message_id(limit_msg_id, ids)
                            if resolved and resolved in ids:
                                idx = ids.index(resolved)
                                allowed_ids.extend(ids[: idx + 1])
                        else:
                            allowed_ids.extend(ids)

                    if not allowed_ids:
                        return []
                    allowed_ids = list(dict.fromkeys(allowed_ids))

                if is_global:
                    user_email = self._resolve_user_email(cur, user_id)
                    cur.execute(
                        """
                        SELECT id as message_id, role, content as text,
                               (1 - (embedding <=> %s::vector)) as score
                        FROM messages
                        WHERE user_email = %s
                          AND embedding IS NOT NULL
                          AND (1 - (embedding <=> %s::vector)) >= %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (query_embeddings, user_email, query_embeddings, min_score, query_embeddings, top_k),
                    )
                else:
                    # Use = ANY(%s::text[]) — correct for any list size
                    cur.execute(
                        """
                        SELECT id as message_id, role, content as text,
                               (1 - (embedding <=> %s::vector)) as score
                        FROM messages
                        WHERE id = ANY(%s::text[])
                          AND embedding IS NOT NULL
                          AND (1 - (embedding <=> %s::vector)) >= %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (query_embeddings, allowed_ids, query_embeddings, min_score, query_embeddings, top_k),
                    )

                return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            logger.error(f"Error in find_similar: {e}")
            return []

    # ── Fork ───────────────────────────────────────────────────────────────────

    def fork_thread(
        self,
        user_id: str,
        source_thread_id: str,
        new_thread_id: str,
        fork_at_message_id: str,
        summary: str = None,
        summary_embedding: List[float] = None,
        initial_messages: List[dict] = [],
        memory_facts: Optional[dict] = None,
    ) -> bool:
        """
        Atomically initialize a forked node.

        All writes happen under a transaction-scoped advisory lock keyed on
        `new_thread_id`, so concurrent first-message requests for the same fork
        serialize and the second one observes `is_initialized=true` and exits
        without redoing the work.

        Memory facts are written here (not in a separate trip) so the whole
        fork init is one atomic unit — we never end up with a node row that
        has buffer messages but no facts, or vice versa.
        """
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                now = datetime.utcnow()
                user_email = self._resolve_user_email(cur, user_id)

                # Serialize concurrent forks for the same target thread.
                cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s))",
                    (new_thread_id,),
                )

                # Idempotency: if another request already finished this fork,
                # bail out without touching anything.
                cur.execute(
                    """
                    SELECT is_initialized, ancestor_ids
                    FROM nodes
                    WHERE id = %s
                    """,
                    (new_thread_id,),
                )
                existing = cur.fetchone()
                if existing and bool(existing[0]) and list(existing[1] or []):
                    logger.info(
                        "fork_thread: %s already initialized; skipping redo",
                        new_thread_id,
                    )
                    return True
                if existing and bool(existing[0]) and not list(existing[1] or []):
                    logger.warning(
                        "fork_thread: %s was marked initialized without ancestry; repairing fork inheritance",
                        new_thread_id,
                    )

                cur.execute(
                    "SELECT canvas_id, ancestor_ids FROM nodes WHERE id = %s",
                    (source_thread_id,),
                )
                source = cur.fetchone()
                if not source:
                    return False
                canvas_id = source[0]
                parent_ancestors = list(source[1] or [])
                child_ancestors = parent_ancestors + [source_thread_id]

                se = self._normalize_embedding(summary_embedding)
                facts_payload = json.dumps(memory_facts or {})

                if existing:
                    # Row exists but is_initialized=false (a previous attempt
                    # crashed before completing). Take it over.
                    cur.execute(
                        """
                        UPDATE nodes SET
                            canvas_id = %s,
                            user_email = %s,
                            parent_node_id = %s,
                            forked_from_message_id = %s,
                            summary = %s,
                            summary_embedding = %s,
                            ancestor_ids = %s,
                            is_initialized = true,
                            data = jsonb_set(
                                jsonb_set(coalesce(data, '{}'::jsonb),
                                    '{memoryFacts}', %s::jsonb, true),
                                '{data,memoryFacts}', %s::jsonb, true)
                        WHERE id = %s
                        """,
                        (
                            canvas_id, user_email, source_thread_id, fork_at_message_id,
                            summary, se, child_ancestors,
                            facts_payload, facts_payload,
                            new_thread_id,
                        ),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO nodes (
                            id, canvas_id, user_email, parent_node_id,
                            forked_from_message_id, summary, summary_embedding,
                            ancestor_ids, is_initialized, created_at, is_primary,
                            data
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, true, %s, false,
                                  jsonb_build_object('memoryFacts', %s::jsonb,
                                                     'data', jsonb_build_object('memoryFacts', %s::jsonb)))
                        """,
                        (
                            new_thread_id, canvas_id, user_email, source_thread_id,
                            fork_at_message_id, summary, se, child_ancestors, now,
                            facts_payload, facts_payload,
                        ),
                    )

                if initial_messages:
                    self._insert_buffer_messages(cur, new_thread_id, user_email, initial_messages, now)
                return True
        except Exception as e:
            logger.error(f"Error forking thread: {e}")
            return False

    def _insert_buffer_messages(self, cur, node_id: str, user_email: str, messages: list, now):
        """Insert copied buffer messages with fresh IDs (messages are independent in new branch)."""
        for msg in messages:
            emb = self._normalize_embedding(msg.get("embedding"))
            content = msg.get("text", "") or msg.get("content", "")
            if isinstance(content, (dict, list)):
                try:
                    content = json.dumps(content)
                except Exception:
                    content = str(content)
            elif not isinstance(content, str):
                content = str(content)

            timestamp = msg.get("timestamp", now)
            if not isinstance(timestamp, datetime):
                timestamp = now

            cur.execute(
                """
                INSERT INTO messages (id, node_id, role, content, embedding, timestamp, user_email, position)
                SELECT %s, %s, %s, %s, %s, %s, %s,
                       COALESCE((SELECT MAX(position) FROM messages WHERE node_id = %s), 0) + 1
                """,
                (
                    str(uuid.uuid4()),
                    node_id,
                    msg.get("role"),
                    content,
                    emb,
                    timestamp,
                    user_email,
                    node_id,
                ),
            )

    # ── File context ───────────────────────────────────────────────────────────

    def get_file_binary(self, file_id: str) -> Optional[dict]:
        try:
            with self._get_conn() as conn:
                cur = conn.cursor(cursor_factory=DictCursor)
                cur.execute(
                    "SELECT id, file_name, mime_type, data FROM external_files WHERE id = %s",
                    (file_id,),
                )
                row = cur.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error fetching file binary: {e}")
            raise

    def save_file_chunks(
        self,
        file_id: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
    ):
        if metadatas is None:
            metadatas = [{"source": "unknown"}] * len(chunks)

        data_list = [
            (file_id, idx, text, emb, json.dumps(meta))
            for idx, (text, emb, meta) in enumerate(zip(chunks, embeddings, metadatas))
        ]

        with self._get_conn() as conn:
            cur = conn.cursor()
            execute_values(
                cur,
                "INSERT INTO file_chunks (file_id, chunk_index, chunk_text, embedding, metadata) VALUES %s",
                data_list,
            )
            cur.execute(
                "UPDATE external_files SET processed = true, updated_at = NOW() WHERE id = %s",
                (file_id,),
            )

    def get_related_file_context(
        self,
        node_id: str,
        query_embedding: List[float],
        limit: int = 5,
        context_node_ids: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Find nearest file chunks for the given chat node.

        When `context_node_ids` is provided, those IDs are the source of truth
        for which external-context nodes are currently attached (the request
        payload says so, regardless of what the canvas edges say). When None,
        falls back to the persisted `edges` graph for backward compatibility.

        Passing an empty list explicitly disables RAG over external context for
        this turn.
        """
        if not query_embedding:
            return []
        if context_node_ids is not None and len(context_node_ids) == 0:
            return []
        try:
            with self._get_conn() as conn:
                cur = conn.cursor(cursor_factory=DictCursor)
                if context_node_ids is None:
                    # Legacy path: derive attached files from the canvas edges.
                    cur.execute(
                        """
                        WITH connected_files AS (
                            SELECT n.id AS file_node_id
                            FROM nodes n
                            JOIN edges e ON (e.from_node = n.id OR e.to_node = n.id)
                            WHERE (e.from_node = %s OR e.to_node = %s)
                              AND n.id != %s
                              AND n.type = 'externalContext'
                        )
                        SELECT
                            fc.chunk_text,
                            fc.metadata,
                            fc.chunk_index,
                            (fc.embedding <=> %s::vector) AS distance,
                            ef.file_name,
                            ef.file_type
                        FROM file_chunks fc
                        JOIN external_files ef ON fc.file_id = ef.id
                        JOIN connected_files cf ON ef.node_id = cf.file_node_id
                        ORDER BY distance ASC
                        LIMIT %s
                        """,
                        (node_id, node_id, node_id, np.array(query_embedding), limit),
                    )
                else:
                    # Runtime path: trust the request's active set.
                    cur.execute(
                        """
                        SELECT
                            fc.chunk_text,
                            fc.metadata,
                            fc.chunk_index,
                            (fc.embedding <=> %s::vector) AS distance,
                            ef.file_name,
                            ef.file_type
                        FROM file_chunks fc
                        JOIN external_files ef ON fc.file_id = ef.id
                        WHERE ef.node_id = ANY(%s::text[])
                        ORDER BY distance ASC
                        LIMIT %s
                        """,
                        (np.array(query_embedding), list(context_node_ids), limit),
                    )
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching related file context: {e}")
            return []

    # ── Node content updates ───────────────────────────────────────────────────

    def update_node_data_content(self, node_id: str, content: str) -> bool:
        try:
            text_value = content or ""
            contract_value = content if content else "(No text content extracted)"
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    UPDATE nodes
                    SET data = jsonb_set(
                                jsonb_set(
                                    jsonb_set(
                                        jsonb_set(
                                            jsonb_set(
                                                jsonb_set(
                                                    jsonb_set(
                                                        jsonb_set(coalesce(data, '{}'::jsonb),
                                                            '{data,content}', to_jsonb(%s::text), true),
                                                        '{data,loading}', 'false'::jsonb, true),
                                                    '{data,error}', 'null'::jsonb, true),
                                                '{content}', to_jsonb(%s::text), true),
                                            '{loading}', 'false'::jsonb, true),
                                        '{error}', 'null'::jsonb, true),
                                    '{contextContract}', to_jsonb(%s::text), true),
                                '{data,contextContract}', to_jsonb(%s::text), true)
                    WHERE id = %s
                    """,
                    (text_value, text_value, contract_value, contract_value, node_id),
                )
                return cur.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating node data content: {e}")
            return False

    def update_node_processing_error(self, node_id: str, message: str) -> bool:
        try:
            error_value = message or "Failed to process file"
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    UPDATE nodes
                    SET data = jsonb_set(
                                jsonb_set(
                                    jsonb_set(
                                        jsonb_set(
                                            jsonb_set(
                                                jsonb_set(coalesce(data, '{}'::jsonb),
                                                    '{data,error}', to_jsonb(%s::text), true),
                                                '{data,loading}', 'false'::jsonb, true),
                                            '{error}', to_jsonb(%s::text), true),
                                        '{loading}', 'false'::jsonb, true),
                                    '{contextContract}', to_jsonb(%s::text), true),
                                '{data,contextContract}', to_jsonb(%s::text), true)
                    WHERE id = %s
                    """,
                    (error_value, error_value, error_value, error_value, node_id),
                )
                return cur.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating node processing error: {e}")
            return False

    # ── ID resolution ──────────────────────────────────────────────────────────

    def _resolve_message_id(self, target_id: str, candidate_ids: List[str]) -> Optional[str]:
        """
        Fuzzy match to handle frontend suffix conventions (_u, _ai, -a, -u)
        vs backend stored IDs.
        """
        if target_id in candidate_ids:
            return target_id

        assistant_suffixes = ("_assistant", "-assistant", "_ai", "_a", "-a")
        user_suffixes = ("_user", "-user", "_u", "-u")

        def normalize_and_role(mid: str) -> Tuple[str, str]:
            value = str(mid or "")
            role = "user"

            changed = True
            while changed and value:
                changed = False
                for suffix in assistant_suffixes:
                    if value.endswith(suffix):
                        if role == "user":
                            role = "assistant"
                        value = value[: -len(suffix)]
                        changed = True
                        break
                if changed:
                    continue
                for suffix in user_suffixes:
                    if value.endswith(suffix):
                        value = value[: -len(suffix)]
                        changed = True
                        break

            return value, role

        req_base, req_role = normalize_and_role(target_id)

        for mid in candidate_ids:
            if not isinstance(mid, str):
                continue
            cand_base, cand_role = normalize_and_role(mid)
            if cand_base == req_base and cand_role == req_role:
                return mid

        # Looser fallback
        for mid in candidate_ids:
            if not isinstance(mid, str):
                continue
            cand_base, _ = normalize_and_role(mid)
            if cand_base == req_base:
                return mid

        return None
