import os
import json
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Any
import psycopg2
from psycopg2.extras import Json, DictCursor
from psycopg2 import pool
from pgvector.psycopg2 import register_vector
import numpy as np
from dotenv import load_dotenv
from app.core.logger import logger

load_dotenv()

class PostgresConversationStore:
    def __init__(
        self,
        db_url: Optional[str] = None,
        embedding_dim: int = 768
    ):
        self.db_url = db_url or os.getenv("DATABASE_URL")
        # Ensure we use the pooled 'postgres' database if generic
        if not self.db_url:
             raise ValueError("DATABASE_URL not found.")
             
        self.embedding_dim = embedding_dim
        # We assume tables exist now (nodes, messages, etc.)

    def _get_conn(self):
        conn = psycopg2.connect(self.db_url)
        return conn
        
    def _resolve_user_email(self, cur, user_id: str) -> str:
        """
        Resolves user_id (which might be UUID or email) to an email address.
        Returns the resolved email or defaults to user_id if lookup fails.
        """
        if not user_id:
            return "anonymous"
        # If it looks like an email, return it
        if "@" in user_id:
            return user_id
            
        # Try to look up by ID
        try:
            cur.execute("SELECT email FROM users WHERE id = %s", (user_id,))
            row = cur.fetchone()
            if row:
                return row[0]
        except Exception:
            pass # Ignore errors (e.g. invalid UUID syntax)
            
        return user_id

    def _get_default_canvas(self, cur, user_email: str) -> str:
        """
        Finds a canvas for the user to attach new nodes to.
        Prioritizes existing canvases. If none, creates a 'General' canvas.
        """
        cur.execute("SELECT id FROM canvases WHERE user_email = %s LIMIT 1", (user_email,))
        row = cur.fetchone()
        if row:
            return row[0]
            
        # Create default canvas
        import uuid
        new_id = str(uuid.uuid4())
        cur.execute("INSERT INTO canvases (id, user_email, data) VALUES (%s, %s, '{}')", (new_id, user_email))
        return new_id

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
        context_fn=None
    ):
        conn = self._get_conn()
        try:
            register_vector(conn)
            cur = conn.cursor()
            
            user_email = self._resolve_user_email(cur, user_id)
            now = datetime.utcnow()

            # 1. Ensure Node (Thread) Creation/Update
            cur.execute("SELECT id FROM nodes WHERE id = %s", (thread_id,))
            exists = cur.fetchone()
            
            summary_embedding = None
            if summary and embed_summary_fn and callable(embed_summary_fn):
                try:
                    summary_embedding = embed_summary_fn(summary) or []
                except Exception:
                    summary_embedding = []

            if exists:
                if summary:
                    if summary_embedding:
                         cur.execute("UPDATE nodes SET summary = %s, summary_embedding = %s WHERE id = %s", (summary, summary_embedding, thread_id))
                    else:
                         cur.execute("UPDATE nodes SET summary = %s WHERE id = %s", (summary, thread_id))
            else:
                # Create Node. We need a canvas_id.
                canvas_id = self._get_default_canvas(cur, user_email)
                cur.execute("""
                    INSERT INTO nodes (id, canvas_id, user_email, summary, summary_embedding, created_at, is_primary)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (thread_id, canvas_id, user_email, summary, summary_embedding, now, True))
            
            # 2. Insert Message
            # 'messages' table: id, node_id, role, content, (user_email from schema?), embedding
            cur.execute("""
                INSERT INTO messages (id, node_id, role, content, embedding, timestamp, user_email)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (message_id, thread_id, role, text, embedding, now, user_email))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error adding message: {e}")
            raise e
        finally:
            conn.close()

    def get_thread_messages(self, user_id: str, thread_id: str) -> List[dict]:
        """
        Retrieves messages for a node.
        In the 'Arc' architecture, nodes are independent. 
        This method retrieves ONLY the messages belonging to the specific thread_id,
        without traversing parents.
        """
        conn = self._get_conn()
        try:
            register_vector(conn)
            cur = conn.cursor(cursor_factory=DictCursor)
            
            query = """
                SELECT id as message_id, role, content as text, timestamp, position, embedding
                FROM messages 
                WHERE node_id = %s
                ORDER BY timestamp, position
            """
            cur.execute(query, (thread_id,))
            msgs = [dict(r) for r in cur.fetchall()]
            
            return msgs
            
        except Exception as e:
            logger.error(f"Error getting thread messages: {e}")
            return []
        finally:
            conn.close()

    def get_thread_ancestry(self, thread_id: str) -> List[str]:
        """
        Returns a list of node_ids from current thread up to the root, 
        following parent_node_id links.
        """
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            ancestry = []
            queue = [thread_id]
            
            # Simple recursive CTE replacement or loop
            # Using loop for safety against cycles if any
            visited = set()
            curr = thread_id
            
            while curr and curr not in visited:
                visited.add(curr)
                ancestry.append(curr)
                cur.execute("SELECT parent_node_id FROM nodes WHERE id = %s", (curr,))
                row = cur.fetchone()
                if row and row[0]:
                    curr = row[0]
                else:
                    curr = None
            
            return ancestry
        except Exception as e:
            logger.error(f"Error getting thread ancestry: {e}")
            return [thread_id]
        finally:
            conn.close()

    def get_thread_recent_messages(self, user_id: str, thread_id: str, limit: int) -> List[dict]:
        """
        Retrieves the most recent `limit` messages for a node, ordered oldest->newest.
        """
        conn = self._get_conn()
        try:
            register_vector(conn)
            cur = conn.cursor(cursor_factory=DictCursor)

            query = """
                SELECT id as message_id, role, content as text, timestamp, position, embedding
                FROM messages
                WHERE node_id = %s
                ORDER BY timestamp DESC, position DESC
                LIMIT %s
            """
            cur.execute(query, (thread_id, limit))
            msgs = [dict(r) for r in cur.fetchall()]
            return list(reversed(msgs))
        except Exception as e:
            logger.error(f"Error getting recent thread messages: {e}")
            return []
        finally:
            conn.close()

    def prune_thread_messages(self, thread_id: str, keep_last_n: int) -> int:
        """
        Deletes older messages in a thread, keeping only the last `keep_last_n`.
        Returns the number of deleted messages.
        """
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id
                FROM messages
                WHERE node_id = %s
                ORDER BY timestamp, position
                """,
                (thread_id,)
            )
            ids = [row[0] for row in cur.fetchall()]

            if keep_last_n <= 0 or len(ids) <= keep_last_n:
                return 0

            ids_to_delete = ids[:-keep_last_n]
            cur.execute(
                "DELETE FROM messages WHERE node_id = %s AND id = ANY(%s)",
                (thread_id, ids_to_delete),
            )
            deleted = cur.rowcount
            conn.commit()
            return deleted
        except Exception as e:
            conn.rollback()
            logger.error(f"Error pruning thread messages: {e}")
            return 0
        finally:
            conn.close()

    def find_similar_by_message_id(
        self,
        user_id: str,
        thread_queries: List[Tuple[str, Optional[str]]],
        query_embeddings: List[float],
        top_k: int = 3
    ) -> List[dict]:
        conn = self._get_conn()
        try:
            register_vector(conn)
            cur = conn.cursor(cursor_factory=DictCursor)
            
            # Since threads are trees, "messages in a thread" is complex.
            # But the user query semantics usually mean "messages belonging to this conversation view".
            # We must resolve the "accessible message IDs" for each query (thread_id, msg_id).
            
            allowed_message_ids = set()
            is_global_search = not thread_queries

            if not is_global_search:
                for thread_id, limit_msg_id in thread_queries:
                    # We reuse get_thread_messages logic to get the list of relevant messages
                    msgs = self.get_thread_messages(user_id, thread_id)
                    ids = [m['message_id'] for m in msgs]
                    
                    if limit_msg_id:
                        resolved_limit_id = self._resolve_message_id(limit_msg_id, ids)
                        if resolved_limit_id and resolved_limit_id in ids:
                            idx = ids.index(resolved_limit_id)
                            allowed_message_ids.update(ids[:idx])
                    else:
                        allowed_message_ids.update(ids)
                
                if not allowed_message_ids:
                    return []

            # Vector Search
            if is_global_search:
                user_email_resolved = self._resolve_user_email(cur, user_id)
                query = f"""
                    SELECT id as message_id, role, content as text, (1 - (embedding <=> %s::vector)) as score
                    FROM messages
                    WHERE user_email = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """
                cur.execute(query, [query_embeddings, user_email_resolved, query_embeddings, top_k])
            else:
                if len(allowed_message_ids) == 1:
                    in_clause = "(%s)"
                    ids_param = list(allowed_message_ids)[0]
                else:
                    in_clause = "%s"
                    ids_param = tuple(allowed_message_ids)

                query = f"""
                    SELECT id as message_id, role, content as text, (1 - (embedding <=> %s::vector)) as score
                    FROM messages
                    WHERE id IN {in_clause}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """
                cur.execute(query, [query_embeddings, ids_param, query_embeddings, top_k])

            return [dict(r) for r in cur.fetchall()]

        except Exception as e:
            logger.error(f"Error in find_similar: {e}")
            return []
        finally:
            conn.close()

    def update_thread_summary(self, user_id: str, thread_id: str, summary: str) -> bool:
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute("UPDATE nodes SET summary = %s WHERE id = %s", (summary, thread_id))
            conn.commit()
            return cur.rowcount > 0
        except Exception:
            conn.rollback()
            return False
        finally:
            conn.close()

    def get_thread_summary(self, user_id: str, thread_id: str) -> str:
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT summary FROM nodes WHERE id = %s", (thread_id,))
            row = cur.fetchone()
            return row[0] if row else ""
        except Exception:
            return ""
        finally:
            conn.close()

    def get_thread_message_count(self, thread_id: str) -> int:
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM messages WHERE node_id = %s", (thread_id,))
            row = cur.fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0
        finally:
            conn.close()

    def get_message_by_id(self, message_id: str) -> Optional[dict]:
        conn = self._get_conn()
        try:
            cur = conn.cursor(cursor_factory=DictCursor)
            cur.execute(
                """
                SELECT id as message_id, node_id as thread_id, role, content as text, timestamp, embedding
                FROM messages
                WHERE id = %s
                """,
                (message_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error getting message by id: {e}")
            return None
        finally:
            conn.close()

    def get_recent_messages(self, limit: int = 10) -> List[dict]:
        conn = self._get_conn()
        try:
            cur = conn.cursor(cursor_factory=DictCursor)
            cur.execute(
                """
                SELECT id as message_id, node_id as thread_id, role, content as text, timestamp
                FROM messages
                ORDER BY timestamp DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Error getting recent messages: {e}")
            return []
        finally:
            conn.close()

    def thread_exists(self, thread_id: str) -> bool:
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM nodes WHERE id = %s", (thread_id,))
            return cur.fetchone() is not None
        except Exception:
            return False
        finally:
            conn.close()

    def _resolve_message_id(self, target_id: str, candidate_ids: List[str]) -> Optional[str]:
        """
        Smart fuzzy matching to handle frontend (-u/-a) vs backend (_ai/_) ID mismatches.
        Returns the matching ID from candidate_ids if found, else None.
        """
        if target_id in candidate_ids:
            return target_id

        def normalize_id(mid):
            if not isinstance(mid, str): return str(mid)
            # Strip known suffixes to get base ID
            base = mid
            for suffix in ["_ai", "_a", "-a", "_u", "-u"]:
                if base.endswith(suffix):
                    base = base[:-len(suffix)]
                    break
            return base

        def get_role_hint(mid):
                if not isinstance(mid, str): return "unknown"
                if mid.endswith(("_ai", "_a", "-a")): return "assistant"
                if mid.endswith(("_u", "-u")): return "user"
                return "user" # Default assumption for base IDs

        req_base = normalize_id(target_id)
        req_role = get_role_hint(target_id)

        for mid in candidate_ids:
            if not isinstance(mid, str): continue
            
            db_base = normalize_id(mid)
            db_role = get_role_hint(mid)
            
            if req_base == db_base:
                if req_role == db_role:
                    return mid
                
                # Fallback: if we are looking for 'user' and DB has base ID only
                if req_role == "user" and db_role == "user": 
                        return mid
                
                # Fallback: if we look for assistant and DB has _ai or _a
                if req_role == "assistant" and db_role == "assistant":
                        return mid

        # Original legacy fallback
        for mid in candidate_ids:
            if isinstance(mid, str) and (mid.startswith(target_id) or target_id.startswith(mid) or mid.replace("_u", "") == target_id.replace("_u", "")):
                return mid
        
        return None

    def get_messages_until(self, user_id: str, thread_id: str, message_id: str) -> Tuple[str, List[dict]]:
        summary = self.get_thread_summary(user_id, thread_id)
        msgs = self.get_thread_messages(user_id, thread_id)
        
        target_msgs = []
        found = False
        ids = [m['message_id'] for m in msgs]

        resolved_message_id = self._resolve_message_id(message_id, ids)
        if not resolved_message_id:
             # If strictly not found, assume not found (already tried fallbacks)
             pass

        for m in msgs:
            target_msgs.append(m)
            if m['message_id'] == resolved_message_id:
                found = True
                break
        
        if not found:
            return summary, []
            
        return summary, target_msgs

    def fork_thread(
        self,
        user_id: str,
        source_thread_id: str,
        new_thread_id: str,
        fork_at_message_id: str,
        summary: str = None,
        summary_embedding: List[float] = None,
        initial_messages: List[dict] = []
    ) -> bool:
        conn = self._get_conn()
        try:
            register_vector(conn)
            cur = conn.cursor()
            
            user_email = self._resolve_user_email(cur, user_id)
            now = datetime.utcnow()
            
            # Get Source Node Details (Canvas ID mostly)
            cur.execute("SELECT canvas_id, user_email FROM nodes WHERE id = %s", (source_thread_id,))
            source_node = cur.fetchone()
            if not source_node:
                return False
            canvas_id = source_node[0]

            # If child already exists, update summary and optionally seed buffer
            cur.execute("SELECT id FROM nodes WHERE id = %s", (new_thread_id,))
            child_exists = cur.fetchone() is not None
            if child_exists:
                if summary is not None:
                    if summary_embedding is not None:
                        cur.execute(
                            "UPDATE nodes SET summary = %s, summary_embedding = %s WHERE id = %s",
                            (summary, summary_embedding, new_thread_id),
                        )
                    else:
                        cur.execute(
                            "UPDATE nodes SET summary = %s WHERE id = %s",
                            (summary, new_thread_id),
                        )

                cur.execute("SELECT COUNT(*) FROM messages WHERE node_id = %s", (new_thread_id,))
                count_row = cur.fetchone()
                child_msg_count = int(count_row[0]) if count_row else 0

                # Allow insertion if count is 0 OR 1 (assuming the 1 is the new user trigger message)
                if initial_messages and child_msg_count <= 1:
                    # Check if we already have these messages (deduplication heuristic)
                    # If count is 1, and we insert, we might duplicate if we already did this?
                    # But if we did this, count would be 1 + len(initial_messages) > 1 (assuming len > 0).
                    # So checking count <= 1 is safe provided len(initial_messages) >= 1.
                    
                    for idx, msg in enumerate(initial_messages):
                        import uuid
                        new_msg_id = str(uuid.uuid4())

                        emb = msg.get('embedding')
                        if emb is None or emb == {}:
                            emb = []
                        elif isinstance(emb, np.ndarray):
                            emb = emb.tolist()
                        elif not isinstance(emb, list):
                            emb = []
                        
                        if not emb:
                             emb = None

                        cur.execute(
                            """
                            INSERT INTO messages (id, node_id, role, content, embedding, timestamp, user_email)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                new_msg_id,
                                new_thread_id,
                                msg.get('role'),
                                msg.get('text', '') or msg.get('content', ''),
                                emb,
                                msg.get('timestamp', now),
                                user_email,
                            ),
                        )

                conn.commit()
                return True

            # In Mongo store, if summary fallback was needed:
            # final_summary = summary if summary is not None else source.get("summary")
            # We rely on provided summary or null. (Code provided in MongoStore mostly passed it or used existing)
            
            cur.execute("""
                INSERT INTO nodes (
                    id, canvas_id, user_email, parent_node_id, 
                    forked_from_message_id, summary, summary_embedding, created_at, is_primary
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                new_thread_id,
                canvas_id,
                user_email,
                source_thread_id,
                fork_at_message_id,
                summary,
                summary_embedding,
                now,
                False 
            ))
            
            # Insert Initial Messages (Buffer)
            if initial_messages:
                for idx, msg in enumerate(initial_messages):
                     # Ensure we have message_id. 
                     # If these are copies, they should theoretically have new IDs if we want them independent?
                     # The prompt says: "Take a small buffer of exact messages". "raw_messages_B = [buffer messages]".
                     # If we reuse IDs, future edits/deletions in B might affect A if not careful? 
                     # But 'messages' table has 'node_id' as PK part? No, 'id' (UUID) is usually PK.
                     # If 'id' is PK, we cannot insert same message_id for different node_id.
                     # We MUST generate new IDs for the copied messages in B.
                     # Or the schema uses (id, node_id) as PK? 
                     # The insert in add_message uses: ON CONFLICT (id) DO NOTHING.
                     # This suggests 'id' is unique global.
                     # So we MUST generate NEW IDs for the buffer messages in the new thread.
                     
                     import uuid
                     new_msg_id = str(uuid.uuid4())
                     
                     # We might want to keep reference to original? But ARC says "No dependency".
                     
                     emb = msg.get('embedding')
                     if emb is None or emb == {}:
                         emb = []
                     elif isinstance(emb, np.ndarray):
                         emb = emb.tolist()
                     elif not isinstance(emb, list):
                         emb = []

                     # Fix for "invalid input syntax for type vector: '{}'"
                     if not emb:
                         emb = None
                     
                     cur.execute("""
                        INSERT INTO messages (id, node_id, role, content, embedding, timestamp, user_email)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                     """, (
                        new_msg_id, 
                        new_thread_id, 
                        msg.get('role'), 
                        msg.get('text', '') or msg.get('content', ''), 
                        emb, 
                        msg.get('timestamp', now), 
                        user_email
                     ))
            
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Error forking: {e}")
            return False
        finally:
            conn.close()

    def get_file_binary(self, file_id: str) -> Optional[dict]:
        """
        Retrieves the binary content and metadata of a file from external_files.
        """
        conn = self._get_conn()
        try:
            cur = conn.cursor(cursor_factory=DictCursor)
            cur.execute("""
                SELECT id, file_name, mime_type, data 
                FROM external_files 
                WHERE id = %s
            """, (file_id,))
            row = cur.fetchone()
            if row:
                return dict(row)
            return None
        except Exception as e:
            logger.error(f"Error fetching file binary: {e}")
            raise e
        finally:
            conn.close()

    def save_file_chunks(self, file_id: str, chunks: List[str], embeddings: List[List[float]], metadatas: Optional[List[dict]] = None):
        """
        Saves chunks and embeddings to file_chunks table.
        Updates external_files.processed = true.
        """
        conn = self._get_conn()
        try:
            register_vector(conn)
            cur = conn.cursor()
            
            # Ensure metadatas is list of same length if None
            if metadatas is None:
                metadatas = [{'source': 'unknown'}] * len(chunks)
            
            # 1. Insert Chunks
            data_list = []
            for idx, (text, emb, meta) in enumerate(zip(chunks, embeddings, metadatas)):
                import json
                # Ensure metadata is json serializable dict
                meta_json = json.dumps(meta)
                item = (file_id, idx, text, emb, meta_json)
                data_list.append(item)
            
            from psycopg2.extras import execute_values
            
            insert_query = """
                INSERT INTO file_chunks (file_id, chunk_index, chunk_text, embedding, metadata)
                VALUES %s
            """
            
            execute_values(cur, insert_query, data_list)

            # 2. Update processed status
            cur.execute("""
                UPDATE external_files 
                SET processed = true, updated_at = NOW() 
                WHERE id = %s
            """, (file_id,))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error saving file chunks: {e}")
            raise e
        finally:
            conn.close()

    def get_related_file_context(self, node_id: str, query_embedding: List[float], limit: int = 3) -> List[dict]:
        """
        Finds the nearest file chunks connected to a given node (via edges).
        Returns chunk text plus metadata so the caller can format rich context.
        """
        if not query_embedding:
            return []

        conn = self._get_conn()
        try:
            register_vector(conn)
            cur = conn.cursor(cursor_factory=DictCursor)
            
            query = """
                WITH connected_files AS (
                    SELECT n.id as file_node_id
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
                    (fc.embedding <=> %s::vector) as distance,
                    ef.file_name,
                    ef.file_type
                FROM file_chunks fc
                JOIN external_files ef ON fc.file_id = ef.id
                JOIN connected_files cf ON ef.node_id = cf.file_node_id 
                ORDER BY distance ASC
                LIMIT %s
            """
            
            cur.execute(query, (node_id, node_id, node_id, np.array(query_embedding), limit))
            rows = cur.fetchall()
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error fetching related file context: {e}")
            return []
        finally:
            conn.close()

    def update_node_data_content(self, node_id: str, content: str) -> bool:
        """
        Persist extracted text onto the node's data jsonb and clear any loading flags.
        Also replaces the placeholder contextContract ("Processing...") with the
        extracted text or a friendly fallback when the file has no text.
        """
        conn = self._get_conn()
        try:
            cur = conn.cursor()

            text_value = content or ""
            contract_value = content if content else "(No text content extracted)"

            cur.execute(
                """
                UPDATE nodes 
                SET data = jsonb_set(
                            jsonb_set(
                                jsonb_set(
                                    jsonb_set(
                                        jsonb_set(
                                            jsonb_set(coalesce(data, '{}'::jsonb), '{data,content}', to_jsonb(%s::text), true),
                                            '{data,loading}', 'false'::jsonb, true
                                        ),
                                        '{content}', to_jsonb(%s::text), true
                                    ),
                                    '{loading}', 'false'::jsonb, true
                                ),
                                '{contextContract}', to_jsonb(%s::text), true
                            ),
                            '{data,contextContract}', to_jsonb(%s::text), true
                        )
                WHERE id = %s
                """,
                (text_value, text_value, contract_value, contract_value, node_id)
            )
            conn.commit()
            return cur.rowcount > 0
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating node data content: {e}")
            return False
        finally:
            conn.close()
