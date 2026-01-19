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
        embedding_dim: int = 1536
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
            
            for thread_id, limit_msg_id in thread_queries:
                # We reuse get_thread_messages logic to get the list of relevant messages
                # This is inefficient for large scale but correct.
                # Creating a temporary list.
                # Could be optimized with recursive SQL CTE but pgvector + CTE is fine.
                
                # Retrieve full history for this thread
                msgs = self.get_thread_messages(user_id, thread_id)
                ids = [m['message_id'] for m in msgs]
                
                if limit_msg_id:
                    if limit_msg_id in ids:
                        idx = ids.index(limit_msg_id)
                        allowed_message_ids.update(ids[:idx])
                else:
                    allowed_message_ids.update(ids)
            
            if not allowed_message_ids:
                return []
                
            # Vector Search
            if len(allowed_message_ids) == 1:
                in_clause = "(%s)"
                params = [list(allowed_message_ids)[0]]
            else:
                in_clause = "%s"
                params = [tuple(allowed_message_ids)]
                
            query = f"""
                SELECT id as message_id, role, content as text, (1 - (embedding <=> %s::vector)) as score
                FROM messages
                WHERE id IN {in_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            
            cur.execute(query, [query_embeddings, tuple(allowed_message_ids) if len(allowed_message_ids)>1 else list(allowed_message_ids)[0], query_embeddings, top_k])
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

    def get_messages_until(self, user_id: str, thread_id: str, message_id: str) -> Tuple[str, List[dict]]:
        summary = self.get_thread_summary(user_id, thread_id)
        msgs = self.get_thread_messages(user_id, thread_id)
        
        target_msgs = []
        found = False
        for m in msgs:
            target_msgs.append(m)
            if m['message_id'] == message_id:
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
                     if emb is None:
                         emb = []
                     elif isinstance(emb, np.ndarray):
                         emb = emb.tolist()
                     elif not isinstance(emb, list):
                         # If it's something else (e.g. string), leave it or handle it?
                         # For now assume list/array/None
                         pass

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
