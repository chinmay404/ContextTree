import os
from dotenv import load_dotenv
import urllib.parse
from datetime import datetime
from pymongo.operations import SearchIndexModel
from pymongo import MongoClient, errors
from app.agent.utils.embeddings import get_embedding

load_dotenv()


class MongoConversationStore:
    def __init__(
        self,
        db_name: str = "Conversationstore",
        user_coll: str = "users",
        msg_coll_name: str = "messages",
        embedding_dim: int = 768
    ):
        """
        Initializes collections:
        - `users`: stores per-user threads referencing message_ids.
        - `messages`: flat collection for full messages and vector search.
        """
        mongo_username = os.getenv("MONGO_USERNAME")
        mongo_password = os.getenv("MONGO_PASSWORD")
        mongo_uri = os.getenv("MONGODB_URI")

        if mongo_uri:
             self.client = MongoClient(mongo_uri)
        elif mongo_username and mongo_password:
            self.username = urllib.parse.quote_plus(mongo_username)
            self.password = urllib.parse.quote_plus(mongo_password)
            self.client = MongoClient(
                f"mongodb+srv://{self.username}:{self.password}@contexttree.4g4brxh.mongodb.net/?retryWrites=true&w=majority&tls=true&appName=ContextTree"
            )
        else:
            raise ValueError("MONGO_USERNAME/MONGO_PASSWORD or MONGODB_URI not found.")
        self.user_col = self.client[db_name][user_coll]
        self.msg_coll = self.client[db_name][msg_coll_name]

        # Ensure `messages` collection exists
        db = self.client[db_name]
        if msg_coll_name not in db.list_collection_names():
            db.create_collection(msg_coll_name)

        # Ensure Atlas Vector Search index exists
        self._ensure_vector_index(embedding_dim)

    def _ensure_vector_index(self, dim: int):
        """
        Create an Atlas Vector Search index on 'embedding' if missing.
        """
        try:
            existing = [idx["name"]
                        for idx in self.msg_coll.list_search_indexes()]
        except errors.OperationFailure:
            existing = []

        if "embedding_vector_index" not in existing:
            vector_idx = SearchIndexModel(
                definition={
                    "fields": [
                        {"type": "vector", "path": "embedding",
                            "similarity": "cosine", "dims": dim}
                    ]
                },
                name="embedding_vector_index",
                type="vectorSearch"
            )
            self.msg_coll.create_search_index(model=vector_idx)

    def add_message(
        self,
        user_id: str,
        thread_id: str,
        message_id: str,
        role: str,
        text: str,
        embedding: list[float],
        summary: str = None,
        summarize_fn=None,
        embed_summary_fn=None,
        context_fn=None
    ):
        """
        Add a message:
        1. Upsert user->thread referencing only message_id.
        2. Write full message into flat `messages` collection.
        3. Update context and summary on the thread doc.
        """
        now = datetime.utcnow()
        self.user_col.update_one(
            {"user_id": user_id},
            {"$setOnInsert": {"user_id": user_id, "threads": []}},
            upsert=True
        )
        # Compute summary embedding if possible
        summary_embedding = []
        if summary and embed_summary_fn and callable(embed_summary_fn):
            try:
                summary_embedding = embed_summary_fn(summary) or []
            except Exception:
                summary_embedding = []

        update_query = {"$push": {"threads.$.message_ids": message_id}}
        if summary:
            set_fields = {"threads.$.summary": summary}
            if summary_embedding:
                set_fields["threads.$.summary_embedding"] = summary_embedding
            update_query["$set"] = set_fields
            
        result = self.user_col.update_one(
            {"user_id": user_id, "threads._id": thread_id},
            update_query
        )
        if result.matched_count == 0:
            new_thread = {
                "_id": thread_id,
                "started_at": now,
                "summary": summary or "",
                "summary_embedding": summary_embedding,
                "context": [],
                "message_ids": [message_id],
                "parent_thread": [],
                "child_thread": [],
            }
            self.user_col.update_one(
                {"user_id": user_id},
                {"$push": {"threads": new_thread}}
            )
        flat_doc = {
            "user_id":    user_id,
            "thread_id":  thread_id,
            "message_id": message_id,
            "role":       role,
            "text":       text,
            "embedding":  embedding,
            "timestamp":  now
        }
        self.msg_coll.replace_one(
            {"message_id": message_id},
            flat_doc,
            upsert=True
        )

    def get_thread_messages(self, user_id: str, thread_id: str) -> list[dict]:
        """
        Retrieve full message documents for a given user_id and thread_id,
        in original insertion order.
        """
        doc = self.user_col.find_one(
            {"user_id": user_id, "threads._id": thread_id},
            {"threads.$.message_ids": 1}
        )
        if not doc or "threads" not in doc:
            return []
        message_ids = doc["threads"][0]["message_ids"]
        msgs = list(self.msg_coll.find(
            {"message_id": {"$in": message_ids}},
            {"_id": 0, "message_id": 1, "role": 1, "text": 1, "timestamp": 1}
        ))
        id_to_msg = {m["message_id"]: m for m in msgs}
        return [id_to_msg[mid] for mid in message_ids if mid in id_to_msg]

    # Need to work on this

    def find_similar_by_message_id(
        self,
        user_id: str,
        thread_queries: list[tuple[str, str | None]],
        query_embeddings: list[float],
        top_k: int = 3
    ) -> list[dict]:
        """
        Given a list of (thread_id, message_id) pairs (message_id can be None),
        and a query_embeddings vector, search for top_k messages similar to it.

        If message_id is None, search the whole thread.
        If message_id is provided, search only messages before that point.
        """
        candidate_ids = []
        for thread_id, msg_id in thread_queries:
            user_doc = self.user_col.find_one(
                {"user_id": user_id, "threads._id": thread_id}, {"threads.$": 1}
            )
            if not user_doc or not user_doc.get("threads"):
                continue
            ids = user_doc["threads"][0].get("message_ids", [])
            if msg_id and msg_id in ids:
                candidate_ids.extend(ids[:ids.index(msg_id)])
            elif msg_id is None:
                candidate_ids.extend(ids)
        if not candidate_ids:
            return []

        # Ensure numCandidates >= limit
        num_candidates = max(top_k * 10, top_k)

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "embedding_vector_index",
                    "queryVector": query_embeddings,
                    "path": "embedding",
                    "numCandidates": num_candidates,
                    "limit": top_k,
                    "filter": {"message_id": {"$in": candidate_ids}}
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "message_id": 1,
                    "role": 1,
                    "text": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        return list(self.msg_coll.aggregate(pipeline))

    def update_thread_summary(self, user_id: str, thread_id: str, summary: str):
        """
        Update the summary for a specific thread.
        """
        result = self.user_col.update_one(
            {"user_id": user_id, "threads._id": thread_id},
            {"$set": {"threads.$.summary": summary}}
        )
        return result.modified_count > 0

    def get_thread_summary(self, user_id: str, thread_id: str) -> str:
        """
        Retrieve the summary for a specific thread.
        Returns empty string if thread not found or has no summary.
        """
        doc = self.user_col.find_one(
            {"user_id": user_id, "threads._id": thread_id},
            {"threads.$": 1}
        )
        if doc and "threads" in doc and len(doc["threads"]) > 0:
            return doc["threads"][0].get("summary", "")
        return ""

    def get_messages_until(self, user_id: str, thread_id: str, message_id: str) -> tuple[str, list[dict]]:
        """
        Get the summary and all messages in a thread up to a specific message_id.
        Returns: (summary, list_of_message_dicts)
        """
        doc = self.user_col.find_one(
            {"user_id": user_id, "threads._id": thread_id},
            {"threads.$": 1}
        )
        if not doc or "threads" not in doc:
            return "", []
            
        thread = doc["threads"][0]
        summary = thread.get("summary", "")
        message_ids = thread.get("message_ids", [])
        
        if message_id not in message_ids:
            return summary, [] # Or verify if we should return all?
            
        target_index = message_ids.index(message_id)
        target_ids = message_ids[:target_index + 1]
        
        msgs = list(self.msg_coll.find(
            {"message_id": {"$in": target_ids}},
            {"_id": 0, "message_id": 1, "role": 1, "text": 1, "timestamp": 1}
        ))
        
        # Sort by order in message_ids
        id_to_msg = {m["message_id"]: m for m in msgs}
        ordered_msgs = [id_to_msg[mid] for mid in target_ids if mid in id_to_msg]
        
        return summary, ordered_msgs


    def fork_thread(
        self,
        user_id: str,
        source_thread_id: str,
        new_thread_id: str,
        fork_at_message_id: str,
        summary: str = None,
        summary_embedding: list[float] = None
    ) -> bool:
        """
        Fork a thread at a specific message:
        1. Find all messages up to and including fork_at_message_id
        2. Create new thread with those messages
        3. Copy summary from source thread (or use provided one)
        4. Update parent/child relationships
        """
        # Get source thread
        source_doc = self.user_col.find_one(
            {"user_id": user_id, "threads._id": source_thread_id},
            {"threads.$": 1}
        )
        
        if not source_doc or "threads" not in source_doc:
            return False
        
        source_thread = source_doc["threads"][0]
        message_ids = source_thread.get("message_ids", [])
        
        # Find fork point
        if fork_at_message_id not in message_ids:
            return False
        
        fork_index = message_ids.index(fork_at_message_id)
        forked_message_ids = message_ids[:fork_index + 1]
        
        # Use provided summary or fallback to source thread's summary
        final_summary = summary if summary is not None else source_thread.get("summary", "")
        final_summary_embedding = summary_embedding if summary_embedding is not None else source_thread.get("summary_embedding", [])

        # Create new thread
        now = datetime.utcnow()
        new_thread = {
            "_id": new_thread_id,
            "started_at": now,
            "summary": final_summary,
            "summary_embedding": final_summary_embedding,
            "context": source_thread.get("context", []),
            "message_ids": forked_message_ids,
            "parent_thread": [source_thread_id],
            "child_thread": [],
        }
        
        # Add new thread to user
        result = self.user_col.update_one(
            {"user_id": user_id},
            {"$push": {"threads": new_thread}}
        )
        
        # Update parent thread's child_thread list
        if result.modified_count > 0:
            self.user_col.update_one(
                {"user_id": user_id, "threads._id": source_thread_id},
                {"$push": {"threads.$.child_thread": new_thread_id}}
            )
            return True
        
        return False


def main():
    store = MongoConversationStore(
        db_name="Conversationstore",
        user_coll="users",
        msg_coll_name="messages",
        embedding_dim=768
    )

    user_id = "555"
    # Define threads and optional message_id to bound search
    # Here, we search in thread 'threadA' up to message 'msg42',
    # and entire thread 'threadB' (message_id=None)
    thread_queries = [
        ("000", "12344"),
    ]

    # Create or retrieve the embedding vector you want to query.
    # For example, generate via your embedding model:
    query_embeddings = get_embedding("Deepseek ")

    # Find top 5 similar messages
    similar_messages = store.find_similar_by_message_id(
        user_id=user_id,
        thread_queries=thread_queries,
        query_embeddings=query_embeddings,
        top_k=5
    )

    print("Similar messages:")
    print(similar_messages)
    for msg in similar_messages:
        print(
            f"Score: {msg['score']:.4f} | Thread: {msg.get('thread_id', 'unknown')} | Role: {msg['role']} | Text: {msg['text']}")


# main()
