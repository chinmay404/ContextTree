from pymongo import MongoClient, UpdateOne
from datetime import datetime
import urllib.parse
from dotenv import load_dotenv
import os

load_dotenv()

class MongoConversationStore:
    def __init__(self, db_name: str = "Conversationstore", coll_name: str = "users"):
        """
        uri: your MongoDB connection string, e.g.
             "mongodb+srv://<user>:<pass>@cluster0.mongodb.net/?retryWrites=true&w=majority"
        This store now has one document per user, with multiple threads each.
        """
        self.username = urllib.parse.quote_plus(os.getenv("MONGO_USERNAME"))
        self.password = urllib.parse.quote_plus(os.getenv("MONGO_PASSWORD"))
        self.client = MongoClient(f"mongodb+srv://{self.username}:{self.password}@contexttree.4g4brxh.mongodb.net/?retryWrites=true&w=majority&appName=ContextTree")
        # self.client = MongoClient("mongodb://localhost:27017/")
        
        self.col = self.client[db_name][coll_name]

    def add_message(
        self,
        user_id: str,
        thread_id: str,
        role: str,
        text: str,
        embedding: list[float],
        summarize_fn=None,
        embed_summary_fn=None,
        context_fn=None
    ):
        """
        Add a message under user_id → thread_id. 
        If this is the first time we see user_id, create the user doc.
        If thread_id is new, push a new thread entry; 
        otherwise append to existing thread.messages.
        Then recompute context & summary for that thread.
        """
        now = datetime.utcnow()
        msg_doc = {
            "role": role,
            "text": text,
            "timestamp": now,
            "embedding": embedding
        }

        # 1) Ensure the user document exists
        self.col.update_one(
            {"user_id": user_id},
            {"$setOnInsert": {"user_id": user_id, "threads": []}},
            upsert=True
        )

        # 2) Try to append to an existing thread
        result = self.col.update_one(
            {"user_id": user_id, "threads._id": thread_id},
            {"$push": {"threads.$.messages": msg_doc}}
        )

        # 3) If no thread was updated, create a brand-new thread entry
        if result.matched_count == 0:
            new_thread = {
                "_id": thread_id,
                "started_at": now,
                "summary": "",
                "summary_embedding": [],
                "context": [],
                "messages": [msg_doc],
                "context": [],
                "parent_thread": [],
                "child_thread": [],
            }
            self.col.update_one(
                {"user_id": user_id},
                {"$push": {"threads": new_thread}}
            )

        # 4) Re-fetch this thread’s messages to recalc context/summary
        user_doc = self.col.find_one(
            {"user_id": user_id, "threads._id": thread_id},
            {"threads.$": 1}
        )
        thread = user_doc["threads"][0]
        embeddings = [m["embedding"] for m in thread["messages"]]

        # 5) Compute new context vector
        new_context = context_fn(embeddings) if context_fn else []

        # 6) Compute new summary text
        texts = [f'{m["role"]}: {m["text"]}' for m in thread["messages"]]
        # new_summary = summarize_fn(texts) if summarize_fn else ""
        new_summary = None
        # 7) Compute new summary embedding
        if new_summary:
            new_sum_emb = embed_summary_fn(new_summary) if embed_summary_fn else []
        else:
            new_sum_emb = []
        # 8) Update that specific thread in-place
        self.col.update_one(
            {"user_id": user_id, "threads._id": thread_id},
            {"$set": {
                "threads.$.context": new_context,
                "threads.$.summary": new_summary,
                "threads.$.summary_embedding": new_sum_emb
            }}
        )

    def get_thread(self, user_id: str, thread_id: str) -> dict:
        """
        Return the specific thread for a user, including messages, context, and summary.
        """
        doc = self.col.find_one(
            {"user_id": user_id, "threads._id": thread_id},
            {"threads.$": 1}
        )
        return doc["threads"][0] if doc and "threads" in doc else None
