import unittest
import uuid
import os
import sys
import psycopg2
from datetime import datetime
from dotenv import load_dotenv

# Ensure app modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
load_dotenv()

from app.agent.store.PostgresStore import PostgresConversationStore

# Mock embedding function to avoid API calls during DB tests
def mock_get_embedding(text: str):
    # Return a random normalized vector of dimension 1536
    import random
    vec = [random.random() for _ in range(1536)]
    norm = sum(x*x for x in vec) ** 0.5
    return [x/norm for x in vec]

class TestPostgresStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize the store and create a dummy user for testing."""
        cls.store = PostgresConversationStore()
        
        # Create a unique test user email
        cls.test_email = f"test_user_{uuid.uuid4().hex[:8]}@example.com"
        cls.user_id = str(uuid.uuid4())
        
        # Manually insert a test user into the database to satisfy Foreign Keys
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        try:
            with conn.cursor() as cur:
                # Check if users table uses email or id as primary/fk reference based on schema
                # Based on provided schema: id is PK, email is UNIQUE.
                cur.execute(
                    "INSERT INTO users (id, email, name, created_at, updated_at) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (email) DO NOTHING",
                    (cls.user_id, cls.test_email, "Test User", datetime.now(), datetime.now())
                )
            conn.commit()
        finally:
            conn.close()
            
        print(f"--- Setup: Created test user {cls.test_email} ---")

    def setUp(self):
        self.thread_id = str(uuid.uuid4())

    def test_1_add_and_get_messages(self):
        """Test adding messages and retrieving them from a thread."""
        msg_id_1 = str(uuid.uuid4())
        text_1 = "Hello, this is a test message."
        
        # 1. Add User Message
        self.store.add_message(
            user_id=self.test_email, # Using email as user_id based on store logic
            thread_id=self.thread_id,
            message_id=msg_id_1,
            role="user",
            text=text_1,
            embedding=mock_get_embedding(text_1),
            summary="Initial summary"
        )
        
        msg_id_2 = str(uuid.uuid4())
        text_2 = "I am the assistant response."
        
        # 2. Add Assistant Message
        self.store.add_message(
            user_id=self.test_email,
            thread_id=self.thread_id,
            message_id=msg_id_2,
            role="assistant",
            text=text_2,
            embedding=mock_get_embedding(text_2),
            summary="Updated summary"
        )
        
        # 3. Retrieve Messages
        messages = self.store.get_thread_messages(self.test_email, self.thread_id)
        
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]['text'], text_1)
        self.assertEqual(messages[0]['role'], 'user')
        self.assertEqual(messages[1]['text'], text_2)
        self.assertEqual(messages[1]['role'], 'assistant')
        
        print(f"--- Passed: add_message and get_thread_messages ---")

    def test_2_vector_search(self):
        """Test finding similar messages using vector search."""
        # Insert a specific message to find
        target_text = "The quick brown fox jumps over the lazy dog."
        msg_id = str(uuid.uuid4())
        
        embedding = mock_get_embedding(target_text)
        
        self.store.add_message(
            user_id=self.test_email,
            thread_id=self.thread_id,
            message_id=msg_id,
            role="user",
            text=target_text,
            embedding=embedding
        )
        
        # Perform search using the same embedding
        # Pass the thread_id to restrict search or None to search all (impl dependent)
        # The Store's signature: find_similar_by_message_id(user_id, thread_queries, query_embeddings, top_k)
        
        # thread_queries is list of (thread_id, msg_id_limit)
        results = self.store.find_similar_by_message_id(
            user_id=self.test_email,
            thread_queries=[(self.thread_id, None)], 
            query_embeddings=embedding,
            top_k=1
        )
        
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]['text'], target_text)
        # Cosine similarity for identical vector should be close to 1 (or 0 distance depending on metric, usually 1 in pgvector <=> cosine)
        # Note: pgvector cosine operator <=> is actually 1 - cosine_similarity. 
        # But aggregate pipeline usually transforms it. Let's just check equality.
        
        print(f"--- Passed: vector_search found 1 match with score {results[0].get('score')} ---")

    def test_3_fork_thread(self):
        """Test forking a thread from a specific message."""
        # 1. Create a root thread with 3 messages
        msg_ids = [str(uuid.uuid4()) for _ in range(3)]
        for i, mid in enumerate(msg_ids):
            self.store.add_message(
                user_id=self.test_email,
                thread_id=self.thread_id,
                message_id=mid,
                role="user",
                text=f"Message {i}",
                embedding=mock_get_embedding(f"Message {i}")
            )
            
        # 2. Fork at the second message (index 1)
        new_thread_id = str(uuid.uuid4())
        fork_point = msg_ids[1]
        
        success = self.store.fork_thread(
            user_id=self.test_email,
            source_thread_id=self.thread_id,
            new_thread_id=new_thread_id,
            fork_at_message_id=fork_point,
            summary="Forked Summary"
        )
        
        self.assertTrue(success)
        
        # 3. Verify new thread history
        # Should contain Message 0 and Message 1, but NOT Message 2
        new_history = self.store.get_thread_messages(self.test_email, new_thread_id)
        
        self.assertEqual(len(new_history), 2)
        self.assertEqual(new_history[0]['message_id'], msg_ids[0])
        self.assertEqual(new_history[1]['message_id'], msg_ids[1])
        
        print(f"--- Passed: fork_thread verified history containment ---")

if __name__ == '__main__':
    unittest.main()