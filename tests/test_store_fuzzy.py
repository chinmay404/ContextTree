import unittest
import uuid
import os
import sys
import psycopg2
from datetime import datetime
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
load_dotenv()

from app.agent.store.PostgresStore import PostgresConversationStore

def mock_get_embedding(text: str):
    return [0.1] * 768

class TestPostgresStoreFuzzy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = PostgresConversationStore()
        cls.test_email = f"test_user_fuzzy_{uuid.uuid4().hex[:8]}@example.com"
        cls.user_id = str(uuid.uuid4())
        
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (id, email, name, created_at, updated_at) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (email) DO NOTHING",
                    (cls.user_id, cls.test_email, "Test Fuzzy User", datetime.now(), datetime.now())
                )
            conn.commit()
        finally:
            conn.close()

    def setUp(self):
        self.thread_id = str(uuid.uuid4())

    def test_fuzzy_matching(self):
        # Case 1: DB has base ID, Frontend asks for suffixed ID
        base_id = str(uuid.uuid4())
        
        # Insert "User" message with base ID
        self.store.add_message(
            user_id=self.test_email,
            thread_id=self.thread_id,
            message_id=base_id,
            role="user",
            text="User message base",
            embedding=mock_get_embedding("User message base")
        )

        # Frontend asks for "base_id-u"
        frontend_id = f"{base_id}-u"
        summary, msgs = self.store.get_messages_until(self.test_email, self.thread_id, frontend_id)
        
        self.assertTrue(len(msgs) > 0, "Should find message despite suffix mismatch")
        self.assertEqual(msgs[-1]['message_id'], base_id)
        print("Success: Mapped -u request to base ID")

        # Case 2: DB has _ai ID, Frontend asks for -a ID
        ai_base_id = str(uuid.uuid4())
        db_ai_id = f"{ai_base_id}_ai"
        
        self.store.add_message(
            user_id=self.test_email,
            thread_id=self.thread_id,
            message_id=db_ai_id,
            role="assistant",
            text="AI response",
            embedding=mock_get_embedding("AI response")
        )

        # Frontend asks for "base_id-a"
        frontend_ai_id = f"{ai_base_id}-a"
        summary, msgs = self.store.get_messages_until(self.test_email, self.thread_id, frontend_ai_id)

        self.assertTrue(len(msgs) > 0, "Should find AI message despite suffix mismatch")
        self.assertEqual(msgs[-1]['message_id'], db_ai_id)
        print("Success: Mapped -a request to _ai ID")

if __name__ == "__main__":
    unittest.main()
