import os
import sys
import psycopg2
from dotenv import load_dotenv

# Add parent directory to path to import correctly if run from scripts folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

def migrate_vectors():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("Error: DATABASE_URL not found in environment variables.")
        return

    print("Connecting to database...")
    try:
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
        cur = conn.cursor()

        print("Updating vector dimensions from 1536 to 768...")
        
        # 1. file_chunks
        try:
            print("Migrating table: file_chunks...")
            # We must drop the index first as it depends on the column type
            cur.execute("DROP INDEX IF EXISTS idx_file_chunks_embedding;")
            # Alter column type. Note: This clears data if casting is impossible, but for vectors it usually requires
            # explicit USING clause or dropping data. Since existing 1536 vectors can't map to 768, we wipe them.
            cur.execute("ALTER TABLE file_chunks ALTER COLUMN embedding TYPE vector(768) USING NULL;") 
            # Recreate index
            cur.execute("CREATE INDEX idx_file_chunks_embedding ON file_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
            print("file_chunks migrated successfully (embeddings cleared).")
        except Exception as e:
            print(f"Error migrating file_chunks: {e}")

        # 2. context_chunks
        try:
            print("Migrating table: context_chunks...")
            cur.execute("ALTER TABLE context_chunks ALTER COLUMN embedding TYPE vector(768) USING NULL;")
            print("context_chunks migrated successfully.")
        except Exception as e:
            print(f"Error migrating context_chunks: {e}")

        # 3. nodes
        try:
            print("Migrating table: nodes...")
            cur.execute("ALTER TABLE nodes ALTER COLUMN summary_embedding TYPE vector(768) USING NULL;")
            print("nodes migrated successfully.")
        except Exception as e:
            print(f"Error migrating nodes: {e}")
            
        # 4. messages (if applicable)
        try:
             print("Migrating table: messages...")
             # Check if column exists first? Usually assumes standard schema.
             cur.execute("ALTER TABLE messages ALTER COLUMN embedding TYPE vector(768) USING NULL;")
             print("messages migrated successfully.")
        except Exception as e:
             # Messages might not have embedding column in all schemas or might fail
             print(f"Error migrating messages (might be expected): {e}")

        conn.close()
        print("\nMigration complete! You may need to re-upload files to regenerate proper embeddings.")

    except Exception as e:
        print(f"Critical connection error: {e}")

if __name__ == "__main__":
    migrate_vectors()
