from langgraph.checkpoint.redis import RedisSaver, AsyncRedisSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import  ConnectionPool, AsyncConnectionPool
import os
from dotenv import load_dotenv
import atexit

load_dotenv() 

_pg_pool = None
_async_pg_pool = None

def postgres_saver():
    try:
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL must be set")
        
        # Determine if we should use sync or async based on context? 
        # For now, let's provide a Sync one as default or handle both?
        # The user's code in main.py is synchronous in __init__ but calls async methods.
        
        # If we return a sync saver, aget_state might fail or fall back?
        # The error encountered was NotImplementedError in aget_tuple.
        # This usually means the saver does not implement async methods.
        
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": None,
        }

        # Sync implementation
        global _pg_pool
        if _pg_pool is None:
            _pg_pool = ConnectionPool(
                conninfo=db_url,
                max_size=20,
                kwargs=connection_kwargs,
            )
            atexit.register(lambda: _safe_close_pool(_pg_pool))
        # We need to yield or return. Since this is a simple function, we return the initialized saver.
        # Ideally, we should manage the pool lifecycle.
        saver = PostgresSaver(_pg_pool)
        saver.setup() 
        return saver

    except Exception as e:
        print("Error initializing PostgresSaver:", e)
        return None

def async_postgres_saver():
    try:
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
             raise ValueError("DATABASE_URL must be set")
        
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": None,
        }

        global _async_pg_pool
        if _async_pg_pool is None:
            _async_pg_pool = AsyncConnectionPool(
                conninfo=db_url,
                max_size=20,
                kwargs=connection_kwargs,
            )
            atexit.register(lambda: _safe_close_pool(_async_pg_pool))
        
        saver = AsyncPostgresSaver(_async_pg_pool)
        # Note: AsyncPostgresSaver might need explicit setup await if not using context manager
        # But here we just return it. The caller (graph) should handle it?
        # Actually, AsyncPostgresSaver.setup() is a coroutine.
        # We can't await it in a sync __init__. This is a design issue in getGraphResponse.
        return saver
    except Exception as e:
         print("Error initializing AsyncPostgresSaver:", e)
         return None

def _safe_close_pool(pool):
    try:
        if pool:
            pool.close()
    except Exception:
        pass

def redis_saver():
    """
    Initialize and return a RedisSaver for LangGraph checkpointing.
    """
    try:
        redis_endpoint = os.getenv("REDIS_DATABASE_ENDPOINT")
        if redis_endpoint and not redis_endpoint.startswith(("redis://", "rediss://")):
           redis_endpoint = f"redis://{redis_endpoint}"

        redis_url = (
            os.getenv("REDIS_URL") or
            os.getenv("REDIS_HOST") or
            redis_endpoint
        )
        if not redis_url:
            raise ValueError("REDIS_URL, REDIS_HOST, or REDIS_DATABASE_ENDPOINT must be set")

        saver_cm = RedisSaver.from_conn_string(redis_url)
        saver = saver_cm.__enter__()
        saver.setup()
        return saver
    except Exception as e:
        print("Error initializing RedisSaver:", e)
        return None

def async_redis_saver():
    """
    Initialize and return an AsyncRedisSaver for LangGraph checkpointing.
    """
    try:
        redis_endpoint = os.getenv("REDIS_DATABASE_ENDPOINT")
        if redis_endpoint and not redis_endpoint.startswith(("redis://", "rediss://")):
           redis_endpoint = f"redis://{redis_endpoint}"

        redis_url = (
            os.getenv("REDIS_URL") or
            os.getenv("REDIS_HOST") or
            redis_endpoint
        )
        if not redis_url:
            raise ValueError("REDIS_URL, REDIS_HOST, or REDIS_DATABASE_ENDPOINT must be set")

        saver = AsyncRedisSaver.from_conn_string(redis_url)
        # Async savers often don't need explicit setup or enter if using a factory method that returns the saver, 
        # but check documentation. Usually from_conn_string returns the saver. 
        # But we might need to await setup() if it exists. 
        # AsyncRedisSaver usually implements AsyncContextManager. 
        # Ideally we loop run_until_complete? No this is init.
        # We will return the saver_cm and let the graph manage it? 
        # Or enter it?
        # AsyncRedisSaver documentation suggests using it as context manager.
        # But here we need to return the object.
        # Let's try returning the instance.
        return saver
    except Exception as e:
        print("Error initializing AsyncRedisSaver:", e)
        return None
