from langgraph.checkpoint.redis import RedisSaver
import os
from dotenv import load_dotenv
import os

load_dotenv() 


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
