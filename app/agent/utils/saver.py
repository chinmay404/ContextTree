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
        redis_url = (
            os.getenv("REDIS_URL") or
            os.getenv("REDIS_HOST")
        )
        saver_cm = RedisSaver.from_conn_string(redis_url)
        saver = saver_cm.__enter__()
        saver.setup()
        return saver
    except Exception as e:
        print("Error initializing RedisSaver:", e)
        return None
