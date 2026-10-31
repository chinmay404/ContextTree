"""What each V3 turn sent and left out, for the context inspector and for evals."""
from __future__ import annotations

from typing import Optional

from app.agent.store.board_store import dumps_json

KEEP_PER_CHAT = 20


def save_trace(store, node_id: str, user_email: str, message_id: Optional[str], data: dict) -> None:
    with store.tx() as cur:
        cur.execute(
            "INSERT INTO context_traces (node_id, user_email, message_id, data) VALUES (%s,%s,%s,%s::jsonb)",
            (node_id, user_email, message_id, dumps_json(data)),
        )
        cur.execute(
            "DELETE FROM context_traces WHERE node_id = %s AND id NOT IN "
            "(SELECT id FROM context_traces WHERE node_id = %s ORDER BY id DESC LIMIT %s)",
            (node_id, node_id, KEEP_PER_CHAT),
        )


def latest_trace(store, node_id: str, user_email: str) -> Optional[dict]:
    with store.tx() as cur:
        cur.execute(
            "SELECT message_id, data FROM context_traces WHERE node_id = %s AND user_email = %s "
            "ORDER BY id DESC LIMIT 1",
            (node_id, user_email),
        )
        row = cur.fetchone()
    return {"messageId": row["message_id"], **row["data"]} if row else None
