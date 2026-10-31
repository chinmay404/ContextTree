"""
After a chat's first answer: one line on why each connection is there, and a name if it has none.

An LLM writes these (Jev cannot write). It runs once per chat, on the chat's own model and the
user's key, after the answer. What the user typed always wins and is never overwritten, and any
failure leaves everything as it was: this is a nicety, never worth a failed turn.
"""
from __future__ import annotations

import json
import re

from app.agent.board_context import _label
from app.agent.helpers import jev
from app.agent.helpers.v3_llm import invoke_on_user_key
from app.agent.store.board_store import dumps_json
from app.agent.utils.text import clip_text
from app.core.logger import logger

WHY_CHARS = 120
NAME_CHARS = 60
_EXCERPT_CHARS = 1500
_FENCE = re.compile(r"^```[a-z]*\s*|\s*```$", re.I)


def parse_reply(text) -> dict:
    """{key: clean line}, with `names` kept as a list; {} for anything that isn't a JSON object."""
    try:
        data = json.loads(_FENCE.sub("", str(text or "").strip()))
    except ValueError:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict = {}
    for key, value in data.items():
        if key == "names":
            if isinstance(value, list):
                out["names"] = [n for n in (_label(v, NAME_CHARS) for v in value if isinstance(v, str)) if n]
        elif isinstance(value, str) and _label(value, WHY_CHARS):
            out[key] = _label(value, WHY_CHARS)
    return out


def pick_name(question: str, answer: str, names: list[str]) -> str | None:
    """Jev chooses among the names it is given; it never makes one up. Off or failing: the first."""
    if not names:
        return None
    if len(names) > 1 and jev.enabled():
        answers = jev.ask_nouls(
            {"question": clip_text(question, 600), "answer": clip_text(answer, 1200),
             "names": {f"n{i}": name for i, name in enumerate(names)}},
            {f"n{i}": f"Is name n{i} a good short title for this conversation?" for i in range(len(names))},
        )
        if answers:
            return names[max(range(len(names)), key=lambda i: answers[f"n{i}"])]
    return names[0]


def write_relations(store, user_email: str, chat_id: str, llm, provider: str, question: str, answer: str) -> dict | None:
    """{"name": suggestion} when the chat needs a name, otherwise None."""
    try:
        return _write(store, user_email, chat_id, llm, provider, question, answer)
    except Exception as e:  # noqa: BLE001 - a refused key included: the answer already reached the user
        logger.warning("Relations not written for %s: %s", chat_id, type(e).__name__)
        return None


def _write(store, user_email, chat_id, llm, provider, question, answer):
    with store.tx() as cur:
        cur.execute("SELECT count(*) AS n FROM messages WHERE node_id = %s AND user_email = %s "
                    "AND role = 'assistant' AND NOT inherited", (chat_id, user_email))
        if cur.fetchone()["n"] != 1:
            return None
        cur.execute("SELECT data, parent_node_id FROM nodes WHERE id = %s AND user_email = %s", (chat_id, user_email))
        chat = cur.fetchone()
        if not chat:
            return None
        cur.execute(
            "SELECT e.id, e.kind, e.data AS edata, n.data AS ndata FROM edges e JOIN nodes n ON n.id = e.from_node "
            "WHERE e.to_node = %s AND e.user_email = %s AND e.kind IN ('wire','answer') AND e.deleted_at IS NULL "
            "AND n.deleted_at IS NULL ORDER BY e.created_at, e.id", (chat_id, user_email),
        )
        edges = [e for e in cur.fetchall() if not (e["edata"] or {}).get("why")]
        parent_name = ""
        if chat["parent_node_id"]:
            cur.execute("SELECT data FROM nodes WHERE id = %s AND user_email = %s", (chat["parent_node_id"], user_email))
            parent = cur.fetchone()
            parent_name = _label(((parent or {}).get("data") or {}).get("name"))

    data = chat["data"] or {}
    needs_branch = bool(chat["parent_node_id"]) and not data.get("branchWhy")
    needs_name = not _label(data.get("name"))
    if not (edges or needs_branch or needs_name):
        return None

    asks = []
    keys = {}
    for i, e in enumerate(edges, 1):
        source = e["ndata"] or {}
        keys[f"e{i}"] = e["id"]
        title = _label(source.get("title") or source.get("name") or source.get("fileName")) or "Untitled"
        desc = _label(source.get("desc"))
        asks.append(f'"e{i}": why "{title}"{f" ({desc})" if desc else ""} is connected to this question')
    if needs_branch:
        persona = _label(data.get("persona"), 200)
        asks.append(f'"branch": why this question branched from "{parent_name or "its parent"}"'
                    + (f', answering as {persona}' if persona else ""))
    if needs_name:
        asks.append('"names": a list of 3 short titles for this conversation')
    prompt = ("Reply with one JSON object and nothing else. Each value is one plain line of at most 12 words.\n"
              "Keys:\n" + "\n".join(asks)
              + f"\n\nThe user asked:\n{clip_text(question, _EXCERPT_CHARS)}\n\nThe answer was:\n{clip_text(answer, _EXCERPT_CHARS)}")
    reply = parse_reply(getattr(invoke_on_user_key(llm, prompt, provider), "content", ""))

    with store.tx() as cur:
        for key, edge_id in keys.items():
            if reply.get(key):
                cur.execute(
                    "UPDATE edges SET data = COALESCE(data, '{}'::jsonb) || %s::jsonb WHERE id = %s AND user_email = %s "
                    "AND COALESCE(data->>'why', '') = ''",
                    (dumps_json({"why": reply[key], "whyBy": "llm"}), edge_id, user_email),
                )
        if needs_branch and reply.get("branch"):
            cur.execute(
                "UPDATE nodes SET data = COALESCE(data, '{}'::jsonb) || %s::jsonb WHERE id = %s AND user_email = %s "
                "AND COALESCE(data->>'branchWhy', '') = ''",
                (dumps_json({"branchWhy": reply["branch"], "branchWhyBy": "llm"}), chat_id, user_email),
            )
    name = pick_name(question, answer, reply.get("names") or []) if needs_name else None
    return {"name": name} if name else None
