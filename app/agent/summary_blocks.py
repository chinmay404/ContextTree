"""
Frozen summary blocks: what a branch inherits, made once and shared.

A block summarises one chat's messages from_position..to_position and never changes. A fork reads
the blocks that end at or before its fork point, so it cannot see later messages, and only the part
no block covers yet costs an LLM call. Boundaries fall on every BLOCK_MESSAGES-th message, so forks
at nearby messages reuse the same blocks and their prompts share a cacheable prefix.
"""
from __future__ import annotations

from typing import Callable

BLOCK_MESSAGES = 8
MAX_BLOCKS = 6  # more than this and the oldest merge into one, so a long chat's lineage stays small


def _stored(store, user_email: str, node_id: str, first: int, last: int) -> dict[int, list[tuple[int, str]]]:
    with store.tx() as cur:
        cur.execute(
            "SELECT from_position, to_position, text FROM summary_blocks "
            "WHERE node_id = %s AND user_email = %s AND from_position >= %s AND to_position <= %s",
            (node_id, user_email, first, last),
        )
        rows = cur.fetchall()
    out: dict[int, list[tuple[int, str]]] = {}
    for r in rows:
        out.setdefault(r["from_position"], []).append((r["to_position"], r["text"]))
    return out


def _save(store, user_email: str, node_id: str, first: int, last: int, text: str) -> None:
    with store.tx() as cur:  # two racing forks may both pay for a block; the first one stored wins
        cur.execute(
            "INSERT INTO summary_blocks (node_id, user_email, from_position, to_position, text) "
            "SELECT %s, %s, %s, %s, %s WHERE EXISTS (SELECT 1 FROM nodes WHERE id = %s AND user_email = %s) "
            "ON CONFLICT (node_id, from_position, to_position) DO NOTHING",
            (node_id, user_email, first, last, text, node_id, user_email),
        )


def cover_messages(store, user_email: str, node_id: str, messages: list[dict],
                   summarize: Callable[..., str]) -> str:
    """The summary of `messages` (all of a chat's rows, oldest first, up to some point).

    `summarize(rows)` writes a new block; `summarize([], previous=text)` merges older blocks.
    """
    if not messages:
        return ""
    positions = [m["position"] for m in messages]
    index_of = {p: i for i, p in enumerate(positions)}
    stored = _stored(store, user_email, node_id, positions[0], positions[-1])
    blocks: list[tuple[int, int, str]] = []
    i = 0
    while i < len(messages):
        options = stored.get(positions[i])
        if options:
            to_position, text = max(options)
            blocks.append((positions[i], to_position, text))
            i = index_of[to_position] + 1
            continue
        next_start = next((j for j in range(i + 1, len(messages)) if positions[j] in stored), len(messages))
        boundary = next_start // BLOCK_MESSAGES * BLOCK_MESSAGES
        end = boundary if boundary > i else next_start
        rows = messages[i:end]
        text = summarize(rows)
        _save(store, user_email, node_id, rows[0]["position"], rows[-1]["position"], text)
        blocks.append((rows[0]["position"], rows[-1]["position"], text))
        i = end

    if len(blocks) > MAX_BLOCKS:
        old = blocks[:-MAX_BLOCKS + 1]
        merged = summarize([], previous="\n\n".join(b[2] for b in old))
        _save(store, user_email, node_id, old[0][0], old[-1][1], merged)
        blocks = [(old[0][0], old[-1][1], merged)] + blocks[-MAX_BLOCKS + 1:]
    return "\n\n".join(b[2] for b in blocks)
