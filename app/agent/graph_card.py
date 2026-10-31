"""
The board map: a chat's place on the canvas and why each thing it reads is connected.

The canvas already is the graph (parent links, desk_id, pile_id, edges); this only reads it. Only
what the chat can read is named, so a sibling's title can never reach the model through here.
Every title goes through `_label`: one line, no prompt delimiters, clipped.
"""
from __future__ import annotations

from app.agent.board_context import _label
from app.agent.reads import Reads

MAX_CARD_CHARS = 3000
_QUESTION_CHARS = 120


def _name(row: dict) -> str:
    data = row["data"] or {}
    return _label(data.get("title") or data.get("name") or data.get("fileName")) or "Untitled"


def _fork_question(cur, parent_id: str, fork_message_id: str | None, user_email: str) -> str:
    """The user's question at or just before the fork point."""
    if not fork_message_id:
        return ""
    cur.execute(
        "SELECT content FROM messages WHERE node_id = %s AND user_email = %s AND role = 'user' AND position <= "
        "(SELECT position FROM messages WHERE id = %s AND node_id = %s) ORDER BY position DESC LIMIT 1",
        (parent_id, user_email, fork_message_id, parent_id),
    )
    row = cur.fetchone()
    return _label(row["content"], _QUESTION_CHARS) if row else ""


def build_graph_card(store, reads: Reads, user_email: str) -> str:
    items = reads.readable()
    with store.tx() as cur:
        cur.execute(
            "SELECT id, data, parent_node_id, forked_from_message_id, ancestor_ids FROM nodes "
            "WHERE id = %s AND user_email = %s", (reads.chat_id, user_email),
        )
        me = cur.fetchone()
        if not me:
            return ""
        lineage_ids = list(me["ancestor_ids"] or [])
        wanted = list({i.card_id for i in items} | set(lineage_ids))
        cur.execute(
            "SELECT id, data, source_type, pile_id, parent_node_id, forked_from_message_id FROM nodes "
            "WHERE id = ANY(%s) AND user_email = %s AND deleted_at IS NULL", (wanted, user_email),
        )
        rows = {r["id"]: r for r in cur.fetchall()}
        pile_ids = [r["pile_id"] for r in rows.values() if r["pile_id"]]
        if pile_ids:
            cur.execute("SELECT id, data, source_type, pile_id, parent_node_id, forked_from_message_id FROM nodes "
                        "WHERE id = ANY(%s) AND user_email = %s", (pile_ids, user_email))
            rows.update({r["id"]: r for r in cur.fetchall()})
        via_ids = list({i.via for i in items} | {reads.chat_id})
        cur.execute(
            "SELECT from_node, to_node, data FROM edges WHERE to_node = ANY(%s) AND user_email = %s "
            "AND kind IN ('wire','answer') AND deleted_at IS NULL", (via_ids, user_email),
        )
        whys = {(e["from_node"], e["to_node"]): _label((e["data"] or {}).get("why")) for e in cur.fetchall()}

        lines = [f"YOU ARE IN: {_label((me['data'] or {}).get('name')) or 'this question'}"]
        child = me
        for depth, parent_id in enumerate(reversed(lineage_ids)):
            parent = rows.get(parent_id)
            if not parent:
                break
            question = _fork_question(cur, parent_id, child["forked_from_message_id"], user_email)
            word = "BRANCHED FROM" if depth == 0 else "WHICH BRANCHED FROM"
            line = f'{word}: {_name(parent)}' + (f', at "{question}"' if question else "")
            why = _label((child["data"] or {}).get("branchWhy"))
            lines.append(line + (f" ({why})" if why else ""))
            child = parent

    def describe(item, row, how) -> str:
        pile = rows.get(row["pile_id"]) if row["pile_id"] else None
        name = _name(pile) if pile else _name(row)
        if pile:
            count = sum(1 for i in items if i.via == item.via and rows.get(i.card_id, {}).get("pile_id") == pile["id"])
            name += f" ({count} documents)"
        if item.how == "answer":
            name = f"{name}'s answer"
        desc = _label((row["data"] or {}).get("desc"))
        why = whys.get((pile["id"] if pile else item.card_id, item.via)) or desc
        if not why:
            return f"  {name}"
        return f"  {name}: {why}" if how == "string" else f"  {name} ({why})"

    own_desk, strung, inherited = [], [], {}
    for item in items:
        row = rows.get(item.card_id)
        if not row:
            continue
        if item.inherited:
            inherited.setdefault(item.via, []).append(describe(item, row, "desk"))
        elif item.how == "desk":
            own_desk.append(describe(item, row, "desk"))
        else:
            strung.append(describe(item, row, "string"))

    def section(title: str, entries: list[str]) -> None:
        unique = list(dict.fromkeys(entries))
        if unique:
            lines.append(title)
            lines.extend(unique)

    section("YOUR DESK:", own_desk)
    section("CONNECTED BY THE USER:", sorted(strung, key=lambda s: "'s answer" in s))
    for chat_id, entries in inherited.items():
        owner = rows.get(chat_id)
        section(f"FROM {_name(owner).upper() if owner else 'ABOVE'}:", entries)

    if len(lines) == 1:
        return ""
    out, used = [], 0
    for index, line in enumerate(lines):
        if used + len(line) + 1 > MAX_CARD_CHARS - 40:
            out.append(f"...and {len(lines) - index} more not listed")
            break
        out.append(line)
        used += len(line) + 1
    return "\n".join(out)
