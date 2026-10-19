"""Small text helpers shared by the V2 chat path and the V3 board engine."""


def clip_text(text, limit: int) -> str:
    """Strip ``text`` and cut it to ``limit`` characters, ending with an ellipsis if cut."""
    out = str(text or "").strip()
    if len(out) <= limit:
        return out
    return f"{out[: limit - 1].rstrip()}…"
