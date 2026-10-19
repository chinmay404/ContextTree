"""Small text helpers shared by the V2 chat path and the V3 board engine."""


def clip_text(value, max_chars: int) -> str:
    """Strip ``value`` and cut it to ``max_chars`` characters, ending with an ellipsis if cut."""
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 1].rstrip()}…"
