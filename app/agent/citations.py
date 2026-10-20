"""
V3 citations (contract amendment 2026-10-20).

Every passage a V3 turn puts in the prompt is numbered ``[n]``; the model is told to cite with
those markers. After the stream ends, ``extract_citations`` reads the markers the answer used
and maps them back to the passages that were sent.
"""
from __future__ import annotations

import re

QUOTE_MAX_CHARS = 1200

# Sent as its own system message only when a turn numbered at least one passage.
# The fake LLM keys on this exact text, so keep it a constant.
CITE_INSTRUCTION = (
    "The board passages above are numbered [1], [2], and so on. When a claim uses one, put its "
    "number right after that claim. Use only numbers that exist, write [2][3] rather than [2, 3], "
    "and add no list of references at the end."
)

_MARK = re.compile(r"\[\s*(\d{1,4}(?:\s*,\s*\d{1,4})*)\s*\]")
_THINK = re.compile(r"<think>.*?(?:</think>|\Z)", re.S)
_FENCE_OPEN = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_INLINE_CODE = re.compile(r"(`+)(?!`).+?(?<!`)\1(?!`)", re.S)


def _blank(m: re.Match) -> str:
    return re.sub(r"[^\n]", " ", m.group(0))


def _mask(text: str) -> str:
    """Same length as ``text``, with reasoning, fenced code and inline code blanked out."""
    text = _THINK.sub(_blank, text)
    lines = text.split("\n")
    fence: str | None = None
    for i, line in enumerate(lines):
        if fence is None:
            m = _FENCE_OPEN.match(line)
            if m:
                fence = m.group(1)
                lines[i] = " " * len(line)
            continue
        s = line.strip()
        if s and set(s) == {fence[0]} and len(s) >= len(fence):
            fence = None
        lines[i] = " " * len(line)
    return _INLINE_CODE.sub(_blank, "\n".join(lines))


def _markers(text: str):
    """(match, [numbers]) for each marker outside code and reasoning."""
    for m in _MARK.finditer(_mask(text)):
        yield m, [int(p) for p in m.group(1).split(",")]


def extract_citations(answer: str, passages: dict[int, dict]) -> list[dict]:
    """The sent passages the answer cites, by n. Unknown numbers are dropped; ``[2, 3]`` counts both."""
    used = {n for _, ns in _markers(answer or "") for n in ns if n in passages}
    return [dict(passages[n]) for n in sorted(used)]


def strip_markers(text: str) -> str:
    """Remove citation markers (outside code) — another chat's markers mean nothing in this turn."""
    out, last = [], 0
    for m, _ in _markers(text):
        out.append(text[last:m.start()])
        last = m.end()
    out.append(text[last:])
    return "".join(out)
