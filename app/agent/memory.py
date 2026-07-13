from __future__ import annotations

import re
from typing import Any, Iterable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


_INTERNAL_HEADER_RE = re.compile(
    r"^\s*(?:[#>*`\-\s]*)?(?:conversation summary|summary|base context|previous conversation summary|"
    r"memory facts|thread summary|relevant context|external context)\b[:\s-]*$",
    re.IGNORECASE,
)
_LEADING_SUMMARY_PREFIX_RE = re.compile(
    r"^\s*(?:[#>*`\-\s]*)?(?:conversation summary|summary)\s*:\s*",
    re.IGNORECASE,
)
_USER_NAME_PATTERNS = (
    re.compile(r"\bmy name is\s+([A-Za-z][\w.-]{1,39}(?:\s+[A-Za-z][\w.-]{1,39})?)", re.IGNORECASE),
    re.compile(r"\bcall me\s+([A-Za-z][\w.-]{1,39}(?:\s+[A-Za-z][\w.-]{1,39})?)", re.IGNORECASE),
    re.compile(r"\b[Ii](?:'?m| am)\s+([A-Z][\w.-]{1,39})"),
)
_ASSISTANT_NAME_PATTERNS = (
    re.compile(r"\bi(?: would|'d)? like to call you\s+([A-Za-z][\w.-]{1,39})", re.IGNORECASE),
    re.compile(r"\blet me call you\s+([A-Za-z][\w.-]{1,39})", re.IGNORECASE),
    re.compile(r"\bcall you\s+([A-Za-z][\w.-]{1,39})", re.IGNORECASE),
    re.compile(r"\byour name is\s+([A-Za-z][\w.-]{1,39})", re.IGNORECASE),
)
_DECISION_MARKERS = (
    "let's ",
    "lets ",
    "we should ",
    "we will ",
    "we'll ",
    "go with ",
    "stick with ",
    "use ",
    "please use ",
    "from now on ",
)
_CONSTRAINT_MARKERS = (
    "must ",
    "must not",
    "should not",
    "do not",
    "don't",
    "never ",
    "always ",
    "need to ",
    "only ",
)
_MAX_FACT_ITEMS = 6
_MAX_FACT_TEXT_CHARS = 180


def sanitize_summary_text(summary: Any) -> str:
    text = str(summary or "").replace("\r\n", "\n").strip()
    if not text:
        return ""

    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        if _INTERNAL_HEADER_RE.match(line):
            continue
        line = _LEADING_SUMMARY_PREFIX_RE.sub("", line)
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def sanitize_chat_response_text(content: Any) -> str:
    text = str(content or "").replace("\r\n", "\n").strip()
    if not text:
        return ""

    lines = text.splitlines()
    while lines and _INTERNAL_HEADER_RE.match(lines[0].strip()):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)

    cleaned = "\n".join(lines).strip()
    return cleaned or text


def extract_memory_facts(previous_facts: dict[str, Any] | None, messages: Iterable[Any]) -> dict[str, Any]:
    facts = _coerce_memory_facts(previous_facts)

    for role, text in _iter_messages(messages):
        if role != "user":
            continue

        user_name = _extract_first_match(_USER_NAME_PATTERNS, text)
        if user_name:
            facts["user_profile"]["name"] = user_name

        assistant_name = _extract_first_match(_ASSISTANT_NAME_PATTERNS, text)
        if assistant_name:
            facts["assistant_profile"]["preferred_name"] = assistant_name

        lowered = text.lower()
        if "?" not in text and any(marker in lowered for marker in _DECISION_MARKERS):
            _append_unique(facts["decisions"], _compact_fact_text(text))

        if any(marker in lowered for marker in _CONSTRAINT_MARKERS):
            _append_unique(facts["constraints"], _compact_fact_text(text))

    return _trim_memory_facts(facts)


def format_memory_facts_for_prompt(memory_facts: dict[str, Any] | None) -> str:
    facts = _coerce_memory_facts(memory_facts)
    lines: list[str] = []

    user_name = facts["user_profile"].get("name")
    if user_name:
        lines.append(f"- User name: {user_name}")

    assistant_name = facts["assistant_profile"].get("preferred_name")
    if assistant_name:
        lines.append(f"- Preferred assistant name: {assistant_name}")

    decisions = facts.get("decisions") or []
    if decisions:
        lines.append("Decisions:")
        lines.extend(f"- {item}" for item in decisions)

    constraints = facts.get("constraints") or []
    if constraints:
        lines.append("Constraints:")
        lines.extend(f"- {item}" for item in constraints)

    return "\n".join(lines).strip() or "None."


def build_memory_context_block(
    *,
    summary: str | None,
    memory_facts: dict[str, Any] | None,
    context: list[str] | None,
    external_context: str | None,
) -> str:
    clean_summary = sanitize_summary_text(summary)
    parts = [
        "Internal background context for continuity. Use only when relevant and never repeat the labels.",
        "<THREAD_SUMMARY>",
        clean_summary or "None.",
        "</THREAD_SUMMARY>",
        "<MEMORY_FACTS>",
        format_memory_facts_for_prompt(memory_facts),
        "</MEMORY_FACTS>",
    ]

    if context:
        parts.extend(
            [
                "<RELEVANT_CONTEXT>",
                "\n".join(str(item) for item in context if str(item).strip()) or "None.",
                "</RELEVANT_CONTEXT>",
            ]
        )

    if external_context:
        parts.extend(
            [
                "<EXTERNAL_CONTEXT>",
                str(external_context).strip(),
                "</EXTERNAL_CONTEXT>",
            ]
        )

    return "\n".join(parts)


def _iter_messages(messages: Iterable[Any]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for msg in messages or []:
        if isinstance(msg, HumanMessage):
            items.append(("user", str(msg.content or "").strip()))
            continue
        if isinstance(msg, AIMessage):
            items.append(("assistant", str(msg.content or "").strip()))
            continue
        if isinstance(msg, SystemMessage):
            items.append(("system", str(msg.content or "").strip()))
            continue
        if isinstance(msg, dict):
            role = str(msg.get("role") or "unknown").strip().lower()
            text = str(msg.get("text") or msg.get("content") or "").strip()
            items.append((role, text))
            continue
        items.append(("unknown", str(msg or "").strip()))
    return [(role, text) for role, text in items if text]


def _coerce_memory_facts(memory_facts: dict[str, Any] | None) -> dict[str, Any]:
    source = memory_facts if isinstance(memory_facts, dict) else {}
    return {
        "user_profile": {
            "name": _clean_fact_value(source.get("user_profile", {}).get("name") if isinstance(source.get("user_profile"), dict) else None),
        },
        "assistant_profile": {
            "preferred_name": _clean_fact_value(
                source.get("assistant_profile", {}).get("preferred_name")
                if isinstance(source.get("assistant_profile"), dict)
                else None
            ),
        },
        "decisions": _coerce_fact_list(source.get("decisions")),
        "constraints": _coerce_fact_list(source.get("constraints")),
    }


def _coerce_fact_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for entry in value:
        cleaned = _clean_fact_value(entry)
        if cleaned:
            _append_unique(items, cleaned)
    return items[:_MAX_FACT_ITEMS]


def _clean_fact_value(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _compact_fact_text(text: str) -> str:
    compact = " ".join(str(text or "").split()).strip()
    return compact[:_MAX_FACT_TEXT_CHARS].rstrip(" ,.;:") if len(compact) > _MAX_FACT_TEXT_CHARS else compact


def _append_unique(items: list[str], value: str | None) -> None:
    cleaned = _clean_fact_value(value)
    if not cleaned:
        return
    normalized = cleaned.casefold()
    if any(existing.casefold() == normalized for existing in items):
        return
    items.append(cleaned)


def _extract_first_match(patterns: Iterable[re.Pattern[str]], text: str) -> str | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(1).strip(" .,!?:;\"'")
    return None


def _trim_memory_facts(memory_facts: dict[str, Any]) -> dict[str, Any]:
    facts = _coerce_memory_facts(memory_facts)
    facts["decisions"] = facts["decisions"][:_MAX_FACT_ITEMS]
    facts["constraints"] = facts["constraints"][:_MAX_FACT_ITEMS]
    return facts
