from __future__ import annotations

from typing import Any, Iterable


_BASE_TAGS = ("contexttree", "api-service")


def build_langsmith_trace_config(
    run_name: str,
    *,
    conversation_id: str | None = None,
    user_id: str | None = None,
    message_id: str | None = None,
    model_name: str | None = None,
    transport: str | None = None,
    node_name: str | None = None,
    parent_thread_id: str | None = None,
    fork_at_message_id: str | None = None,
    tags: Iterable[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trace_tags = list(_BASE_TAGS)

    for value in (transport, node_name):
        if value:
            trace_tags.append(value)

    if parent_thread_id:
        trace_tags.append("fork")

    if tags:
        trace_tags.extend(str(tag) for tag in tags if tag)

    trace_metadata: dict[str, Any] = {
        "thread_id": conversation_id,
        "conversation_id": conversation_id,
        "user_id": user_id,
        "message_id": message_id,
        "model_name": model_name,
        "transport": transport,
        "node_name": node_name,
        "parent_thread_id": parent_thread_id,
        "fork_at_message_id": fork_at_message_id,
    }
    if metadata:
        trace_metadata.update(metadata)

    trace_metadata = {
        key: value
        for key, value in trace_metadata.items()
        if value is not None and value != ""
    }

    return {
        "run_name": run_name,
        "tags": trace_tags,
        "metadata": trace_metadata,
    }
