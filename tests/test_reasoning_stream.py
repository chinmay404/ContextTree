"""
Reasoning ("thinking") extraction — providers deliver it three ways; all
must surface as inline <think> framing for the UI's live Thinking block.
No network, no DB.
"""

from types import SimpleNamespace

from app.agent.main import _split_reasoning


def _chunk(content=None, **ak):
    return SimpleNamespace(content=content, additional_kwargs=ak or {})


def test_reasoning_content_field_is_separated():
    r, a = _split_reasoning(_chunk("the answer", reasoning_content="hmm let me think"))
    assert r == "hmm let me think"
    assert a == "the answer"


def test_reasoning_alias_field():
    r, a = _split_reasoning(_chunk("x", reasoning="pondering"))
    assert r == "pondering" and a == "x"


def test_gemini_thought_parts_split():
    r, a = _split_reasoning(
        _chunk([
            {"type": "text", "text": "step one", "thought": True},
            {"type": "text", "text": "final answer"},
        ])
    )
    assert r == "step one"
    assert a == "final answer"


def test_thinking_typed_parts_split():
    r, a = _split_reasoning(
        _chunk([{"type": "thinking", "thinking": "chain"}, {"type": "text", "text": "out"}])
    )
    assert r == "chain" and a == "out"


def test_plain_string_content_passthrough():
    r, a = _split_reasoning(_chunk("<think>inline</think>done"))
    assert r == ""
    assert a == "<think>inline</think>done"  # UI already parses inline tags


def test_empty_chunk_is_empty():
    r, a = _split_reasoning(_chunk(None))
    assert r == "" and a == ""
