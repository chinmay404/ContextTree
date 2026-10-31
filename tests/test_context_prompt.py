"""Context engine: the V3 prompt runs stable to volatile, and sources travel as data."""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.prompt_layout import build_v3_messages


def _build(provider="openai", **kw):
    args = dict(
        instructions="INSTRUCTIONS", lineage_summary="LINEAGE", persona="PERSONA", citation_note="CITE",
        graph_card="CARD", summary="OWN SUMMARY", facts_text="FACTS",
        history=[HumanMessage(content="old q", id="1"), AIMessage(content="old a", id="2"),
                 HumanMessage(content="new q", id="3")],
        context=["SIMILAR"], external_context="PDF TEXT", linked_answers="<LINKED_ANSWERS>x</LINKED_ANSWERS>",
        today="2026-10-31", provider=provider,
    )
    args.update(kw)
    return build_v3_messages(**args)


def _text(content):
    return content if isinstance(content, str) else "".join(p["text"] for p in content)


def test_sources_are_never_in_the_system_role():
    system = _text(_build()[0].content)
    for leaked in ("PDF TEXT", "SIMILAR", "LINKED_ANSWERS"):
        assert leaked not in system


def test_sources_and_question_travel_together_at_the_end():
    last = _build()[-1]
    assert isinstance(last, HumanMessage)
    text = _text(last.content)
    assert text.index("PDF TEXT") < text.index("SIMILAR") < text.index("new q")
    assert text.index("<LINKED_ANSWERS>") < text.index("new q")
    assert text.rstrip().endswith("new q")


def test_the_model_is_told_board_data_is_not_instructions():
    assert "BOARD_DATA" in _text(_build()[0].content)


def test_system_prompt_does_not_change_between_turns():
    one = _build(history=[HumanMessage(content="first", id="1")], today="2026-10-31")
    two = _build(history=[HumanMessage(content="second", id="1")], today="2026-11-01", external_context="OTHER")
    assert one[0].content == two[0].content


def test_only_one_system_message_and_history_keeps_its_order():
    messages = _build()
    assert sum(isinstance(m, SystemMessage) for m in messages) == 1
    assert [_text(m.content) for m in messages[1:3]] == ["old q", "old a"]


def test_what_siblings_share_comes_before_what_they_do_not():
    system = _text(_build()[0].content)
    assert system.index("INSTRUCTIONS") < system.index("LINEAGE") < system.index("PERSONA")
    assert system.index("LINEAGE") < system.index("CARD") < system.index("OWN SUMMARY")


def test_siblings_share_the_first_cache_block_when_lineage_matches():
    a = _build("anthropic", graph_card="CARD A", persona="ONE", summary="own a")
    b = _build("anthropic", graph_card="CARD B", persona="TWO", summary="own b")
    assert a[0].content[0] == b[0].content[0]
    assert a[0].content[0]["cache_control"] == {"type": "ephemeral"}


def test_second_cache_line_sits_on_the_last_past_message():
    messages = _build("anthropic")
    past = messages[-2].content
    assert past[-1]["cache_control"] == {"type": "ephemeral"}
    assert isinstance(messages[-1].content, str)


def test_other_providers_get_plain_strings():
    assert all(isinstance(m.content, str) for m in _build("openai"))


def test_anthropic_payload_is_accepted_by_the_real_formatter():
    from langchain_anthropic.chat_models import _format_messages

    system, messages = _format_messages(_build("anthropic"))
    assert [b.get("cache_control") for b in system] == [{"type": "ephemeral"}, None]
    assert any(isinstance(m["content"], list) and m["content"][-1].get("cache_control") for m in messages)


def test_empty_blocks_leave_no_empty_tags():
    messages = _build(lineage_summary="", persona=None, citation_note=None, graph_card="", summary="",
                      facts_text="None.", context=[], external_context=None, linked_answers=None)
    text = _text(messages[0].content) + _text(messages[-1].content)
    for tag in ("LINEAGE_SUMMARY", "BOARD_MAP", "THREAD_SUMMARY", "MEMORY_FACTS", "EXTERNAL_CONTEXT", "RELEVANT_CONTEXT"):
        assert tag not in text.replace("Never repeat", "") or tag in text.split("INSTRUCTIONS")[0]
    assert "<BOARD_DATA>" not in _text(messages[-1].content)


def test_a_turn_with_no_history_still_ends_on_the_question():
    messages = _build(history=[HumanMessage(content="only", id="1")])
    assert len(messages) == 2 and _text(messages[-1].content).endswith("only")


def test_history_objects_are_not_changed():
    history = [HumanMessage(content="old", id="1"), AIMessage(content="ans", id="2"), HumanMessage(content="q", id="3")]
    _build("anthropic", history=history)
    assert [m.content for m in history] == ["old", "ans", "q"]
