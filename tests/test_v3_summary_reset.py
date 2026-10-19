from langchain_core.messages import HumanMessage, AIMessage

from tests.v3_support import FakeLLM


def _state(n):
    msgs = [(HumanMessage if i % 2 == 0 else AIMessage)(content=f"m{i}", id=f"id{i}")
            for i in range(n)]
    return {"messages": msgs, "summary": "", "count": 11}


def _nodes():
    from app.agent.nodes.assistant_node import AgentNodes
    return AgentNodes(mongo_store=None)


def test_summurize_resets_count(monkeypatch):
    import app.agent.nodes.assistant_node as an
    monkeypatch.setattr(an, "get_llm", lambda *a, **k: FakeLLM("S"))
    out = _nodes().summurize(_state(10))
    assert out["count"] == 0 and out["summary"] == "S"


def test_decision_false_on_next_turn_after_summary(monkeypatch):
    import app.agent.nodes.assistant_node as an
    monkeypatch.setattr(an, "get_llm", lambda *a, **k: FakeLLM("S"))
    nodes = _nodes()
    st = _state(10)
    assert nodes.summury_decision(st) is True
    out = nodes.summurize(st)
    assert out["count"] == 0
    assert nodes.summury_decision({**st, **out}) is False


def test_decision_fires_once_per_window():
    nodes = _nodes()
    assert nodes.summury_decision({"count": 11}) is True
    assert [nodes.summury_decision({"count": c}) for c in range(0, 11)] == [False] * 11
