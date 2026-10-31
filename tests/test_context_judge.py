"""Context engine: Jev scores what a chat reads; any Jev failure leaves the search order in charge."""
import json

import httpx
import pytest

from app.agent.board_context import BoardContext, ScoredChunk
from app.agent.context_judge import judge_context, rank_chunks
from app.agent.helpers import jev
from app.core.config import settings


def _chunk(card, loc, text="text", score=0.03):
    return ScoredChunk(card, loc, text, score)


def _ctx(notes=(), answers=(), chunks=(), headers=None):
    ctx = BoardContext()
    ctx.chunks = list(chunks)
    ctx.headers = dict(headers or {})
    for cid, body in notes:
        ctx.note_parts.append((cid, f"[source | Note | {cid}]", body, body))
    for cid, body in answers:
        ctx.answer_parts.append((cid, [f"[answer | {cid}]"], body, body))
    return ctx


class Wire:
    """Answers like the real endpoint; the test says which probability each question gets."""

    def __init__(self, answers=None, status=200, boom=None):
        self.answers, self.status, self.boom = answers or {}, status, boom
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.boom:
            raise self.boom
        body = json.loads(request.content)
        self.requests.append((request, body))
        out = {qid: {"type": "noul", "noul": self.answers.get(qid, 0.5)} for qid in body["questions"]}
        return httpx.Response(self.status, json={"model": body["model"], "answers": out,
                                                 "usage": {"input_tokens": 10, "output_tokens": 0}})


@pytest.fixture
def wire(monkeypatch):
    w = Wire()
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "secret-key")
    monkeypatch.setattr(jev, "_transport", lambda: httpx.MockTransport(w))
    return w


def test_no_key_means_jev_is_off_and_no_request_is_made(monkeypatch):
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", None)
    def forbidden():
        raise AssertionError("no request without a key")
    monkeypatch.setattr(jev, "_transport", forbidden)
    j = judge_context("q", "card", _ctx(chunks=[_chunk("a", "1")]), [_chunk("a", "1")])
    assert j.status == "off" and j.scores is None and j.enough is None


def test_request_matches_the_wire_contract(wire):
    judge_context("the question", "the card", _ctx(chunks=[_chunk("a", "1", "hello")]), [_chunk("a", "1", "hello")])
    request, body = wire.requests[0]
    assert str(request.url).endswith("/v1/systemone")
    assert request.headers["authorization"] == "Bearer secret-key"
    assert body["model"] == settings.JEV_MODEL
    assert all(q["type"] == "noul" and q["instructions"] for q in body["questions"].values())
    state = json.loads(body["state"])
    assert state["question"] == "the question" and state["board_map"] == "the card"
    assert "hello" in json.dumps(state["passages"])


def test_passage_score_blends_passage_and_source(wire):
    wire.answers = {"p0": 0.9, "p1": 0.9, "s0": 1.0, "s1": 0.25}
    chunks = [_chunk("good", "1"), _chunk("weak", "1")]
    j = judge_context("q", "card", _ctx(chunks=chunks), chunks)
    assert j.scores[("good", "1")] == pytest.approx(0.9 ** 0.5)
    assert j.scores[("weak", "1")] == pytest.approx((0.9 * 0.25) ** 0.5)


def test_notes_and_answers_get_scores_too(wire):
    wire.answers = {"n0": 0.8, "a0": 0.3}
    j = judge_context("q", "card", _ctx(notes=[("note", "body")], answers=[("chat", "body")]), [])
    assert j.scores[("note", None)] == 0.8 and j.scores[("chat", None)] == 0.3


def test_source_question_sees_content_not_only_a_title(wire):
    chunks = [_chunk("a", "1", "the best passage text")]
    judge_context("q", "card", _ctx(chunks=chunks, headers={"a": "[source | PDF | Report]"}), chunks,
                  outlines={"a": ["p. 1: Pricing"]})
    state = json.loads(wire.requests[0][1]["state"])
    source = state["sources"]["s0"]
    assert source["title"] == "[source | PDF | Report]"
    assert source["sections"] == ["p. 1: Pricing"] and "best passage" in source["best_passage"]


@pytest.mark.parametrize("value, expected", [(0.1, False), (0.35, False), (0.5, None), (0.65, True), (0.9, True)])
def test_enough_only_says_no_when_sure(wire, value, expected):
    wire.answers = {"enough": value}
    assert judge_context("q", "card", _ctx(notes=[("n", "b")]), []).enough is expected


@pytest.mark.parametrize("failure", [
    {"status": 429}, {"status": 529}, {"status": 401}, {"boom": httpx.ReadTimeout("slow")},
    {"boom": httpx.ConnectError("down")},
])
def test_any_failure_falls_back_without_raising(monkeypatch, failure):
    w = Wire(**failure)
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "secret-key")
    monkeypatch.setattr(jev, "_transport", lambda: httpx.MockTransport(w))
    chunks = [_chunk("a", "1")]
    j = judge_context("q", "card", _ctx(chunks=chunks), chunks)
    assert j.status == "error" and j.scores is None and j.enough is None


def test_malformed_answer_falls_back(monkeypatch):
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "secret-key")
    monkeypatch.setattr(jev, "_transport", lambda: httpx.MockTransport(
        lambda request: httpx.Response(200, json={"answers": {"p0": {"type": "noul", "noul": "yes"}}})))
    chunks = [_chunk("a", "1")]
    assert judge_context("q", "card", _ctx(chunks=chunks), chunks).status == "error"


def test_the_key_is_never_in_an_error(monkeypatch, caplog):
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "secret-key")
    monkeypatch.setattr(jev, "_transport", lambda: httpx.MockTransport(Wire(boom=httpx.ConnectError("secret-key"))))
    judge_context("q", "card", _ctx(notes=[("n", "b")]), [])
    assert "secret-key" not in caplog.text


def test_state_stays_inside_jev_input_limit(wire):
    chunks = [_chunk("a", str(i), "x" * 5000) for i in range(60)]
    judge_context("q", "card", _ctx(chunks=chunks), chunks)
    assert len(wire.requests[0][1]["state"]) < jev.MAX_STATE_CHARS


def test_empty_context_asks_nothing(wire):
    j = judge_context("q", "card", _ctx(), [])
    assert wire.requests == [] and j.status == "skipped" and j.enough is False


def test_rank_orders_by_score_and_drops_only_sure_misses():
    chunks = [_chunk("a", "1"), _chunk("a", "2"), _chunk("b", "1")]
    ranked, dropped = rank_chunks(chunks, {("a", "1"): 0.3, ("a", "2"): 0.9, ("b", "1"): 0.1}, per_source=5)
    assert [(c.card_id, c.loc) for c in ranked] == [("a", "2"), ("a", "1")]
    assert dropped == [{"cardId": "b", "loc": "1", "score": 0.1, "reason": "jev"}]


def test_rank_keeps_per_source_limit():
    chunks = [_chunk("a", str(i)) for i in range(4)]
    ranked, dropped = rank_chunks(chunks, {("a", str(i)): 0.9 - i / 10 for i in range(4)}, per_source=2)
    assert [c.loc for c in ranked] == ["0", "1"]
    assert {d["loc"] for d in dropped} == {"2", "3"} and all(d["reason"] == "per_source" for d in dropped)


def test_rank_can_protect_a_string_when_the_agent_cannot_open_it():
    chunks = [_chunk("wired", "1"), _chunk("wired", "2")]
    ranked, _ = rank_chunks(chunks, {("wired", "1"): 0.2, ("wired", "2"): 0.1}, per_source=5, protect={"wired"})
    assert [c.loc for c in ranked] == ["1"]


def test_without_scores_search_order_stands():
    chunks = [_chunk("a", str(i)) for i in range(3)]
    ranked, dropped = rank_chunks(chunks, None, per_source=2)
    assert [c.loc for c in ranked] == ["0", "1"] and dropped[0]["reason"] == "per_source"
