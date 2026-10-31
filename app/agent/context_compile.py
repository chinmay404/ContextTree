"""
Steps 5 to 7 of a V3 turn: rank what was gathered, go deeper if it is not enough, fit the budget.

Code owns scope and the budget, Jev owns the small judgements, and every Jev or agent failure
leaves the search order in charge. `trace` records what happened for the context inspector.
"""
from __future__ import annotations

from dataclasses import replace

from app.agent.board_context import (
    BoardContext, NumberedContext, fit_to_budget, load_outlines, number_passages,
)
from app.agent.context_agent import BoardTools, explore
from app.agent.context_judge import judge_context, rank_chunks
from app.agent.helpers import jev
from app.agent.helpers.v3_llm import get_v3_llm, turn_key_from, v3_route
from app.agent.reads import Reads
from app.agent.store.board_store import store as board_store
from app.core.config import settings


def _estimate(fit, fixed_chars: int) -> int:
    sent = sum(len(c.text) for c in fit.chunks)
    sent += sum(len(p[1]) + len(p[2]) for p in fit.ctx.note_parts)
    sent += sum(sum(map(len, p[1])) + len(p[2]) for p in fit.ctx.answer_parts)
    return (fixed_chars + sent) // 4


def compile_board_context(ctx: BoardContext, reads: Reads, question: str, graph_card: str, user_email: str,
                          top_k: int, budget_tokens: int | None, fixed_chars: int, config: dict | None,
                          trace: dict | None, recent: str = "") -> NumberedContext:
    outlines = None
    if jev.enabled() and ctx.chunks:
        with board_store.tx() as cur:
            outlines = load_outlines(cur, user_email, list({c.card_id for c in ctx.chunks}))
    judgement = judge_context(question, graph_card, ctx, ctx.chunks, outlines, recent)
    scores = dict(judgement.scores) if judgement.scores is not None else None

    turn_key = turn_key_from(config)
    model = ((config or {}).get("configurable") or {}).get("model")
    wants_agent = (settings.CONTEXT_AGENT_ENABLED and jev.enabled() and judgement.enough is False
                   and bool(reads.readable()) and turn_key is not None and bool(model))
    found, steps, agent_failed = [], [], False
    if wants_agent:
        llm = get_v3_llm(model, None, resolved=turn_key.resolved())
        outcome = explore(llm, question, graph_card, BoardTools(board_store, reads, user_email), v3_route(model).provider)
        found, steps, agent_failed = outcome.chunks, outcome.steps, outcome.failed
        ctx = replace(ctx, headers={**outcome.headers, **ctx.headers})
        if scores is not None:
            scores.update({(c.card_id, c.loc): c.score for c in found})

    # A string the agent can't go and open keeps its best passage instead (a model with no tools).
    agent_can_open = settings.CONTEXT_AGENT_ENABLED and jev.enabled() and not agent_failed
    protect = None if agent_can_open else {i.card_id for i in reads.readable() if i.how == "wire"}
    already = {(c.card_id, c.loc, c.text) for c in found}
    ranked, dropped = rank_chunks([c for c in ctx.chunks if (c.card_id, c.loc, c.text) not in already],
                                  scores, top_k, protect)
    fit = fit_to_budget(ctx, found + ranked, fixed_chars, budget_tokens, scores)
    numbered = number_passages(fit.ctx, fit.chunks)

    if trace is not None:
        trace.update({
            "jev": judgement.status,
            "enough": judgement.enough,
            "sideQuestion": judgement.side,
            "budgetTokens": budget_tokens,
            "estimatedTokens": _estimate(fit, fixed_chars),
            "sent": [{"n": p["n"], "cardId": p["cardId"], "loc": p["loc"],
                      "score": scores.get((p["cardId"], p["loc"])) if scores else None}
                     for p in numbered.passages.values()],
            "dropped": dropped + fit.dropped,
            "agent": {"ran": wants_agent, "failed": agent_failed, "steps": steps},
        })
    return numbered
