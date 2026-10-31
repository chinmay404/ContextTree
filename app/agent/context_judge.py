"""
Step 5 of a turn: Jev scores what the chat could send, and says whether it is enough to answer.

Scores only rank. A note or a linked answer is never dropped on a score, and a passage is dropped
only when Jev read its text and was sure it does not help; a source is never judged by its title
alone, because a judge that cannot see the content reads that as a confident no.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from app.agent.board_context import BoardContext, ScoredChunk
from app.agent.helpers import jev
from app.agent.utils.text import clip_text

PASSAGE_CHARS = 900
QUESTION_CHARS = 2000
CARD_CHARS = 3000
MAX_SECTIONS = 30
DROP_BELOW = 0.25  # "sure it does not help"; anything above stays in the running
ENOUGH_NO = 0.35   # the agent spends the user's key, so only a clear no sends it exploring
ENOUGH_YES = 0.65


@dataclass
class Judgement:
    status: str  # ok | off | error | skipped
    scores: dict[tuple[str, str | None], float] | None = None
    enough: bool | None = None


def judge_context(question: str, graph_card: str, ctx: BoardContext, chunks: list[ScoredChunk],
                  outlines: dict[str, list[str]] | None = None) -> Judgement:
    if not (chunks or ctx.note_parts or ctx.answer_parts):
        return Judgement("skipped", enough=False)  # nothing to read: only exploring could help
    if not jev.enabled():
        return Judgement("off")

    passages: dict[str, str] = {}
    keys: dict[str, tuple[str, str | None]] = {}
    for i, c in enumerate(chunks):
        passages[f"p{i}"], keys[f"p{i}"] = clip_text(c.text, PASSAGE_CHARS), (c.card_id, c.loc)
    for i, part in enumerate(ctx.note_parts):
        passages[f"n{i}"], keys[f"n{i}"] = clip_text(part[2], PASSAGE_CHARS), (part[0], None)
    for i, part in enumerate(ctx.answer_parts):
        passages[f"a{i}"], keys[f"a{i}"] = clip_text(part[2], PASSAGE_CHARS), (part[0], None)

    source_ids: dict[str, str] = {}
    sources: dict[str, dict] = {}
    for c in chunks:  # best-first order, so the first seen passage of a card is its best
        if c.card_id not in source_ids.values():
            sid = f"s{len(source_ids)}"
            source_ids[sid] = c.card_id
            sources[sid] = {"title": ctx.headers.get(c.card_id) or c.card_id,
                            "sections": (outlines or {}).get(c.card_id, [])[:MAX_SECTIONS],
                            "best_passage": clip_text(c.text, PASSAGE_CHARS)}

    state = {"question": clip_text(question, QUESTION_CHARS), "board_map": clip_text(graph_card or "", CARD_CHARS),
             "passages": passages, "sources": sources}
    questions = {pid: f"Does passage {pid} help answer the question?" for pid in passages}
    questions.update({sid: f"Is source {sid} likely to hold what the question needs, "
                           "judging by its sections and best passage?" for sid in sources})
    questions["enough"] = ("Taken together, do these passages hold what is needed to answer the question well? "
                           "Answer no if the question needs a part of a source that is not here.")

    answers = jev.ask_nouls(state, questions)
    if answers is None:
        return Judgement("error")

    source_of = {card: answers[sid] for sid, card in source_ids.items()}
    scores: dict[tuple[str, str | None], float] = {}
    for pid, key in keys.items():
        p = answers[pid]
        scores[key] = sqrt(p * source_of[key[0]]) if pid.startswith("p") else p
    enough_p = answers["enough"]
    enough = False if enough_p <= ENOUGH_NO else True if enough_p >= ENOUGH_YES else None
    return Judgement("ok", scores, enough)


def rank_chunks(chunks: list[ScoredChunk], scores: dict[tuple[str, str | None], float] | None, per_source: int,
                protect: set[str] | None = None) -> tuple[list[ScoredChunk], list[dict]]:
    """Best first, at most `per_source` of each card; passages Jev is sure about go. `protect` names
    cards that keep their best passage even so: strings, on a model that cannot go and open them."""
    ordered = list(chunks) if scores is None else sorted(chunks, key=lambda c: -scores.get((c.card_id, c.loc), 0.0))
    seen: dict[str, int] = {}
    kept: list[ScoredChunk] = []
    dropped: list[dict] = []
    for c in ordered:
        score = None if scores is None else scores.get((c.card_id, c.loc), 0.0)
        count = seen.get(c.card_id, 0)
        if score is not None and score < DROP_BELOW and not (protect and c.card_id in protect and count == 0):
            dropped.append({"cardId": c.card_id, "loc": c.loc, "score": score, "reason": "jev"})
        elif count >= per_source:
            dropped.append({"cardId": c.card_id, "loc": c.loc, "score": score, "reason": "per_source"})
        else:
            seen[c.card_id] = count + 1
            kept.append(c)
    return kept, dropped
