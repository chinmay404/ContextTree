"""Connectors: a link to a PDF, thin pages, YouTube transcripts and pasted transcripts."""
import pytest

from app.agent import ingest
from app.agent.helpers.youtube import TranscriptError
from app.api.v1.endpoints import sources
from app.core.config import settings
from tests.v3_support import (  # noqa: F401
    FakeLLM, api, db, make_board, new_id, requires_db, seed_board, seed_card, unit_emb, user,
)

pytestmark = requires_db

URL = "https://www.youtube.com/watch?v=abcdefghijk"


@pytest.fixture
def stubs(monkeypatch):
    monkeypatch.setattr(ingest, "get_embeddings", lambda texts: [unit_emb(i % 768) for i, _ in enumerate(texts)])
    monkeypatch.setattr(ingest, "get_embedding", lambda t: unit_emb(0))
    monkeypatch.setattr(ingest, "get_llm", lambda *a, **k: FakeLLM("An overview."))


def _row(db, card):
    with db.cursor() as cur:
        cur.execute("SELECT status, source_type, data FROM nodes WHERE id = %s", (card,))
        return cur.fetchone()


def _chunks(db, card):
    with db.cursor() as cur:
        cur.execute("SELECT fc.metadata->>'loc', fc.chunk_text FROM file_chunks fc "
                    "JOIN external_files ef ON ef.id = fc.file_id WHERE ef.node_id = %s "
                    "ORDER BY ef.created_at, fc.chunk_index", (card,))
        return cur.fetchall()


def _web(db, user):
    board = seed_board(db, user)
    return board, seed_card(db, user, board, kind="source", source_type="web", status="processing",
                            data={"url": "https://8.8.8.8/papers/p.pdf", "title": "8.8.8.8", "meta": "8.8.8.8"})


def _video(db, user, status="processing", **data):
    board = seed_board(db, user)
    return board, seed_card(db, user, board, kind="source", source_type="video", status=status,
                            data={"url": URL, "ytid": "abcdefghijk", "title": "A talk", "meta": "YouTube", **data})


# ── a link to a PDF ──────────────────────────────────────────────


def test_a_link_to_a_pdf_is_read_like_an_uploaded_pdf(db, user, stubs, monkeypatch):
    board, card = _web(db, user)
    monkeypatch.setattr(ingest, "fetch_url_content",
                        lambda u, m: {"url": u, "title": "Paper.pdf", "pdf": b"%PDF-1.7 body"})
    monkeypatch.setattr(ingest, "load_and_split_documents",
                        lambda data, mime, name: (["one", "two"], [{"page": 0}, {"page": 4}], "one two", None))
    ingest.ingest_web_source(card, user)
    status, source_type, data = _row(db, card)
    assert (status, source_type) == ("ready", "pdf")
    assert data["meta"] == "PDF · 5 pages" and data["fileName"] == "Paper.pdf" and data["title"] == "Paper"
    assert [loc for loc, _ in _chunks(db, card)] == ["p. 1", "p. 5"]
    with db.cursor() as cur:
        cur.execute("SELECT data, mime_type, file_size FROM external_files WHERE node_id = %s", (card,))
        raw, mime, size = cur.fetchone()
    assert bytes(raw) == b"%PDF-1.7 body" and mime == "application/pdf" and size == 13


def test_a_pdf_link_that_is_not_really_a_pdf_fails(db, user, stubs, monkeypatch):
    _, card = _web(db, user)
    monkeypatch.setattr(ingest, "fetch_url_content", lambda u, m: {"url": u, "title": "x", "pdf": b"<html>nope"})
    ingest.ingest_web_source(card, user)
    status, source_type, data = _row(db, card)
    assert (status, source_type) == ("failed", "web") and data["error"] == "Couldn't read this page"


def test_a_pdf_that_is_too_big_says_so(db, user, stubs, monkeypatch):
    _, card = _web(db, user)
    monkeypatch.setattr(ingest, "fetch_url_content", lambda u, m: {"url": u, "error": "This PDF is over 4 MB, which is the limit"})
    ingest.ingest_web_source(card, user)
    status, _, data = _row(db, card)
    assert status == "failed" and "4 MB" in data["error"]


def test_a_page_with_almost_no_text_fails_honestly(db, user, stubs, monkeypatch):
    _, card = _web(db, user)
    monkeypatch.setattr(ingest, "fetch_url_content", lambda u, m: {"url": u, "title": "App", "text": "Loading..."})
    ingest.ingest_web_source(card, user)
    status, _, data = _row(db, card)
    assert status == "failed" and "almost no readable text" in data["error"]


# ── YouTube ──────────────────────────────────────────────────────


PARTS = [{"start": float(i * 30), "text": "spoken word " * 30} for i in range(5)]


def test_a_video_with_captions_is_read(db, user, stubs, monkeypatch):
    _, card = _video(db, user)
    monkeypatch.setattr(ingest, "fetch_captions", lambda ytid: PARTS)
    ingest.ingest_video_source(card, user)
    status, _, data = _row(db, card)
    assert status == "ready" and data["meta"] == "YouTube · 2 min" and data["transcriptOrigin"] == "captions"
    assert data["summary"] == "An overview." and data["excerpts"]
    locs = [loc for loc, _ in _chunks(db, card)]
    assert locs[0] == "0:00" and len(set(locs)) == len(locs)


def test_ingesting_a_video_twice_leaves_one_set_of_chunks(db, user, stubs, monkeypatch):
    _, card = _video(db, user)
    monkeypatch.setattr(ingest, "fetch_captions", lambda ytid: PARTS)
    ingest.ingest_video_source(card, user)
    n = len(_chunks(db, card))
    with db.cursor() as cur:
        cur.execute("UPDATE nodes SET status = 'processing' WHERE id = %s", (card,))
    ingest.ingest_video_source(card, user)
    assert len(_chunks(db, card)) == n


@pytest.mark.parametrize("reason,phrase", [
    ("no_captions", "no captions"), ("blocked", "didn't let our server"), ("failed", "Couldn't read"),
])
def test_a_video_that_cannot_be_read_offers_pasting(db, user, stubs, monkeypatch, reason, phrase):
    _, card = _video(db, user)

    def refuse(ytid):
        raise TranscriptError(reason)

    monkeypatch.setattr(ingest, "fetch_captions", refuse)
    monkeypatch.setattr(settings, "YOUTUBE_GEMINI_FALLBACK", False)
    ingest.ingest_video_source(card, user)
    status, _, data = _row(db, card)
    assert status == "failed" and phrase in data["error"] and "paste" in data["error"].lower()


def _blocked(ytid):
    raise TranscriptError("blocked")


def test_the_gemini_fallback_stays_off_unless_switched_on(db, user, stubs, monkeypatch):
    _, card = _video(db, user)
    monkeypatch.setattr(ingest, "fetch_captions", _blocked)
    monkeypatch.setattr(settings, "YOUTUBE_GEMINI_FALLBACK", False)
    monkeypatch.setattr(ingest, "fetch_ai_transcript", lambda ytid: pytest.fail("must not spend"))
    ingest.ingest_video_source(card, user)
    assert _row(db, card)[0] == "failed"


def test_the_gemini_fallback_labels_its_text_as_ai(db, user, stubs, monkeypatch):
    _, card = _video(db, user)
    monkeypatch.setattr(ingest, "fetch_captions", _blocked)
    monkeypatch.setattr(settings, "YOUTUBE_GEMINI_FALLBACK", True)
    monkeypatch.setattr(ingest, "fetch_ai_transcript", lambda ytid: PARTS)
    ingest.ingest_video_source(card, user)
    status, _, data = _row(db, card)
    assert status == "ready" and data["transcriptOrigin"] == "ai"
    assert data["meta"] == "YouTube · AI transcript · 2 min"


def test_the_gemini_fallback_has_a_daily_limit_per_user(db, user, stubs, monkeypatch):
    monkeypatch.setattr(settings, "YOUTUBE_GEMINI_FALLBACK", True)
    monkeypatch.setattr(ingest, "AI_TRANSCRIPTS_PER_DAY", 1)
    monkeypatch.setattr(ingest, "fetch_captions", _blocked)
    monkeypatch.setattr(ingest, "fetch_ai_transcript", lambda ytid: PARTS)
    _, first = _video(db, user)
    ingest.ingest_video_source(first, user)
    assert _row(db, first)[0] == "ready"
    _, second = _video(db, user)
    monkeypatch.setattr(ingest, "fetch_ai_transcript", lambda ytid: pytest.fail("over the limit"))
    ingest.ingest_video_source(second, user)
    assert _row(db, second)[0] == "failed"


def test_a_video_link_starts_processing_and_the_ingest_route_reads_it(api, db, stubs, monkeypatch):
    b = make_board(api)["id"]
    monkeypatch.setattr(sources, "fetch_youtube_oembed", lambda u: {"title": "A talk"})
    monkeypatch.setattr(ingest, "fetch_captions", lambda ytid: PARTS)
    r = api.post(f"/boards/{b}/sources/link", json={"id": new_id("c"), "url": URL, "x": 0, "y": 0})
    assert r.status_code == 202 and (r.json()["sourceType"], r.json()["status"]) == ("video", "processing")
    done = api.post(f"/boards/{b}/cards/{r.json()['id']}/ingest")
    assert done.status_code == 200 and done.json()["status"] == "ready"
    passages = api.get(f"/files/{r.json()['id']}/passages")
    assert passages.status_code == 200 and passages.json()["passages"][0]["loc"] == "0:00"


def test_an_old_coming_soon_video_starts_when_kicked(api, db, stubs, monkeypatch):
    b = make_board(api)["id"]
    card = seed_card(db, api.email, b, kind="source", source_type="video", status="soon",
                     data={"url": URL, "ytid": "abcdefghijk", "title": "Old", "meta": "YouTube"})
    monkeypatch.setattr(ingest, "fetch_captions", lambda ytid: PARTS)
    done = api.post(f"/boards/{b}/cards/{card}/ingest")
    assert done.status_code == 200 and done.json()["status"] == "ready"


# ── a pasted transcript ──────────────────────────────────────────


def _paste(api, b, card, text):
    return api.post(f"/boards/{b}/cards/{card}/transcript", json={"text": text})


def test_a_pasted_transcript_rescues_a_failed_video(api, db, stubs):
    b = make_board(api)["id"]
    card = seed_card(db, api.email, b, kind="source", source_type="video", status="failed",
                     data={"url": URL, "ytid": "abcdefghijk", "title": "A talk", "meta": "YouTube", "error": "blocked"})
    r = _paste(api, b, card, "words from the talk. " * 200)
    assert r.status_code == 200 and r.json()["status"] == "ready"
    data = r.json()["data"]
    assert data["transcriptOrigin"] == "pasted" and data["meta"] == "YouTube · pasted transcript"
    assert not data.get("error")
    assert _chunks(db, card)[0][0] == "§ 1"


def test_a_pasted_transcript_with_timestamps_keeps_them(api, db, stubs):
    b = make_board(api)["id"]
    card = seed_card(db, api.email, b, kind="source", source_type="video", status="failed",
                     data={"url": URL, "ytid": "abcdefghijk", "title": "T", "meta": "YouTube"})
    _paste(api, b, card, "[00:10] first part of it\n[05:00] second part of it")
    assert [loc for loc, _ in _chunks(db, card)][0] == "0:10"


def test_pasting_replaces_what_was_read(api, db, stubs):
    b = make_board(api)["id"]
    card = seed_card(db, api.email, b, kind="source", source_type="video", status="failed",
                     data={"url": URL, "ytid": "abcdefghijk", "title": "T", "meta": "YouTube"})
    _paste(api, b, card, "old words " * 50)
    _paste(api, b, card, "new words " * 50)
    assert all("new words" in text for _, text in _chunks(db, card))


def test_only_a_video_takes_a_transcript(api, db, stubs):
    b = make_board(api)["id"]
    card = seed_card(db, api.email, b, kind="source", source_type="pdf", status="ready", data={"title": "T"})
    assert _paste(api, b, card, "words").status_code == 400


def test_an_empty_or_huge_transcript_is_refused(api, db, stubs):
    b = make_board(api)["id"]
    card = seed_card(db, api.email, b, kind="source", source_type="video", status="failed",
                     data={"url": URL, "title": "T", "meta": "YouTube"})
    assert _paste(api, b, card, "   ").status_code == 400
    assert _paste(api, b, card, "x" * 200_001).status_code in (400, 422)


def test_a_transcript_for_someone_elses_card_is_404(api, db, stubs, user):
    other = seed_board(db, user)
    card = seed_card(db, user, other, kind="source", source_type="video", status="failed", data={"url": URL})
    b = make_board(api)["id"]
    assert _paste(api, b, card, "words").status_code == 404
