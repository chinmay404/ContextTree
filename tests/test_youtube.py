"""YouTube transcripts: captions, the Gemini fallback, and timed chunks. No network."""
import httpx
import pytest
from youtube_transcript_api import IpBlocked, NoTranscriptFound, RequestBlocked, TranscriptsDisabled

from app.agent.helpers import youtube
from app.agent.helpers.youtube import TranscriptError, parse_timed_lines, transcript_chunks
from app.core.config import settings

ID = "abcdefghijk"


class FakeTranscript:
    def __init__(self, rows):
        self.rows = rows

    def fetch(self):
        return self

    def to_raw_data(self):
        return self.rows


class FakeList:
    def __init__(self, english=None, others=()):
        self.english, self.others = english, list(others)

    def find_transcript(self, langs):
        if self.english is None:
            raise NoTranscriptFound(ID, langs, self)
        return self.english

    def __iter__(self):
        return iter(self.others)


class FakeApi:
    def __init__(self, result):
        self.result = result

    def list(self, video_id):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _use(monkeypatch, result):
    monkeypatch.setattr(youtube, "_api", lambda: FakeApi(result))


def test_captions_prefer_english(monkeypatch):
    _use(monkeypatch, FakeList(FakeTranscript([{"text": "hello", "start": 1.5, "duration": 2}])))
    assert youtube.fetch_captions(ID) == [{"start": 1.5, "text": "hello"}]


def test_captions_fall_back_to_any_language(monkeypatch):
    _use(monkeypatch, FakeList(None, [FakeTranscript([{"text": "hola", "start": 0, "duration": 1}])]))
    assert youtube.fetch_captions(ID) == [{"start": 0.0, "text": "hola"}]


def test_blank_and_music_only_lines_are_dropped(monkeypatch):
    rows = [{"text": " ", "start": 0, "duration": 1}, {"text": "[Music]", "start": 1, "duration": 1},
            {"text": "real words", "start": 2, "duration": 1}]
    _use(monkeypatch, FakeList(FakeTranscript(rows)))
    assert youtube.fetch_captions(ID) == [{"start": 2.0, "text": "real words"}]


@pytest.mark.parametrize("error,reason", [
    (TranscriptsDisabled(ID), "no_captions"),
    (RequestBlocked(ID), "blocked"),
    (IpBlocked(ID), "blocked"),
    (RuntimeError("anything else"), "failed"),
])
def test_failures_say_why(monkeypatch, error, reason):
    _use(monkeypatch, error)
    with pytest.raises(TranscriptError) as e:
        youtube.fetch_captions(ID)
    assert e.value.reason == reason


def test_a_video_with_no_transcripts_at_all_is_no_captions(monkeypatch):
    _use(monkeypatch, FakeList(None, []))
    with pytest.raises(TranscriptError) as e:
        youtube.fetch_captions(ID)
    assert e.value.reason == "no_captions"


def test_a_hung_request_is_abandoned(monkeypatch):
    import threading
    release = threading.Event()

    class Hung:
        def list(self, video_id):
            release.wait(5)
            raise RuntimeError("too late")

    monkeypatch.setattr(youtube, "_api", lambda: Hung())
    monkeypatch.setattr(youtube, "CAPTIONS_TIMEOUT_SECONDS", 0.05)
    with pytest.raises(TranscriptError) as e:
        youtube.fetch_captions(ID)
    release.set()
    assert e.value.reason == "failed"


def test_the_proxy_setting_reaches_the_client(monkeypatch):
    seen = {}

    class Recording:
        def __init__(self, proxy_config=None):
            seen["config"] = proxy_config

    monkeypatch.setattr(youtube, "YouTubeTranscriptApi", Recording)
    monkeypatch.setattr(settings, "YOUTUBE_PROXY_URL", "http://user:pw@proxy.example:8080")
    youtube._api()
    assert seen["config"].http_url == "http://user:pw@proxy.example:8080"


def test_no_proxy_by_default(monkeypatch):
    seen = {}

    class Recording:
        def __init__(self, proxy_config=None):
            seen["config"] = proxy_config

    monkeypatch.setattr(youtube, "YouTubeTranscriptApi", Recording)
    monkeypatch.setattr(settings, "YOUTUBE_PROXY_URL", None)
    youtube._api()
    assert seen["config"] is None


# ── chunks ───────────────────────────────────────────────────────


def test_chunks_group_lines_and_start_at_their_first_timestamp():
    parts = [{"start": float(i * 10), "text": ("word " * 40).strip()} for i in range(12)]
    chunks, metas, full = transcript_chunks(parts)
    assert len(chunks) == 3
    assert [m["loc"] for m in metas] == ["0:00", "0:50", "1:40"]
    assert [m["start_s"] for m in metas] == [0, 50, 100]
    assert all(len(c) <= 1000 for c in chunks)
    assert full.count("word") == 480


def test_hour_long_videos_show_hours():
    _, metas, _ = transcript_chunks([{"start": 3725.0, "text": "late"}])
    assert metas[0]["loc"] == "1:02:05"


def test_a_very_long_line_is_split_so_each_chunk_stays_embeddable():
    chunks, metas, _ = transcript_chunks([{"start": 60.0, "text": "word " * 700}])
    assert len(chunks) > 1 and all(len(c) <= 1000 for c in chunks)
    assert len({m["loc"] for m in metas}) == len(metas), "locations must stay unique"


def test_duration_label():
    assert youtube.duration_label([{"start": 725.0, "text": "x"}]) == "12 min"
    assert youtube.duration_label([{"start": 20.0, "text": "x"}]) == "1 min"
    assert youtube.duration_label([{"start": 3900.0, "text": "x"}]) == "1 h 5 min"


# ── Gemini fallback ──────────────────────────────────────────────


def test_timed_lines_parse_into_parts():
    text = "[00:01] hello there\nmore of that line\n[1:02:03] much later\n\n"
    assert parse_timed_lines(text) == [
        {"start": 1.0, "text": "hello there more of that line"},
        {"start": 3723.0, "text": "much later"},
    ]


def test_a_reply_with_no_timestamps_is_refused():
    assert parse_timed_lines("I cannot watch videos.") == []


def _gemini(monkeypatch, handler):
    calls = []

    def post(url, **kw):
        calls.append((url, kw))
        return handler(httpx.Request("POST", url))

    monkeypatch.setattr(youtube.httpx, "post", post)
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "test-key")
    return calls


def test_gemini_is_asked_about_the_video_url(monkeypatch):
    calls = _gemini(monkeypatch, lambda req: httpx.Response(200, request=req, json={
        "candidates": [{"content": {"parts": [{"text": "[00:05] spoken words"}]}}]}))
    out = youtube.fetch_ai_transcript(ID)
    assert out == [{"start": 5.0, "text": "spoken words"}]
    url, kw = calls[0]
    assert "test-key" not in url and kw["headers"]["x-goog-api-key"] == "test-key"
    assert kw["json"]["contents"][0]["parts"][0]["fileData"]["fileUri"] == f"https://www.youtube.com/watch?v={ID}"


def test_gemini_failure_is_an_empty_answer(monkeypatch):
    _gemini(monkeypatch, lambda req: httpx.Response(500, request=req, text="key secret"))
    assert youtube.fetch_ai_transcript(ID) == []


def test_gemini_without_a_key_is_never_called(monkeypatch):
    calls = _gemini(monkeypatch, lambda req: httpx.Response(200, request=req, json={}))
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", None)
    assert youtube.fetch_ai_transcript(ID) == [] and calls == []
