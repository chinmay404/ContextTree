"""
Web search injection — unit tests, no network. The live DDG call is mocked;
what's under test is the contract: formatting, clipping, failure isolation,
and the opt-in flag on the request schema.
"""

from unittest.mock import patch

from app.agent.helpers.web_search import format_web_snippets, search_web
from app.schemas.item import ChatMessage


def _msg(**extra):
    return ChatMessage(message="hi", message_id="m1", **{"nodeId": "n1", **extra})


def test_web_search_defaults_off_and_alias_maps():
    assert _msg().web_search is False
    assert _msg(webSearch=True).web_search is True


def test_format_web_snippets_labels_and_clips():
    rows = [{"title": "Tokyo", "url": "https://x.y/t", "snippet": "a" * 100}]
    out = format_web_snippets(rows, clip_chars=20)
    assert len(out) == 1
    assert out[0].startswith("[web 1 | Tokyo | https://x.y/t] ")
    assert out[0].endswith("…") and len(out[0].split("] ", 1)[1]) == 20


def test_search_web_normalizes_ddgs_rows():
    fake_rows = [
        {"title": "T1", "href": "https://a", "body": "B1"},
        {"title": "", "href": "", "body": ""},  # dropped: nothing usable
        {"title": "T3", "url": "https://c", "snippet": "B3"},  # alt keys
    ]

    class FakeDDGS:
        def __init__(self, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, query, max_results=3):
            return fake_rows

    with patch("ddgs.DDGS", FakeDDGS):
        results = search_web("tokyo trip", max_results=3)
    assert [r["title"] for r in results] == ["T1", "T3"]
    assert results[1]["url"] == "https://c" and results[1]["snippet"] == "B3"


def test_search_web_failure_is_silent_empty():
    class BoomDDGS:
        def __init__(self, timeout=None):
            raise RuntimeError("rate limited")

    with patch("ddgs.DDGS", BoomDDGS):
        assert search_web("anything") == []


def test_search_web_empty_query_short_circuits():
    assert search_web("   ") == []
