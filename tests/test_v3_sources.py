import pytest

from app.api.v1.endpoints import sources
from tests.v3_support import requires_db, db, user, api, make_board, new_id, seed_card, seed_board  # noqa: F401

pytestmark = requires_db


class Rec:
    def __init__(self):
        self.file_calls = []
        self.web_calls = []


@pytest.fixture
def rec(monkeypatch):
    r = Rec()
    monkeypatch.setattr(sources, "ingest_file_source", lambda cid, email: r.file_calls.append((cid, email)))
    monkeypatch.setattr(sources, "ingest_web_source", lambda cid, email: r.web_calls.append((cid, email)))
    return r


def up(api, b, name, content, ctype="application/pdf", **fields):
    return api.post(f"/boards/{b}/sources/file",
                    data={"id": fields.pop("id", new_id("c")), "x": "1", "y": "2", **fields},
                    files={"file": (name, content, ctype)})


def link(api, b, url, **kw):
    return api.post(f"/boards/{b}/sources/link", json={"id": new_id("c"), "url": url, "x": 0, "y": 0, **kw})


def test_upload_pdf_202_processing_and_schedules_ingest(api, rec):
    b = make_board(api)["id"]
    r = up(api, b, "Survey.pdf", b"%PDF-1.4 ...")
    assert r.status_code == 202
    c = r.json()
    assert (c["kind"], c["sourceType"], c["status"]) == ("source", "pdf", "processing")
    assert c["data"]["fileName"] == "Survey.pdf" and c["data"]["title"] == "Survey"
    assert c["data"]["meta"] == "PDF · 12 B" and c["data"]["fileSize"] == 12
    assert rec.file_calls == [(c["id"], api.email)]


def test_upload_image_soon_no_ingest(api, rec):
    b = make_board(api)["id"]
    r = up(api, b, "shot.png", b"\x89PNG", "image/png", desc="why")
    c = r.json()
    assert r.status_code == 202 and (c["sourceType"], c["status"]) == ("image", "soon")
    assert c["data"]["desc"] == "why" and rec.file_calls == []


def test_upload_audio_soon(api, rec):
    b = make_board(api)["id"]
    c = up(api, b, "ep.m4a", b"abc", "audio/mp4").json()
    assert (c["sourceType"], c["status"]) == ("audio", "soon")


def test_upload_over_4mb_413(api, rec):
    b = make_board(api)["id"]
    r = up(api, b, "a.txt", b"x" * (4 * 1024 * 1024 + 1), "text/plain")
    assert r.status_code == 413 and r.json()["error"] == "too_large"


def test_upload_exactly_4mb_ok(api, rec):
    b = make_board(api)["id"]
    r = up(api, b, "a.txt", b"x" * (4 * 1024 * 1024), "text/plain")
    assert r.status_code == 202 and r.json()["data"]["meta"] == "DOC · 4.0 MB"


def test_upload_exe_415(api, rec):
    b = make_board(api)["id"]
    r = up(api, b, "a.exe", b"MZ", "application/octet-stream")
    assert r.status_code == 415 and r.json()["error"] == "unsupported_type"


def test_upload_idempotent_200(api, rec, db):
    b = make_board(api)["id"]
    cid = new_id("c")
    assert up(api, b, "a.txt", b"hello", "text/plain", id=cid).status_code == 202
    assert up(api, b, "a.txt", b"hello", "text/plain", id=cid).status_code == 200
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM external_files WHERE node_id = %s", (cid,))
        assert cur.fetchone()[0] == 1
    assert rec.file_calls == [(cid, api.email)]


def test_upload_desk_must_be_live_chat_400(api, db, rec):
    b = make_board(api)["id"]
    note = seed_card(db, api.email, b, kind="source", source_type="note", status="ready")
    r = up(api, b, "a.txt", b"hi", "text/plain", deskId=note)
    assert r.status_code == 400
    assert up(api, b, "a.txt", b"hi", "text/plain", deskId="bad id!").status_code == 400
    chat = seed_card(db, api.email, b, kind="chat")
    r = up(api, b, "a.txt", b"hi", "text/plain", deskId=chat)
    assert r.status_code == 202 and r.json()["deskId"] == chat


def test_upload_counts_toward_card_limit_409(api, db, rec):
    b = make_board(api)["id"]
    for _ in range(50):
        seed_card(db, api.email, b, kind="source", source_type="note", status="ready")
    r = up(api, b, "a.txt", b"hi", "text/plain")
    assert r.status_code == 409 and r.json()["error"] == "card_limit"


def test_link_youtube_soon_with_oembed_title(api, monkeypatch, rec):
    b = make_board(api)["id"]
    monkeypatch.setattr(sources, "fetch_youtube_oembed", lambda u: {"title": "How to price a developer tool", "thumbnail_url": "https://i.ytimg.com/vi/abcdefghijk/hq.jpg"})
    r = link(api, b, "https://www.youtube.com/watch?v=abcdefghijk")
    c = r.json()
    assert r.status_code == 202 and (c["sourceType"], c["status"]) == ("video", "soon")
    assert c["data"]["ytid"] == "abcdefghijk" and c["data"]["title"] == "How to price a developer tool"
    assert "thumbnail" not in c["data"] and rec.web_calls == []


def test_link_youtu_be_short_form(api, monkeypatch, rec):
    b = make_board(api)["id"]
    monkeypatch.setattr(sources, "fetch_youtube_oembed", lambda u: None)
    c = link(api, b, "https://youtu.be/abcdefghijk?t=3").json()
    assert c["data"]["ytid"] == "abcdefghijk" and c["data"]["title"] == "YouTube video"


def test_parse_youtube_id_forms():
    p = sources.parse_youtube_id
    assert p("https://m.youtube.com/watch?v=abcdefghijk&x=1") == "abcdefghijk"
    assert p("https://youtube.com/shorts/abcdefghijk") == "abcdefghijk"
    assert p("https://www.youtube.com/watch?v=short") is None
    assert p("https://evil.com/watch?v=abcdefghijk") is None


def test_link_web_processing_schedules_ingest(api, monkeypatch, rec):
    b = make_board(api)["id"]
    monkeypatch.setattr(sources, "host_is_public", lambda h: True)
    r = link(api, b, "https://harbor.dev/blog")
    c = r.json()
    assert r.status_code == 202 and (c["sourceType"], c["status"]) == ("web", "processing")
    assert c["data"]["title"] == "harbor.dev" and rec.web_calls == [(c["id"], api.email)]


def test_link_private_host_400(api, rec):
    b = make_board(api)["id"]
    for u in ("http://127.0.0.1/admin", "http://localhost:8000"):
        r = link(api, b, u)
        assert r.status_code == 400 and r.json()["error"] == "invalid"
    assert rec.web_calls == []


def test_link_non_http_scheme_400(api, rec):
    b = make_board(api)["id"]
    for u in ("ftp://x.org/a", "javascript:alert(1)"):
        assert link(api, b, u).status_code == 400


def test_link_idempotent_no_second_ingest(api, monkeypatch, rec):
    b = make_board(api)["id"]
    monkeypatch.setattr(sources, "host_is_public", lambda h: True)
    body = {"id": new_id("c"), "url": "https://harbor.dev/x", "x": 0, "y": 0}
    assert api.post(f"/boards/{b}/sources/link", json=body).status_code == 202
    assert api.post(f"/boards/{b}/sources/link", json=body).status_code == 200
    assert len(rec.web_calls) == 1


def test_raw_returns_bytes_owner_only(api, db, rec):
    from tests.v3_support import Api
    b = make_board(api)["id"]
    cid = up(api, b, 'n"q.txt', b"hello", "text/plain").json()["id"]
    r = api.get(f"/files/{cid}/raw")
    assert r.status_code == 200 and r.content == b"hello"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["content-type"].startswith("text/plain")
    assert r.headers["content-security-policy"] == "sandbox"
    other = f"{new_id('o')}@example.com"
    with db.cursor() as cur:
        cur.execute("INSERT INTO users (email, name) VALUES (%s, 'o')", (other,))
    try:
        assert Api(api.client, other).get(f"/files/{cid}/raw").status_code == 404
    finally:
        with db.cursor() as cur:
            cur.execute("DELETE FROM users WHERE email = %s", (other,))
    assert api.delete(f"/boards/{b}/cards/{cid}").status_code == 200
    assert api.get(f"/files/{cid}/raw").status_code == 404


def test_raw_404_for_web_card_without_bytes(api, db, monkeypatch, rec):
    b = make_board(api)["id"]
    monkeypatch.setattr(sources, "host_is_public", lambda h: True)
    cid = link(api, b, "https://harbor.dev/x").json()["id"]
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO external_files (id, user_email, node_id, canvas_id, file_name, file_type, "
            "mime_type, content, data, processed) VALUES (%s,%s,%s,%s,'p','text/html','text/html','t',NULL,true)",
            (f"file_{cid}", api.email, cid, b))
    assert api.get(f"/files/{cid}/raw").status_code == 404


def _insert_raw(db, api, b, cid, file_name, mime="text/plain", data=b"hi"):
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO external_files (id, user_email, node_id, canvas_id, file_name, file_type, "
            "mime_type, data, processed) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,true)",
            (f"file_{cid}", api.email, cid, b, file_name, mime, mime, data))


def test_raw_header_injection_and_non_ascii_names(api, db, rec):
    b = make_board(api)["id"]
    cid = seed_card(db, api.email, b, kind="source", source_type="doc", status="ready")
    hostile = 'a"b\r\nX-Evil: 1\x7f\\.txt'
    _insert_raw(db, api, b, cid, hostile)
    r = api.get(f"/files/{cid}/raw")
    cd = r.headers["content-disposition"]
    assert r.status_code == 200 and "x-evil" not in r.headers
    ascii_part = cd.split('filename="')[1].split('"; filename*')[0]
    assert ascii_part == "a_b__X-Evil: 1__.txt" and "\n" not in cd and "\r" not in cd
    cid2 = seed_card(db, api.email, b, kind="source", source_type="pdf", status="ready")
    _insert_raw(db, api, b, cid2, "报告.pdf", "application/pdf")
    r = api.get(f"/files/{cid2}/raw")
    assert r.status_code == 200
    assert r.headers["content-disposition"] == (
        "inline; filename=\"__.pdf\"; filename*=UTF-8''%E6%8A%A5%E5%91%8A.pdf")


def test_raw_404_when_board_deleted(api, db, rec):
    b = make_board(api)["id"]
    cid = up(api, b, "a.txt", b"hi", "text/plain").json()["id"]
    assert api.get(f"/files/{cid}/raw").status_code == 200
    with db.cursor() as cur:
        cur.execute("UPDATE canvases SET deleted_at = now() WHERE id = %s", (b,))
    assert api.get(f"/files/{cid}/raw").status_code == 404


def test_upload_html_content_type_served_as_text_plain(api, rec):
    b = make_board(api)["id"]
    c = up(api, b, "a.txt", b"<script>1</script>", "text/html").json()
    r = api.get(f"/files/{c['id']}/raw")
    assert r.headers["content-type"].startswith("text/plain")


def test_content_type_capped_and_empty_stem(api, db, rec):
    b = make_board(api)["id"]
    c = up(api, b, ".pdf", b"x", "application/" + "a" * 300).json()
    assert c["data"]["title"] == ".pdf" and len(c["data"]["fileType"]) == 100


def test_fetch_youtube_oembed(monkeypatch):
    seen = {}

    class R:
        def raise_for_status(self): pass
        def json(self): return {"title": "T", "thumbnail_url": "u", "x": 1}

    def fake_get(url, params=None, timeout=None):
        seen.update(url=url, params=params, timeout=timeout)
        return R()
    monkeypatch.setattr(sources.httpx, "get", fake_get)
    assert sources.fetch_youtube_oembed("https://youtu.be/abcdefghijk") == {"title": "T", "thumbnail_url": "u"}
    assert seen["url"] == "https://www.youtube.com/oembed"
    assert seen["params"] == {"url": "https://youtu.be/abcdefghijk", "format": "json"} and seen["timeout"] == 5

    def boom(*a, **k): raise RuntimeError("net")
    monkeypatch.setattr(sources.httpx, "get", boom)
    assert sources.fetch_youtube_oembed("https://youtu.be/abcdefghijk") is None


def test_parse_youtube_id_rejects_trailing_newline():
    assert sources._YT_ID.fullmatch("abcdefghijk\n") is None
    assert sources._YT_ID.fullmatch("abcdefghijk") is not None


def test_link_repost_does_no_network(api, monkeypatch, rec):
    b = make_board(api)["id"]
    body = {"id": new_id("c"), "url": "https://youtu.be/abcdefghijk", "x": 0, "y": 0}
    monkeypatch.setattr(sources, "fetch_youtube_oembed", lambda u: None)
    assert api.post(f"/boards/{b}/sources/link", json=body).status_code == 202

    def boom(*a, **k): raise AssertionError("network")
    monkeypatch.setattr(sources, "fetch_youtube_oembed", boom)
    monkeypatch.setattr(sources, "host_is_public", boom)
    assert api.post(f"/boards/{b}/sources/link", json=body).status_code == 200


def test_oversized_streamed_body_413_without_handler(api, monkeypatch, rec):
    b = make_board(api)["id"]
    ran = []
    monkeypatch.setattr(sources, "_create_file_card", lambda *a: ran.append(1))
    boundary = "XBOUND"
    head = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.txt"\r\n'
            "Content-Type: text/plain\r\n\r\n").encode()

    def gen():
        yield head
        for _ in range(60):
            yield b"x" * (100 * 1024)  # ~6 MB, no Content-Length

    r = api.post(f"/boards/{b}/sources/file", content=gen(),
                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert r.status_code == 413 and r.json()["error"] == "too_large" and ran == []


def test_declared_content_length_over_cap_413(api, rec):
    b = make_board(api)["id"]
    r = api.post(f"/boards/{b}/sources/file", content=b"x" * (4 * 1024 * 1024 + 70 * 1024),
                 headers={"Content-Type": "multipart/form-data; boundary=B"})
    assert r.status_code == 413 and r.json()["error"] == "too_large"
