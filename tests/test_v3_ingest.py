"""Task 12: V3 ingestion core (CSV, locators, overview, key passages, status flow, web)."""
import pytest

from app.agent import ingest
from app.agent.ingest import (
    add_locators, make_overview, parse_csv, pick_key_passages, _clip_text,
)
from tests.v3_support import FakeLLM, db, requires_db, seed_board, seed_card, unit_emb, user, api  # noqa: F401


# ── pure helpers ─────────────────────────────────────────────────────────────
def test_csv_chunks_40_rows_with_locators():
    data = ("a,b\n" + "".join(f"{i},x\n" for i in range(1, 96))).encode()
    chunks, metas, table = parse_csv(data)
    assert [m["loc"] for m in metas] == ["rows 1–40", "rows 41–80", "rows 81–95"]
    assert table["columns"] == ["a", "b"] and len(table["rows"]) == 95
    assert chunks[0].startswith("a,b\n1,x")


def test_csv_table_keeps_first_200_rows():
    data = ("a,b\n" + "".join(f"{i},x\n" for i in range(1, 251))).encode()
    chunks, metas, table = parse_csv(data)
    assert len(table["rows"]) == 200 and len(chunks) == 7


def test_csv_bom_and_latin1():
    _, _, table = parse_csv(b"\xef\xbb\xbfa,b\n1,x\n")
    assert table["columns"] == ["a", "b"]
    _, _, table = parse_csv(b"a,b\n\xe9,x\n")
    assert table["rows"] == [["é", "x"]]


def test_pdf_locators_are_pages():
    assert add_locators("pdf", [{"page": 2}]) == [{"page": 2, "loc": "p. 3"}]


def test_doc_locators_are_sections():
    assert [m["loc"] for m in add_locators("doc", [{}, {}])] == ["§ 1", "§ 2"]
    assert add_locators("web", [{"loc": "keep"}])[0]["loc"] == "keep"


def test_key_passages_top5_by_similarity():
    chunks = [f"chunk {i}" for i in range(7)]
    metas = [{"loc": f"§ {i + 1}"} for i in range(7)]
    out = pick_key_passages(chunks, metas, [unit_emb(i) for i in range(7)], unit_emb(3))
    assert out[0] == {"loc": "§ 4", "text": "chunk 3", "on": True}
    assert len(out) == 5 and all(p["on"] is True for p in out)


def test_overview_falls_back_without_llm(monkeypatch):
    monkeypatch.setattr(ingest, "get_llm", lambda *a, **k: None)
    assert make_overview("x" * 500, "e") == _clip_text("x" * 500, 300)


def test_overview_uses_one_call_and_clips(monkeypatch):
    llm = FakeLLM("y" * 900)
    monkeypatch.setattr(ingest, "get_llm", lambda *a, **k: llm)
    out = make_overview("z" * 20000, "e")
    assert len(out) <= 600 and len(llm.calls) == 1
    assert len(llm.calls[0]) < 12_200


# ── DB-backed ────────────────────────────────────────────────────────────────
@pytest.fixture
def stubs(monkeypatch):
    monkeypatch.setattr(ingest, "get_embeddings", lambda texts: [unit_emb(i) for i, _ in enumerate(texts)])
    monkeypatch.setattr(ingest, "get_embedding", lambda t: unit_emb(0))
    monkeypatch.setattr(ingest, "get_llm", lambda *a, **k: FakeLLM("An overview."))


def _file(db, email, board, card, name, mime, data, file_id=None):
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO external_files (id, user_email, node_id, canvas_id, file_name, mime_type, "
            "file_type, data, file_size, processed) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,false)",
            (file_id or f"f_{card}", email, card, board, name, mime, mime, data, len(data)))
    return file_id or f"f_{card}"


def _node(db, card):
    with db.cursor() as cur:
        cur.execute("SELECT status, data FROM nodes WHERE id = %s", (card,))
        return cur.fetchone()


def _chunk_count(db, email, card):
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM file_chunks WHERE file_id IN "
                    "(SELECT id FROM external_files WHERE node_id=%s AND user_email=%s)", (card, email))
        return cur.fetchone()[0]


def _doc_card(db, user, name="n.txt", mime="text/plain", data=b"hello " * 500, source_type="doc"):
    board = seed_board(db, user)
    card = seed_card(db, user, board, kind="source", source_type=source_type, status="processing")
    fid = _file(db, user, board, card, name, mime, data)
    return board, card, fid


@requires_db
def test_ingest_txt_ready(db, user, stubs):
    board, card, fid = _doc_card(db, user)
    ingest.ingest_file_source(card, user)
    status, data = _node(db, card)
    assert status == "ready" and data["summary"] == "An overview."
    assert 1 <= len(data["excerpts"]) <= 5
    assert all(set(e) == {"loc", "text", "on"} for e in data["excerpts"])
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM file_chunks WHERE file_id=%s AND metadata->>'loc' LIKE '§ %%'", (fid,))
        assert cur.fetchone()[0] > 0
        cur.execute("SELECT processed FROM external_files WHERE id=%s", (fid,))
        assert cur.fetchone()[0] is True


@requires_db
def test_ingest_twice_no_duplicate_chunks_or_files(db, user, stubs):
    board, card, fid = _doc_card(db, user)
    ingest.ingest_file_source(card, user)
    first = _chunk_count(db, user, card)
    ingest.ingest_file_source(card, user)
    assert first > 0 and _chunk_count(db, user, card) == first
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM external_files WHERE node_id=%s", (card,))
        assert cur.fetchone()[0] == 1


@requires_db
def test_ingest_zero_embeddings_marks_failed(db, user, stubs, monkeypatch):
    board, card, fid = _doc_card(db, user)
    monkeypatch.setattr(ingest, "get_embeddings", lambda texts: [[0.0] * 768 for _ in texts])
    ingest.ingest_file_source(card, user)
    status, data = _node(db, card)
    assert status == "failed" and data["error"] == "Failed to generate embeddings"
    assert _chunk_count(db, user, card) == 0


@requires_db
def test_ingest_unsupported_bytes_marks_failed(db, user, stubs):
    board, card, fid = _doc_card(db, user, name="x.doc", mime="application/msword", data=b"\xd0\xcf\x11")
    ingest.ingest_file_source(card, user)
    status, data = _node(db, card)
    assert status == "failed" and ".docx" in data["error"]


@requires_db
def test_ingest_loader_error_marks_failed(db, user, stubs):
    board, card, fid = _doc_card(db, user, name="x.pdf", mime="application/pdf", data=b"not a pdf")
    ingest.ingest_file_source(card, user)
    status, data = _node(db, card)
    assert status == "failed" and data["error"]


@requires_db
def test_ingest_empty_text_marks_failed(db, user, stubs):
    board, card, fid = _doc_card(db, user, data=b"")
    ingest.ingest_file_source(card, user)
    assert _node(db, card)[0] == "failed"


@requires_db
def test_ingest_csv_sets_table_and_meta(db, user, stubs):
    csv_bytes = ("a,b\n" + "".join(f"{i},x\n" for i in range(1, 96))).encode()
    board, card, fid = _doc_card(db, user, name="t.csv", mime="text/csv", data=csv_bytes, source_type="csv")
    ingest.ingest_file_source(card, user)
    status, data = _node(db, card)
    assert status == "ready"
    assert data["table"]["columns"] == ["a", "b"] and data["meta"] == "CSV · 95 rows"
    with db.cursor() as cur:
        cur.execute("SELECT metadata->>'loc' FROM file_chunks WHERE file_id=%s ORDER BY chunk_index", (fid,))
        assert [r[0] for r in cur.fetchall()] == ["rows 1–40", "rows 41–80", "rows 81–95"]


def _web_card(db, user):
    board = seed_board(db, user)
    card = seed_card(db, user, board, kind="source", source_type="web", status="processing",
                     data={"url": "https://harbor.dev/post"})
    return board, card


@requires_db
def test_ingest_web_ready(db, user, stubs, monkeypatch):
    board, card = _web_card(db, user)
    monkeypatch.setattr(ingest, "fetch_url_content",
                        lambda u, m: {"url": u, "title": "Harbor", "text": "word " * 900})
    ingest.ingest_web_source(card, user)
    status, data = _node(db, card)
    assert status == "ready" and data["title"] == "Harbor" and data["meta"] == "harbor.dev"
    assert data["summary"] == "An overview."
    with db.cursor() as cur:
        cur.execute("SELECT id, file_type FROM external_files WHERE node_id=%s", (card,))
        assert cur.fetchall() == [(f"file_{card}", "text/html")]
        cur.execute("SELECT count(*) FROM file_chunks WHERE file_id=%s AND metadata->>'loc' LIKE '§ %%'",
                    (f"file_{card}",))
        assert cur.fetchone()[0] > 0
    n = _chunk_count(db, user, card)
    ingest.ingest_web_source(card, user)
    assert _chunk_count(db, user, card) == n


@requires_db
def test_ingest_web_refused_marks_failed(db, user, stubs, monkeypatch):
    board, card = _web_card(db, user)
    monkeypatch.setattr(ingest, "fetch_url_content", lambda u, m: None)
    ingest.ingest_web_source(card, user)
    status, data = _node(db, card)
    assert status == "failed" and data["error"] == "Couldn't read this page"


# ── fix round 1 ──────────────────────────────────────────────────────────────
@requires_db
def test_overlapping_ingests_leave_one_chunk_set(db, user, stubs, monkeypatch):
    import threading
    import time
    board, card, fid = _doc_card(db, user)
    ingest.ingest_file_source(card, user)
    expected = _chunk_count(db, user, card)
    with db.cursor() as cur:
        cur.execute("UPDATE nodes SET status='processing' WHERE id=%s", (card,))

    def slow(texts):
        time.sleep(0.4)
        return [unit_emb(i) for i, _ in enumerate(texts)]
    monkeypatch.setattr(ingest, "get_embeddings", slow)
    ts = [threading.Thread(target=ingest.ingest_file_source, args=(card, user)) for _ in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert _chunk_count(db, user, card) == expected
    with db.cursor() as cur:
        cur.execute("SELECT count(*), count(DISTINCT chunk_index) FROM file_chunks WHERE file_id=%s", (fid,))
        assert cur.fetchone() == (expected, expected)
    assert _node(db, card)[0] == "ready"


@requires_db
def test_card_reset_mid_ingest_is_not_overwritten(db, user, stubs, monkeypatch):
    board, card, fid = _doc_card(db, user)

    def reset_then_embed(texts):
        with db.cursor() as cur:
            cur.execute("UPDATE nodes SET status='idle' WHERE id=%s", (card,))
        return [unit_emb(i) for i, _ in enumerate(texts)]
    monkeypatch.setattr(ingest, "get_embeddings", reset_then_embed)
    ingest.ingest_file_source(card, user)
    status, data = _node(db, card)
    assert status == "idle" and "summary" not in data


@requires_db
def test_deleted_card_mid_ingest_is_not_overwritten(db, user, stubs, monkeypatch):
    board, card, fid = _doc_card(db, user)

    def delete_then_embed(texts):
        with db.cursor() as cur:
            cur.execute("UPDATE nodes SET deleted_at=now() WHERE id=%s", (card,))
        return [unit_emb(i) for i, _ in enumerate(texts)]
    monkeypatch.setattr(ingest, "get_embeddings", delete_then_embed)
    ingest.ingest_file_source(card, user)
    status, data = _node(db, card)
    assert status == "processing" and "summary" not in data


@requires_db
def test_set_card_status_wrong_user_leaves_card(db, user, stubs):
    from app.agent.store.board_store import store
    board, card, fid = _doc_card(db, user)
    ingest.set_card_status(store, card, "failed", {"error": "x"}, "someone-else@example.com")
    status, data = _node(db, card)
    assert status == "processing" and "error" not in data


@requires_db
def test_chunk_cap_truncates(db, user, stubs, monkeypatch):
    monkeypatch.setattr(ingest, "MAX_EMBED_CHUNKS", 3)
    csv_bytes = ("a,b\n" + "".join(f"{i},x\n" for i in range(1, 251))).encode()
    board, card, fid = _doc_card(db, user, name="t.csv", mime="text/csv", data=csv_bytes, source_type="csv")
    ingest.ingest_file_source(card, user)
    assert _node(db, card)[0] == "ready" and _chunk_count(db, user, card) == 3


def test_csv_cells_with_commas_and_quotes_stay_unambiguous():
    chunks, _, _ = parse_csv(b'a,b\n"x, y","say ""hi"""\n')
    assert chunks[0] == 'a,b\n"x, y","say ""hi"""'


@requires_db
def test_stored_overview_is_clipped(db, user, monkeypatch):
    monkeypatch.setattr(ingest, "get_embeddings", lambda texts: [unit_emb(i) for i, _ in enumerate(texts)])
    monkeypatch.setattr(ingest, "get_embedding", lambda t: unit_emb(0))
    monkeypatch.setattr(ingest, "get_llm", lambda *a, **k: FakeLLM("o" * 5000))
    board, card, fid = _doc_card(db, user)
    ingest.ingest_file_source(card, user)
    assert len(_node(db, card)[1]["summary"]) <= 600


@requires_db
def test_no_connection_held_during_embed_or_llm(db, user, monkeypatch):
    from contextlib import contextmanager
    from app.agent.store.board_store import store as bs
    real = bs._get_conn
    state = {"open": 0, "seen": []}

    @contextmanager
    def tracked():
        state["open"] += 1
        try:
            with real() as conn:
                yield conn
        finally:
            state["open"] -= 1

    def emb(texts):
        state["seen"].append(("embed", state["open"]))
        return [unit_emb(i) for i, _ in enumerate(texts)]

    def one(t):
        state["seen"].append(("embed1", state["open"]))
        return unit_emb(0)

    class SeenLLM(FakeLLM):
        def invoke(self, prompt):
            state["seen"].append(("llm", state["open"]))
            return super().invoke(prompt)

    monkeypatch.setattr(ingest, "get_embeddings", emb)
    monkeypatch.setattr(ingest, "get_embedding", one)
    monkeypatch.setattr(ingest, "get_llm", lambda *a, **k: SeenLLM("ok"))
    board, card, fid = _doc_card(db, user)
    monkeypatch.setattr(bs, "_get_conn", tracked)
    ingest.ingest_file_source(card, user)
    assert _node(db, card)[0] == "ready"
    assert {k for k, _ in state["seen"]} == {"embed", "embed1", "llm"}
    assert all(n == 0 for _, n in state["seen"])


@requires_db
def test_write_chunks_false_marks_card_failed(api, db, monkeypatch):
    from app.agent import ingest
    from tests.v3_support import make_board, seed_card, unit_emb
    b = make_board(api)["id"]
    cid = seed_card(db, api.email, b, kind="source", source_type="doc", status="processing")
    monkeypatch.setattr(ingest, "get_embeddings", lambda cs: [unit_emb(0) for _ in cs])
    monkeypatch.setattr(ingest, "get_embedding", lambda t: unit_emb(0))
    monkeypatch.setattr(ingest, "make_overview", lambda t, e: "sum")
    monkeypatch.setattr(ingest, "_write_chunks", lambda *a: False)
    ingest._embed_and_finish(cid, "file_gone", api.email, ["a"], [{}], "a", {})
    with db.cursor() as cur:
        cur.execute("SELECT status FROM nodes WHERE id = %s", (cid,))
        assert cur.fetchone()[0] == "failed"
