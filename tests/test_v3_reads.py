from math import ceil

from tests.v3_seed import prototype_graph, NOTE_TEXT, PDF_DESC
from tests.v3_support import (  # noqa: F401
    requires_db, db, user, api, make_board, new_id, seed_board, seed_card, seed_edge, seed_message)
from app.agent.reads import compute_reads, estimate_tokens


def ids(r):
    return [(i.card_id, i.how, i.via, i.inherited) for i in r.items]


def test_desk_reads_main():
    r = compute_reads(prototype_graph(), "main")
    assert ids(r) == [("video", "desk", "main", False), ("pdf", "desk", "main", False), ("note", "desk", "main", False)]
    assert [i.state for i in r.items] == ["soon", "ready", "ready"]
    assert r.total_tokens == ceil((min(5000, 5 * 900) + len(PDF_DESC)) / 4) + ceil(len(NOTE_TEXT) / 4)


def test_inherit_on_side():
    assert ids(compute_reads(prototype_graph(), "side")) == [
        ("video", "desk", "main", True), ("pdf", "desk", "main", True), ("note", "desk", "main", True)]


def test_pile_expands_members_second():
    r = compute_reads(prototype_graph(), "second")
    assert [i.card_id for i in r.items] == ["web", "iv1", "iv2", "iv3", "iv4", "video", "pdf", "note"]
    assert "interviews" not in [i.card_id for i in r.items]
    assert all(i.how == "desk" and i.via == "second" for i in r.items[:5])


def test_hiding_pile_hides_all_members():
    g = prototype_graph(hide={"deep": ["interviews"]})
    r = compute_reads(g, "deep")
    members = [i for i in r.items if i.card_id in ("iv1", "iv2", "iv3", "iv4")]
    assert len(members) == 4 and all(i.hidden and i.inherited for i in members)
    assert r.total_tokens == sum(i.tokens for i in r.items if i.state == "ready" and not i.card_id.startswith("iv"))
    assert [i.card_id for i in r.readable()] == [i.card_id for i in r.items if not i.hidden and i.state == "ready"]


def test_pile_member_switched_off_excluded():
    g = prototype_graph(off={"iv3"})
    assert "iv3" not in [i.card_id for i in compute_reads(g, "second").items]


def test_wire_and_answer_deep():
    r = compute_reads(prototype_graph(), "deep")
    assert ids(r)[:3] == [("shot", "desk", "deep", False), ("csv", "wire", "deep", False), ("side", "answer", "deep", False)]
    assert [(i.card_id, i.via) for i in r.items[3:]] == [
        ("web", "second"), ("iv1", "second"), ("iv2", "second"), ("iv3", "second"), ("iv4", "second"),
        ("video", "main"), ("pdf", "main"), ("note", "main")]
    assert all(i.inherited for i in r.items[3:])


def test_inherit_off():
    g = prototype_graph(no_inherit={"second"})
    assert [i.card_id for i in compute_reads(g, "second").items] == ["web", "iv1", "iv2", "iv3", "iv4"]
    assert "pdf" not in [i.card_id for i in compute_reads(g, "deep").items]


def test_hide_lists_hidden_and_drops_for_branches():
    g = prototype_graph(hide={"deep": ["pdf"]}, extra_chats={"deeper": "deep"})
    r = compute_reads(g, "deep")
    [pdf] = [i for i in r.items if i.card_id == "pdf"]
    assert pdf.hidden and pdf.inherited
    assert r.total_tokens == sum(i.tokens for i in r.items if i.state == "ready" and i.card_id != "pdf")
    assert "pdf" not in [i.card_id for i in compute_reads(g, "deeper").items]


def test_own_beats_inherited():
    g = prototype_graph(extra_wires=[("csv", "second")])
    [csv] = [i for i in compute_reads(g, "deep").items if i.card_id == "csv"]
    assert (csv.how, csv.via, csv.inherited) == ("wire", "deep", False)


def test_soon_processing_failed_listed_not_counted():
    g = prototype_graph(status={"pdf": "processing", "note": "failed"})
    r = compute_reads(g, "main")
    assert [i.state for i in r.items] == ["soon", "processing", "failed"] and r.total_tokens == 0
    assert all(i.tokens == 0 for i in r.items)


def test_answer_waiting():
    g = prototype_graph(extra_chats={"q2": None}, extra_answers=[("q2", "main")], no_answer={"q2"})
    [w] = [i for i in compute_reads(g, "main").items if i.card_id == "q2"]
    assert (w.how, w.state, w.tokens) == ("answer", "waiting", 0)


def test_answer_chat_tokens_clipped():
    g = prototype_graph()
    g.answer_chars["side"] = 9000
    g.summary_chars["side"] = 5000
    assert estimate_tokens(g.cards["side"], g, 5) == ceil((3000 + 1200) / 4)


def test_top_k_from_chat_advanced():
    g = prototype_graph()
    g.cards["main"].data["advanced"] = {"externalContextTopK": 2}
    [pdf] = [i for i in compute_reads(g, "main").items if i.card_id == "pdf"]
    assert pdf.tokens == ceil((2 * 900 + len(PDF_DESC)) / 4)


def test_reads_to_json_contract_shape():
    j = compute_reads(prototype_graph(), "deep").to_json()
    assert set(j) == {"chatId", "items", "totalTokens"}
    assert set(j["items"][0]) == {"cardId", "how", "via", "inherited", "hidden", "state", "tokens"}


# ── DB-backed route tests ────────────────────────────────────────────────────

@requires_db
def test_reads_route_deep(api, db, user):
    b = make_board(api)["id"]
    main = seed_card(db, user, b, kind="chat")
    second = seed_card(db, user, b, kind="chat", parent_id=main)
    deep = seed_card(db, user, b, kind="chat", parent_id=second)
    pdf = seed_card(db, user, b, kind="source", source_type="pdf", status="ready", desk_id=main,
                    data={"desc": "d"})
    csv = seed_card(db, user, b, kind="source", source_type="csv", status="ready")
    gone = seed_card(db, user, b, kind="source", source_type="note", status="ready", desk_id=deep,
                     deleted_at="2026-01-01")
    seed_edge(db, user, b, csv, deep, "wire")
    fid = new_id("f")
    with db.cursor() as cur:
        cur.execute("INSERT INTO external_files (id, node_id, canvas_id, user_email, file_name, file_type) "
                    "VALUES (%s,%s,%s,%s,'f','pdf')", (fid, pdf, b, user))
        cur.execute("INSERT INTO file_chunks (file_id, chunk_index, chunk_text) VALUES (%s,0,%s),(%s,1,%s)",
                    (fid, "a" * 100, fid, "b" * 60))
    for c in (main, second, deep):
        seed_message(db, user, c, "assistant", "x" * 10, 1)
    r = api.get(f"/boards/{b}/cards/{deep}/reads")
    assert r.status_code == 200, r.text
    j = r.json()
    by = {i["cardId"]: i for i in j["items"]}
    assert gone not in by
    assert (by[csv]["how"], by[csv]["via"], by[csv]["inherited"]) == ("wire", deep, False)
    assert (by[pdf]["how"], by[pdf]["via"], by[pdf]["inherited"]) == ("desk", main, True)
    assert by[pdf]["tokens"] == ceil((160 + 1) / 4)
    assert j["chatId"] == deep


@requires_db
def test_reads_route_non_chat_400(api, db, user):
    b = make_board(api)["id"]
    s = seed_card(db, user, b, kind="source", source_type="note", status="ready")
    r = api.get(f"/boards/{b}/cards/{s}/reads")
    assert r.status_code == 400 and r.json()["error"] == "invalid"


@requires_db
def test_reads_route_other_user_404(api, db, user):
    other = f"o_{new_id('u')}@example.com"
    with db.cursor() as cur:
        cur.execute("INSERT INTO users (email, name) VALUES (%s,'o')", (other,))
    try:
        ob = seed_board(db, other)
        oc = seed_card(db, other, ob, kind="chat")
        assert api.get(f"/boards/{ob}/cards/{oc}/reads").status_code == 404
        b = make_board(api)["id"]
        assert api.get(f"/boards/{b}/cards/nope/reads").status_code == 404
    finally:
        with db.cursor() as cur:
            for t in ("messages", "edges", "nodes", "canvases"):
                cur.execute(f"DELETE FROM {t} WHERE user_email = %s", (other,))
            cur.execute("DELETE FROM users WHERE email = %s", (other,))
