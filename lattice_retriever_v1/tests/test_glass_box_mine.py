"""Synthetic gate — glass-box mine lenses on fixture corpus."""

from lattice_retriever_v1.glass_box_mine import aggregate_lenses, headroom_summary, mine_query
from lattice_retriever_v1.stage08_retrieve import FIXTURE_CORPUS, FIXTURE_QUERIES, build_fixture_retriever


def test_mine_fixture_queries():
    r = build_fixture_retriever()
    records = []
    for qid, query, gold in FIXTURE_QUERIES:
        records.append(mine_query(r, qid, query, [gold]))
    lenses = aggregate_lenses(records)
    hr = headroom_summary(records)
    assert hr["n_queries"] == 10
    assert hr["pool_recall"] >= 0.8
    l01 = next(x for x in lenses if x.id == "L01_gold_in_pool")
    assert l01.gold_hits >= 8


def test_rarest_term_lens_on_fixture():
    r = build_fixture_retriever()
    rec = mine_query(r, "q05", "cancer mutation", ["d05"])
    assert rec.buckets.get("L04_gold_has_rarest_term") or rec.buckets.get("L05_rarest_pin_hits_gold")
    assert rec.gold_in_pool
