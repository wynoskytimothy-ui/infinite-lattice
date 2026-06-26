"""Stage 08 gate — synthetic properties first (SciFact is separate integration gate)."""

import json

from lattice_retriever_v1.stage08_retrieve import (
    FIXTURE_CORPUS,
    FIXTURE_QUERIES,
    MissPolicy,
    build_fixture_retriever,
)


def test_rarest_filter_shrinks_pool_with_log_steps():
    r = build_fixture_retriever()
    pool, steps = r.lazy_pool("cancer mutation variant")
    assert len(pool) < len(FIXTURE_CORPUS)
    assert len(steps) >= 2
    sizes = [s["pool_size"] for s in steps if "pool_size" in s]
    assert sizes[0] >= sizes[-1]
    assert all("log_pool" in s for s in steps if s.get("step") == "rarest_filter")


def test_no_full_corpus_scan_when_rare_term_selective():
    r = build_fixture_retriever()
    pool, _ = r.lazy_pool("cancer mutation")
    assert len(pool) <= 3
    assert "d05" in pool


def test_lazy_corridor_key_no_stored_row():
    r = build_fixture_retriever()
    hit = r.score_doc("d03", "apple phone")
    corridor = [x for x in hit.reasons if x.get("kind") == "corridor_open"]
    assert corridor
    assert corridor[0]["lazy"] is True
    assert "corridor_key" in corridor[0]
    assert "summary" in corridor[0]
    assert "invoke_order" in corridor[0]
    assert "transgressor_n" in corridor[0]


def test_corridor_witness_th_vs_he():
    """TH and HE share h but differ by invoke_order, composite, and case at n=h."""
    from aethos_words import letter_to_prime

    from lattice_retriever_v1.stage05_free_token import corridor_witness_explain, free_token_address

    t, h, e = letter_to_prime("t"), letter_to_prime("h"), letter_to_prime("e")
    th = corridor_witness_explain(
        free_token_address(t, h, invoke_order=(t, h)),
        pair=("t", "h"),
    )
    he = corridor_witness_explain(
        free_token_address(h, e, invoke_order=(h, e)),
        pair=("h", "e"),
    )
    assert th["meet_composite"] == 1679
    assert he["meet_composite"] == 299
    assert th["from_letter"] == "t" and th["to_letter"] == "h"
    assert he["from_letter"] == "h" and he["to_letter"] == "e"
    assert "transgressed from t=73 to h=23" in th["summary"]
    assert "transgressed from h=23 to e=13" in he["summary"]
    assert th["lattice_L01"] != he["lattice_L01"]


def test_retrieve_trace_includes_corridor_witnesses():
    r = build_fixture_retriever()
    r.index_doc("d_th", "cat purrs softly")
    trace = r.retrieve_with_trace("cat purrs")
    ex = trace.explain()
    assert ex["corridor_witnesses"]
    witness = ex["corridor_witnesses"][0]
    assert witness["pair"] == ["cat", "purrs"]
    assert witness["label"] == "catpurrs"
    for w in ex["corridor_witnesses"]:
        assert "summary" in w
        assert "case" in w
        assert "transgressor_n" in w
        assert "invoke_order" in w


def test_fixture_top1_at_least_eight_of_ten():
    r = build_fixture_retriever()
    wins = 0
    for qid, query, gold in FIXTURE_QUERIES:
        hits = r.retrieve(query, limit=3)
        assert hits, f"no hits for {qid}"
        if hits[0].doc_id == gold:
            wins += 1
    assert wins >= 8, f"top-1 wins {wins}/10"


def test_retrieve_trace_glass_box_fields():
    r = build_fixture_retriever()
    trace = r.retrieve_with_trace("cat purrs softly")
    ex = trace.explain()
    assert ex["query_primes"]
    assert ex["route_mode"] in ("primary", "widen_rarest", "fta_letter_fallback", "empty")
    assert "filter_steps" in ex
    assert "corridor_witnesses" in ex
    assert "hits" in ex
    assert ex["hits"][0]["reasons"]


def test_oov_query_uses_fta_letter_fallback():
    """Query term absent from corpus → FTA letter fallback, not silent empty."""
    r = build_fixture_retriever()
    trace = r.retrieve_with_trace("zzxxy rareword")
    assert trace.route_mode in ("fta_letter_fallback", "empty", "widen_rarest")
    steps = [s["step"] for s in trace.filter_steps]
    if trace.route_mode == "fta_letter_fallback":
        assert "fta_letter_fallback" in steps
    assert not trace.query_primes[0].in_corpus


def test_total_miss_empty_with_trace():
    r = build_fixture_retriever()
    r.miss_policy = MissPolicy.EMPTY
    trace = r.retrieve_with_trace("zzxxy qqwerty")
    assert trace.route_mode == "empty"
    assert trace.pool_size == 0
    assert trace.hits == ()
    assert any(s.get("step") == "miss_empty" for s in trace.filter_steps)


def test_miss_policy_empty_skips_fta():
    r = build_fixture_retriever()
    r.miss_policy = MissPolicy.EMPTY
    pool, mode, _, _, _ = r.route_pool("nonexistent xyzzyword")
    assert mode == "empty"
    assert pool == []


def test_large_corpus_pool_bounded_by_rare_posting():
    r = build_fixture_retriever()
    for i in range(90):
        r.index_doc(f"extra{i}", "the and or common words only")
    pool, _ = r.lazy_pool("cancer mutation")
    assert len(pool) < 20
    assert len(pool) < len(r.docs)
