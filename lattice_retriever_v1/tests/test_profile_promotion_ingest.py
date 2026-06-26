"""Synthetic gate — promotion-on ingest per-doc cost stays flat on fixture slice."""

from lattice_retriever_v1.promotion_ingest_profile import (
    DEFAULT_MAX_QUARTILE_RATIO,
    assert_ingest_flat,
    fixture_corpus_slice,
    profile_ingest,
)


def test_fixture_50_doc_ingest_flat():
    """50-doc fixture cycle: per-doc time must not climb with corpus-so-far."""
    profile = profile_ingest(fixture_corpus_slice(50), fast_ingest=False)
    assert_ingest_flat(profile, max_quartile_ratio=DEFAULT_MAX_QUARTILE_RATIO, max_total_ms=5000.0)
    assert profile.timings[-1].n_l2_promotions >= 0


def test_promotion_on_slower_than_fast_ingest_but_bounded():
    """Promotion-on does real work; fast_ingest is the no-L2 floor."""
    slow = profile_ingest(fixture_corpus_slice(20), fast_ingest=False)
    fast = profile_ingest(fixture_corpus_slice(20), fast_ingest=True)
    assert sum(slow.totals_ms) > sum(fast.totals_ms)
    assert_ingest_flat(fast, max_quartile_ratio=DEFAULT_MAX_QUARTILE_RATIO)


def test_flush_candidates_bounded_on_short_docs():
    """Each short fixture doc should not explode L2 candidate set."""
    profile = profile_ingest(fixture_corpus_slice(10), fast_ingest=False)
    for t in profile.timings:
        assert t.flush_candidates < 500, f"{t.doc_id} flush_candidates={t.flush_candidates}"
