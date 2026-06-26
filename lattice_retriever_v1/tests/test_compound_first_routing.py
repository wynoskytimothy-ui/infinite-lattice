"""Gate: L37 compound-first — compound unit before intersect, retry without extra terms."""

from lattice_retriever_v1.stage08_retrieve import LatticeRetriever, MissPolicy


def _build_compound_retry_fixture() -> LatticeRetriever:
    r = LatticeRetriever()
    r.miss_policy = MissPolicy.WIDEN_RAREST_ONLY
    for i in range(5):
        r.index_doc(f"quxonly{i}", "qux zzfiller text only")
    for i in range(5):
        r.index_doc(f"pair{i}", "foo bar zznoise")
    r.index_doc("gold", "foo bar baz target")
    return r


def test_routing_units_compound_before_terms():
    r = _build_compound_retry_fixture()
    units = r._routing_units("foo bar qux")
    kinds = [u["kind"] for u in units if u["kind"] != "hub_skip"]
    assert kinds[0] == "compound"


def test_compound_preferred_retry_recovers_gold():
    r = _build_compound_retry_fixture()
    pool, mode, steps, _, _ = r.route_pool("foo bar qux")
    assert mode == "primary"
    assert "gold" in pool
    assert any(s.get("step") == "compound_first_retry" for s in steps)


def test_compound_retry_off_falls_through_without_gold_in_primary():
    r = _build_compound_retry_fixture()
    r.enable_compound_first_routing = False
    pool, mode, _, _, _ = r.route_pool("foo bar qux")
    assert mode == "widen_rarest"
    assert "gold" not in pool
