"""
Gate: filter standalone hubs, preserve compound identities.

High-df standalone "cell" must not narrow; compound stem×cell / cell×phone
carry their own low df and route precisely.
"""

from lattice_retriever_v1.stage08_retrieve import LatticeRetriever


def _build_compound_fixture() -> LatticeRetriever:
    r = LatticeRetriever()
    for i in range(40):
        r.index_doc(f"noise{i}", "cell biology tissue sample")
    r.index_doc("stem_doc", "stem cell therapy works")
    r.index_doc("phone_doc", "cell phone sales rise")
    assert r.semantic.doc_freq["cell"] >= 40
    return r


def test_standalone_cell_is_hub_skipped():
    r = _build_compound_fixture()
    units = r._routing_units("stem cell")
    assert not any(u["kind"] == "term" and u["terms"][0] == "cell" for u in units)


def test_stem_cell_routes_compound_not_phone_doc():
    r = _build_compound_fixture()
    pool, mode, steps, _, _ = r.route_pool("stem cell")
    assert "stem_doc" in pool
    assert "phone_doc" not in pool
    compound = next(s for s in steps if s.get("kind") == "compound")
    assert compound["terms"] == ["stem", "cell"]
    assert compound["pin_doc_freq"] == 1


def test_cell_phone_routes_phone_doc_not_stem():
    r = _build_compound_fixture()
    pool, _, steps, _, _ = r.route_pool("cell phone")
    assert "phone_doc" in pool
    assert "stem_doc" not in pool
    assert any(s.get("kind") == "compound" for s in steps)
