"""k-meet velocity widen — synthetic pool recovery."""

from __future__ import annotations

from lattice_retriever_v1.k_meet import velocity_meet
from lattice_retriever_v1.k_meet_index import widen_pins_from_velocity
from lattice_retriever_v1.stage06_composites import meet_composite_k
from lattice_retriever_v1.stage08_retrieve import LatticeRetriever


def test_velocity_widen_fills_empty_pool(monkeypatch) -> None:
    """Primary intersect empty; velocity composite pin indexed on gold doc recovers pool."""
    r = LatticeRetriever(enable_k_meet_velocity_widen=True, k_meet_min_pool_size=10)
    t1, t2, t3 = "zzalpha", "zzbeta", "zzgamma"
    ps = (3, 5, 7)
    assert velocity_meet(*ps) is not None and velocity_meet(*ps).unified
    comp = meet_composite_k(*ps)

    real_id = r._identity_for

    def _fake_id(term: str) -> int:
        return {t1: 3, t2: 5, t3: 7}.get(term, real_id(term))

    monkeypatch.setattr(r, "_identity_for", _fake_id)

    r.index_doc("gold", f"{t1} {t2} {t3} marker")
    r.index_doc("decoy_a", f"{t1} only common filler")
    r.index_doc("decoy_b", f"{t2} only common filler")
    r.index_doc("decoy_c", f"{t3} only common filler")

    widen = widen_pins_from_velocity(ps, existing_pins=frozenset())
    assert comp in widen
    assert comp in r.postings and "gold" in r.postings[comp]

    query = f"{t1} {t2} {t3}"
    trace = r.retrieve_with_trace(query, limit=5)
    pool_ids = {h.doc_id for h in trace.hits}
    assert "gold" in pool_ids
    assert any(s.get("step") == "k_meet_velocity_widen" for s in trace.filter_steps)
    assert trace.velocity_witness
