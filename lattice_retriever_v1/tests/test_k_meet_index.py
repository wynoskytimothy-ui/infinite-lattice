"""k-meet index — velocity witness + widen pins."""

from __future__ import annotations

from lattice_retriever_v1.k_meet_index import (
    k_meet_index_report,
    query_primes_from_terms,
    velocity_witness_primes,
    widen_pins_from_velocity,
)


def test_velocity_witness_k3_chain():
    witnesses = velocity_witness_primes((3, 5, 7))
    assert len(witnesses) == 1
    w = witnesses[0]
    assert w["k"] == 3
    assert w["velocity"]["unified"] is True
    assert w["pair_branch_compose"]["n_deep"] == 5


def test_velocity_witness_requires_three_primes():
    assert velocity_witness_primes((3, 5)) == []
    assert velocity_witness_primes(()) == []


def test_widen_pins_conservative_unified_only():
    pins = widen_pins_from_velocity((3, 5, 7), existing_pins=frozenset({3, 5, 7}))
    assert 3 * 5 * 7 in pins
    assert 3 not in pins


def test_widen_pins_skips_non_unified_chain():
    assert widen_pins_from_velocity((2, 4, 8), existing_pins=frozenset()) == set()


def test_query_primes_from_terms():
    primes = query_primes_from_terms(("cat", "dog", "ee"), lambda t: len(t))
    assert primes == (3, 2)


def test_k_meet_index_report():
    rep = k_meet_index_report((3, 5, 7, 11))
    assert rep["velocity_witnesses"]
    assert rep["widen_pins"]
