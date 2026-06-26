"""AETHOS PDF §4–§5 — n=1 VA1A meet is unique per pair."""

from __future__ import annotations

from lattice_retriever_v1.aethos_n1_meet import (
    meet_case,
    meet_outer_fingerprint,
    va1_meet_coord,
    va1a_meet_coord,
)
from lattice_retriever_v1.intersection_dot_codec import SymbolAlphabet


def test_n1_all_case_one_distinct_coords():
    alpha = SymbolAlphabet(symbols=tuple(range(10)))
    lp = alpha.prime_for(0)
    coords = {va1a_meet_coord(lp, alpha.prime_for(s), 1) for s in range(10)}
    assert len(coords) == 10
    assert all(meet_case(lp, alpha.prime_for(s), 1) == 1 for s in range(10))


def test_va1_single_prime_n1():
    assert va1_meet_coord(5, 1) == (6, 1, 6)


def test_pick_at_meet_ten_digit_first_step():
    alpha = SymbolAlphabet(symbols=tuple(range(10)))
    lp = alpha.prime_for(0)
    outer = meet_outer_fingerprint(lp, alpha.prime_for(1), 1)
    assert outer == 8
