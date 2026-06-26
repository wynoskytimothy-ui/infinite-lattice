"""
k-meet index helpers — velocity witness + conservative pool widen pins.

Uses compose_k, velocity_meet, pair_branch_compose from k_meet algebra.
No MultiCorpusBrain / BM25 / embedding backends.
"""

from __future__ import annotations

from typing import Callable

from lattice_retriever_v1.k_meet import compose_k, pair_branch_compose, velocity_meet
from lattice_retriever_v1.stage06_composites import meet_composite_k


def _sorted_distinct(*primes: int) -> tuple[int, ...]:
    return tuple(sorted(set(primes)))


def query_primes_from_terms(
    terms: tuple[str, ...] | list[str],
    prime_for_term: Callable[[str], int],
) -> tuple[int, ...]:
    """Distinct identity primes for query terms (read order preserved in tuple)."""
    seen: set[int] = set()
    out: list[int] = []
    for t in terms:
        p = prime_for_term(t)
        if p not in seen:
            seen.add(p)
            out.append(p)
    return tuple(out)


def velocity_witness_primes(query_primes: tuple[int, ...]) -> list[dict]:
    """
    Glass-box witness for k≥3 chains in query.

    Conservative: only sorted contiguous subchains of distinct primes with
    unified velocity extension are reported.
    """
    ps = _sorted_distinct(*query_primes)
    if len(ps) < 3:
        return []
    witnesses: list[dict] = []
    for start in range(len(ps)):
        for end in range(start + 3, len(ps) + 1):
            chain = ps[start:end]
            vel = velocity_meet(*chain)
            if vel is None or not vel.unified:
                continue
            entry: dict = {
                "chain": list(chain),
                "k": len(chain),
                "velocity": vel.explain(),
                "compose": compose_k(*chain).explain(),
            }
            if len(chain) == 3:
                pbc = pair_branch_compose(chain[0], chain[1], chain[2])
                if pbc is not None:
                    coord, n_deep = pbc
                    entry["pair_branch_compose"] = {
                        "coord": list(coord),
                        "n_deep": n_deep,
                    }
            witnesses.append(entry)
    return witnesses


def widen_pins_from_velocity(
    query_primes: tuple[int, ...] | None = None,
    *,
    query_terms: tuple[str, ...] | list[str] | None = None,
    existing_pins: set[int] | frozenset[int] | None = None,
    prime_for_term: Callable[[str], int] | None = None,
) -> set[int]:
    """
    Add composite pins from velocity extension when unified.

    Conservative: only when 3+ distinct query primes form a sorted chain with
    unified velocity meet. Returns pins not already in *existing_pins*.
    """
    if query_primes is None:
        if query_terms is None or prime_for_term is None:
            return set()
        query_primes = query_primes_from_terms(query_terms, prime_for_term)
    ps = _sorted_distinct(*query_primes)
    if len(ps) < 3:
        return set()
    have = set(existing_pins or ())
    out: set[int] = set()
    for start in range(len(ps)):
        for end in range(start + 3, len(ps) + 1):
            chain = ps[start:end]
            vel = velocity_meet(*chain)
            if vel is None or not vel.unified:
                continue
            try:
                comp = meet_composite_k(*chain)
            except (ValueError, TypeError):
                continue
            if comp not in have:
                out.add(comp)
    return out


def k_meet_index_report(
    query_primes: tuple[int, ...],
) -> dict:
    """Full glass-box index report for query prime chain."""
    return {
        "query_primes": list(query_primes),
        "velocity_witnesses": velocity_witness_primes(query_primes),
        "widen_pins": sorted(
            widen_pins_from_velocity(query_primes, existing_pins=frozenset())
        ),
    }
