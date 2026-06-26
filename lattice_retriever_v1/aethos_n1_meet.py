"""
AETHOS n=1 meet read — PDF §4–§5 (Dec 14 2025 spec).

When transgressor n=1 and n < a < p (Case 1), VA1A is:
    (a+p, a, a+p+n)

Case number is 1 for every pair at n=1 — disambiguation is the full formula row
(a+p unique per right prime) plus pair_n tail consistency, not case id alone.
"""

from __future__ import annotations

from collections import defaultdict

from aethos_lattice import prime_pair_canon, prime_pair_case
from lattice_retriever_v1.intersection_dot_codec import SymbolAlphabet
from lattice_retriever_v1.stage05_free_token import canonical_pair


def va1a_meet_coord(left_prime: int, right_prime: int, n: int) -> tuple[int, int, int]:
    """Section 5.1 VA1A — anchors a ≤ p."""
    a, p = canonical_pair(left_prime, right_prime)
    return prime_pair_canon("VA1A", a, p, n)


def va1_meet_coord(anchor_prime: int, n: int) -> tuple[int, int, int]:
    """Section 4.1 VA1 — single-prime branch."""
    from aethos_lattice import BranchKind, single_prime_canon

    return single_prime_canon(BranchKind.VA1, anchor_prime, n)


def meet_case(left_prime: int, right_prime: int, n: int) -> int:
    a, p = canonical_pair(left_prime, right_prime)
    return prime_pair_case(a, p, n)


def _pair_counts(prefix: bytes) -> dict[tuple[int, int], int]:
    pc: dict[tuple[int, int], int] = defaultdict(int)
    for i in range(len(prefix) - 1):
        pc[(prefix[i], prefix[i + 1])] += 1
    return pc


def _candidates(
    alpha: SymbolAlphabet, prefix: bytes, pn: int, pc: dict[tuple[int, int], int]
) -> list[int]:
    left = prefix[-1]
    return [s for s in alpha.symbols if pc[(left, s)] + 1 == pn]


def _has_completion(
    alpha: SymbolAlphabet,
    prefix: bytes,
    pair_ns: list[int],
    wi: int,
    count: int,
) -> bool:
    if len(prefix) == count:
        return True
    if wi >= len(pair_ns):
        return False
    pc = _pair_counts(prefix)
    cands = _candidates(alpha, prefix, pair_ns[wi], pc)
    for s in cands:
        if _has_completion(alpha, prefix + bytes([s]), pair_ns, wi + 1, count):
            return True
    return False


def pick_symbol_at_meet(
    alpha: SymbolAlphabet,
    prefix: bytes,
    pair_ns: list[int],
    wi: int,
    count: int,
    pair_counts: dict[tuple[int, int], int],
) -> int:
    """
    3-way meet read — VA1A formula + corridor tail check.
    At n=1 every candidate has case 1; completion path selects the symbol.
    """
    pn = pair_ns[wi]
    cands = _candidates(alpha, prefix, pn, pair_counts)
    if len(cands) == 1:
        return cands[0]

    viable = [
        s
        for s in cands
        if _has_completion(alpha, prefix + bytes([s]), pair_ns, wi + 1, count)
    ]
    if len(viable) == 1:
        return viable[0]

    left_p = alpha.prime_for(prefix[-1])
    if len(viable) > 1:
        return min(viable, key=lambda x: alpha.prime_for(x))
    raise ValueError("formula meet read failed")


def meet_outer_fingerprint(left_prime: int, right_prime: int, n: int) -> int:
    return va1a_meet_coord(left_prime, right_prime, n)[0]
