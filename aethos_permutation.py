"""
Prime permutation side-channel: k primes -> k! orderings via tiny 3D offset.

Canonical lattice address uses sorted primes (unique set).
Application ORDER is encoded by moving the dot slightly off the canonical point:
  k=2 -> 2 permutations, k=3 -> 6, k=4 -> 24, ...
The offset is sub-lattice (epsilon) so the dot still "looks" like one intersection.
"""

from __future__ import annotations

import math
from functools import lru_cache
from itertools import permutations
from typing import Sequence

# Side offset scale — must be smaller than 1 so integer lattice steps dominate meaning
PERM_EPSILON = 1e-3


def permutation_count(k: int) -> int:
    return math.factorial(k)


@lru_cache(maxsize=32)
def ordered_permutation_list(sorted_primes: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """All k! orderings of distinct primes (deterministic sort order of permutations)."""
    return tuple(sorted(permutations(sorted_primes)))


def permutation_index(order: Sequence[int]) -> int:
    """Index 0 .. k!-1 for this ordered tuple of distinct primes."""
    order_t = tuple(order)
    sorted_p = tuple(sorted(order_t))
    if len(set(order_t)) != len(order_t):
        raise ValueError("prime order requires distinct primes")
    perms = ordered_permutation_list(sorted_p)
    if order_t not in perms:
        raise ValueError("order uses primes not in sorted set")
    return perms.index(order_t)


def order_from_index(sorted_primes: tuple[int, ...], index: int) -> tuple[int, ...]:
    perms = ordered_permutation_list(sorted_primes)
    if not 0 <= index < len(perms):
        raise ValueError(f"permutation index {index} out of range for k={len(sorted_primes)}")
    return perms[index]


def side_offset(k: int, perm_index: int, epsilon: float = PERM_EPSILON) -> tuple[float, float, float]:
    """
    Map permutation index to a unique tiny offset (golden-angle spiral on side plane).
    k=2 -> 2 distinct points; k=3 -> 6; etc.
    """
    if k <= 0:
        return (0.0, 0.0, 0.0)
    n = permutation_count(k)
    i = perm_index % n
    angle = i * (math.pi * (3.0 - math.sqrt(5.0)))  # golden angle
    r = epsilon * (1.0 + 0.05 * i)
    return (
        r * math.cos(angle),
        r * math.sin(angle),
        i * epsilon * 0.01,
    )


def side_offset_from_sequence(order: Sequence[int], epsilon: float = PERM_EPSILON) -> tuple[float, float, float]:
    """
    Side offset for letter sequences that repeat primes (e.g. apple -> two p anchors).
    Distinct from k! permutation channel used when all primes in order are unique.
    """
    if not order:
        return (0.0, 0.0, 0.0)
    h = 0
    for i, p in enumerate(order):
        h = (h * 997 + int(p) * (i + 1)) & 0xFFFFFFFF
    n = len(order)
    angle = (h % 10_000) * (math.pi * 2.0 / 10_000.0)
    r = epsilon * (1.0 + (h % 50) * 0.01)
    return (
        r * math.cos(angle),
        r * math.sin(angle),
        (h % 1000) * epsilon * 0.001,
    )


def apply_sequence_offset(
    base: tuple[float, float, float],
    order: Sequence[int],
    epsilon: float = PERM_EPSILON,
) -> tuple[float, float, float]:
    """Canonical base + side offset for full letter order (duplicates allowed)."""
    ox, oy, oz = side_offset_from_sequence(order, epsilon)
    bx, by, bz = base
    return (bx + ox, by + oy, bz + oz)


def decode_sequence_from_dot(
    base: tuple[float, float, float],
    dot: tuple[float, float, float],
    candidates: Sequence[tuple[int, ...]],
    epsilon: float = PERM_EPSILON,
) -> tuple[int, ...]:
    """Pick candidate order whose sequence offset best matches dot - base."""
    bx, by, bz = base
    dx, dy, dz = dot[0] - bx, dot[1] - by, dot[2] - bz
    best = candidates[0]
    best_d = float("inf")
    for order in candidates:
        ox, oy, oz = side_offset_from_sequence(order, epsilon)
        dist = (dx - ox) ** 2 + (dy - oy) ** 2 + (dz - oz) ** 2
        if dist < best_d:
            best_d = dist
            best = order
    return best


def apply_order_offset(
    base: tuple[float, float, float],
    sorted_primes: tuple[int, ...],
    order: Sequence[int],
    epsilon: float = PERM_EPSILON,
) -> tuple[float, float, float]:
    """Canonical base + side offset encoding prime application order."""
    idx = permutation_index(order)
    ox, oy, oz = side_offset(len(sorted_primes), idx, epsilon)
    bx, by, bz = base
    return (bx + ox, by + oy, bz + oz)


def nearest_permutation_index(
    base: tuple[float, float, float],
    dot: tuple[float, float, float],
    k: int,
    epsilon: float = PERM_EPSILON,
) -> int:
    """Recover order from how far the dot sits to the side of canonical base."""
    bx, by, bz = base
    dx, dy, dz = dot[0] - bx, dot[1] - by, dot[2] - bz
    best_i = 0
    best_d = float("inf")
    for i in range(permutation_count(k)):
        ox, oy, oz = side_offset(k, i, epsilon)
        dist = (dx - ox) ** 2 + (dy - oy) ** 2 + (dz - oz) ** 2
        if dist < best_d:
            best_d = dist
            best_i = i
    return best_i


def decode_order_from_dot(
    base: tuple[float, float, float],
    dot: tuple[float, float, float],
    sorted_primes: tuple[int, ...],
    epsilon: float = PERM_EPSILON,
) -> tuple[int, ...]:
    k = len(sorted_primes)
    idx = nearest_permutation_index(base, dot, k, epsilon)
    return order_from_index(sorted_primes, idx)


def explain_order(sorted_primes: tuple[int, ...], order: tuple[int, ...]) -> str:
    k = len(sorted_primes)
    idx = permutation_index(order)
    return (
        f"prime order {order} is permutation {idx + 1}/{permutation_count(k)} "
        f"(sorted set {sorted_primes})"
    )


def demo() -> None:
    from aethos_lattice import BranchKind, LatticeId, apply_vector, lattice_id_parts
    from aethos_sequences import canon_on_chain

    print("=" * 60)
    print("PRIME PERMUTATION SIDE-CHANNEL (k! offsets)")
    print("=" * 60)

    for sorted_p in ((3, 11), (3, 5, 7)):
        k = len(sorted_p)
        print(f"\n  Sorted primes {sorted_p}: {permutation_count(k)} permutations")
        perms = ordered_permutation_list(sorted_p)
        branch, vector = lattice_id_parts(LatticeId.L01)
        base = apply_vector(canon_on_chain(branch, sorted_p, 5), vector)

        for order in perms:
            dot = apply_order_offset(base, sorted_p, order)
            recovered = decode_order_from_dot(base, dot, sorted_p)
            ox = (dot[0] - base[0], dot[1] - base[1], dot[2] - base[2])
            ok = recovered == order
            print(f"    order {order}  offset~({ox[0]:+.6f},{ox[1]:+.6f},{ox[2]:+.6f})  ok={ok}")


if __name__ == "__main__":
    demo()
