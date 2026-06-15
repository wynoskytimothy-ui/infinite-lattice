"""
Prime engine for the AETHOS core foundation (Step 1).

Odd primes only (2 skipped). L1 uses the first 26; promotion pool follows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


@lru_cache(maxsize=None)
def chain_primes(count: int) -> tuple[int, ...]:
    """Odd primes only — 3, 5, 7, 11, ... (sieve of Eratosthenes, cached).

    Identical output to trial division, but O(n log log n) not O(n*sqrt n): the
    200k-prime pool drops from ~4.8s to ~50ms, and the cache means it is built
    once and shared across all index instances (incl. shards)."""
    if count <= 0:
        return ()
    n = count + 1                                   # 2 is skipped; need count+1 incl. 2
    bound = 15 if n < 6 else int(n * (math.log(n) + math.log(math.log(n)))) + 10
    sieve = bytearray([1]) * (bound + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(bound ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytes(len(range(i * i, bound + 1, i)))
    primes = [i for i in range(3, bound + 1, 2) if sieve[i]]   # odd primes only
    while len(primes) < count:                      # safety if the bound underestimates
        bound *= 2
        sieve = bytearray([1]) * (bound + 1)
        sieve[0:2] = b"\x00\x00"
        for i in range(2, int(bound ** 0.5) + 1):
            if sieve[i]:
                sieve[i * i::i] = bytes(len(range(i * i, bound + 1, i)))
        primes = [i for i in range(3, bound + 1, 2) if sieve[i]]
    return tuple(primes[:count])


# L1: a..z → first 26 odd primes (aligned with aethos_words.LETTER_PRIMES)
LETTER_PRIMES: tuple[int, ...] = chain_primes(26)

# Promotion pool: primes after letter band (index 26+)
PROMOTION_POOL_SIZE = 512
PROMOTION_POOL: tuple[int, ...] = chain_primes(PROMOTION_POOL_SIZE)[26:]


class PoolTier(str, Enum):
    L2_SUBWORD = "L2"
    L3_WORD = "L3"
    SPECIES = "SPECIES"


def pool_bands(pool_len: int) -> dict[PoolTier, tuple[int, int]]:
    l2_end = max(1, int(pool_len * 0.41))
    l3_end = max(l2_end + 1, int(pool_len * 0.82))
    return {
        PoolTier.L2_SUBWORD: (0, l2_end),
        PoolTier.L3_WORD: (l2_end, l3_end),
        PoolTier.SPECIES: (l3_end, pool_len),
    }


@dataclass
class PrimePool:
    """Tiered allocator over PROMOTION_POOL."""

    pool: tuple[int, ...] = PROMOTION_POOL
    cursors: dict[PoolTier, int] = field(default_factory=dict)
    _bands: dict[PoolTier, tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._bands:
            self._bands = pool_bands(len(self.pool))
        if not self.cursors:
            self.cursors = {t: 0 for t in PoolTier}

    def alloc(self, tier: PoolTier) -> int:
        start, end = self._bands[tier]
        idx = self.cursors[tier]
        if start + idx >= end:
            raise RuntimeError(f"promotion pool exhausted in tier {tier.value}")
        prime = self.pool[start + idx]
        self.cursors[tier] = idx + 1
        return prime

    def l2_capacity(self) -> int:
        start, end = self._bands[PoolTier.L2_SUBWORD]
        return end - start

    def l2_used(self) -> int:
        return self.cursors[PoolTier.L2_SUBWORD]

    def l2_remaining(self) -> int:
        return self.l2_capacity() - self.l2_used()


def product_unique(a: int, b: int) -> int:
    """FTA: product of distinct primes is unique."""
    return a * b


def factorize_prime_product(n: int) -> tuple[int, ...]:
    """Return sorted prime factors of n (for tests)."""
    if n < 2:
        return ()
    factors: list[int] = []
    x = 2
    while x * x <= n:
        while n % x == 0:
            factors.append(x)
            n //= x
        x += 1 if x == 2 else 2
    if n > 1:
        factors.append(n)
    return tuple(factors)
