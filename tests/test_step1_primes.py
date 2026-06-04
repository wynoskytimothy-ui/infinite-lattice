"""Step 1 — prime engine tests."""

from __future__ import annotations

import unittest

from core.primes import (
    LETTER_PRIMES,
    PROMOTION_POOL,
    PrimePool,
    PoolTier,
    chain_primes,
    factorize_prime_product,
    product_unique,
    pool_bands,
)


class TestChainPrimes(unittest.TestCase):
    def test_first_five_odd_primes(self) -> None:
        self.assertEqual(chain_primes(5), (3, 5, 7, 11, 13))

    def test_no_two_in_chain(self) -> None:
        self.assertNotIn(2, chain_primes(100))

    def test_strictly_increasing(self) -> None:
        c = chain_primes(50)
        for i in range(1, len(c)):
            self.assertGreater(c[i], c[i - 1])


class TestLetterPrimes(unittest.TestCase):
    def test_letter_count(self) -> None:
        self.assertEqual(len(LETTER_PRIMES), 26)

    def test_a_is_three(self) -> None:
        self.assertEqual(LETTER_PRIMES[0], 3)


class TestPromotionPool(unittest.TestCase):
    def test_pool_after_letters(self) -> None:
        full = chain_primes(512)
        self.assertEqual(PROMOTION_POOL[0], full[26])

    def test_pool_bands_partition(self) -> None:
        bands = pool_bands(len(PROMOTION_POOL))
        total = sum(end - start for start, end in bands.values())
        self.assertEqual(total, len(PROMOTION_POOL))


class TestPrimePool(unittest.TestCase):
    def test_l2_alloc_deterministic(self) -> None:
        p1 = PrimePool()
        p2 = PrimePool()
        a = p1.alloc(PoolTier.L2_SUBWORD)
        b = p2.alloc(PoolTier.L2_SUBWORD)
        self.assertEqual(a, b)

    def test_l2_exhaust_raises(self) -> None:
        pool = PrimePool()
        cap = pool.l2_capacity()
        for _ in range(cap):
            pool.alloc(PoolTier.L2_SUBWORD)
        with self.assertRaises(RuntimeError):
            pool.alloc(PoolTier.L2_SUBWORD)

    def test_l3_band_separate_from_l2(self) -> None:
        pool = PrimePool()
        l2_first = pool.alloc(PoolTier.L2_SUBWORD)
        l3_first = pool.alloc(PoolTier.L3_WORD)
        self.assertNotEqual(l2_first, l3_first)


class TestFTA(unittest.TestCase):
    def test_product_factors(self) -> None:
        a, b = 37, 41
        n = product_unique(a, b)
        self.assertEqual(factorize_prime_product(n), (a, b))

    def test_distinct_pairs_unique(self) -> None:
        pairs = [(3, 5), (3, 7), (5, 7)]
        products = {product_unique(a, b) for a, b in pairs}
        self.assertEqual(len(products), len(pairs))


if __name__ == "__main__":
    unittest.main()
