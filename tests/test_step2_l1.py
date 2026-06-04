"""Step 2 — L1 character layer tests."""

from __future__ import annotations

import unittest

from aethos_words import letter_to_prime as prod_letter_to_prime
from aethos_words import LETTER_PRIMES as PROD_LETTER_PRIMES

from core.l1_characters import (
    char_prime,
    intersection_prime,
    prime_to_char,
    word_letter_chain,
    word_letter_order,
)
from core.primes import LETTER_PRIMES


class TestCharPrime(unittest.TestCase):
    def test_matches_production_letter_primes(self) -> None:
        self.assertEqual(LETTER_PRIMES, PROD_LETTER_PRIMES)

    def test_a_through_z(self) -> None:
        for ch in "abcdefghijklmnopqrstuvwxyz":
            self.assertEqual(char_prime(ch), prod_letter_to_prime(ch))

    def test_case_insensitive(self) -> None:
        self.assertEqual(char_prime("A"), char_prime("a"))

    def test_roundtrip(self) -> None:
        for ch in "aeiou":
            p = char_prime(ch)
            self.assertEqual(prime_to_char(p), ch)


class TestIntersection(unittest.TestCase):
    def test_cat_sum_not_product(self) -> None:
        w = "cat"
        self.assertEqual(intersection_prime(w), sum(word_letter_order(w)))

    def test_intersection_equals_prod(self) -> None:
        from aethos_promotion import intersection_prime as prod_intersection

        for w in ("apple", "phone", "quantum"):
            self.assertEqual(intersection_prime(w), prod_intersection(w))


class TestWordChains(unittest.TestCase):
    def test_order_left_to_right(self) -> None:
        order = word_letter_order("tab")
        self.assertEqual(len(order), 3)

    def test_chain_sorted_unique(self) -> None:
        chain = word_letter_chain("apple")
        self.assertEqual(chain, tuple(sorted(chain)))


if __name__ == "__main__":
    unittest.main()
