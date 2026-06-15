"""Standalone ordered-subword promotion — path-sensitive L2 nodes."""

from __future__ import annotations

import unittest

from aethos_symbol_map import text_icn, text_symbol_chain
from aethos_symbol_promotion import (
    SymbolPromotionRegistry,
    all_path_permutations,
    path_collision_report,
    trigram_path_report,
)


class TestOrderedPromotion(unittest.TestCase):
    def test_ca_ac_same_l1_icn_different_order(self) -> None:
        self.assertEqual(text_icn("ca"), text_icn("ac"))
        self.assertNotEqual(text_symbol_chain("ca"), text_symbol_chain("ac"))

    def test_promotion_separates_anagrams(self) -> None:
        r = path_collision_report("ca", "ac")
        self.assertTrue(r["l1_icn_same"])
        self.assertTrue(r["l1_psi_same"])
        self.assertNotEqual(r["l2_prime_a"], r["l2_prime_b"])
        self.assertFalse(r["l2_psi_same"])

    def test_same_subword_reuses_prime(self) -> None:
        reg = SymbolPromotionRegistry()
        t1 = reg.promote("th")
        t2 = reg.promote("th")
        self.assertIs(t1, t2)
        self.assertEqual(reg._cursor, 1)

    def test_different_order_different_primes(self) -> None:
        reg = SymbolPromotionRegistry()
        p_ca = reg.promote("ca").prime
        p_ac = reg.promote("ac").prime
        self.assertNotEqual(p_ca, p_ac)


class TestTrigramPaths(unittest.TestCase):
    def test_the_has_six_permutations(self) -> None:
        self.assertEqual(len(all_path_permutations("the")), 6)

    def test_see_has_three_permutations(self) -> None:
        self.assertEqual(len(all_path_permutations("see")), 3)

    def test_trigram_siblings_all_promoted(self) -> None:
        reg = SymbolPromotionRegistry()
        reg.promote_with_siblings("the", frequency=5)
        self.assertEqual(len(reg.by_length(3)), 6)
        self.assertEqual(reg.promoted["the"].frequency, 5)
        self.assertEqual(reg.promoted["het"].frequency, 0)

    def test_trigram_l1_collapses_l2_separates(self) -> None:
        r = trigram_path_report("the")
        self.assertEqual(r["n_paths"], 6)
        self.assertEqual(r["l1_unique_coords"], 1)
        self.assertEqual(r["l2_unique_primes"], 6)


if __name__ == "__main__":
    unittest.main()
