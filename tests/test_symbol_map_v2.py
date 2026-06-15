"""L1 English symbol map v2 — full charset, ICN subwords, imaginary line."""

from __future__ import annotations

import unittest

from aethos_complex_plane import imaginary_start
from aethos_species import digit_to_prime
from aethos_symbol_map import (
    ENGLISH_SYMBOLS,
    PUNCT_PRIMES,
    SymbolKind,
    normalize_symbol,
    prime_to_symbol,
    subword_on_imaginary_line,
    symbol_kind,
    symbol_on_imaginary_line,
    symbol_table,
    symbol_to_prime,
    text_icn,
    text_icn_chain,
    text_symbol_chain,
)
from aethos_words import letter_to_prime


class TestSymbolBands(unittest.TestCase):
    def test_alpha_matches_letter_primes(self) -> None:
        for ch in "abcdefghijklmnopqrstuvwxyz":
            self.assertEqual(symbol_to_prime(ch), letter_to_prime(ch))
            self.assertEqual(symbol_kind(ch), SymbolKind.ALPHA)

    def test_digit_matches_species(self) -> None:
        for d in "0123456789":
            self.assertEqual(symbol_to_prime(d), digit_to_prime(d))
            self.assertEqual(symbol_kind(d), SymbolKind.DIGIT)

    def test_punct_disjoint_from_alpha_and_promotion_pool(self) -> None:
        from core.primes import PROMOTION_POOL

        alpha = {symbol_to_prime(c) for c in "abcdefghijklmnopqrstuvwxyz"}
        digit = {symbol_to_prime(d) for d in "0123456789"}
        punct = set(PUNCT_PRIMES)
        self.assertTrue(alpha.isdisjoint(punct))
        self.assertTrue(digit.isdisjoint(punct))
        self.assertTrue(set(PROMOTION_POOL).isdisjoint(punct))

    def test_all_primes_unique(self) -> None:
        primes = [symbol_to_prime(s) for s in ENGLISH_SYMBOLS]
        self.assertEqual(len(primes), len(set(primes)))

    def test_roundtrip(self) -> None:
        for entry in symbol_table():
            self.assertEqual(prime_to_symbol(entry.prime), entry.symbol)


class TestNormalization(unittest.TestCase):
    def test_uppercase_folds(self) -> None:
        self.assertEqual(symbol_to_prime("A"), symbol_to_prime("a"))

    def test_curly_quote_folds(self) -> None:
        self.assertEqual(symbol_to_prime("\u2019"), symbol_to_prime("'"))


class TestSubwordICN(unittest.TestCase):
    def test_th_product_unique(self) -> None:
        icn = text_icn("th")
        self.assertEqual(icn, symbol_to_prime("t") * symbol_to_prime("h"))
        self.assertEqual(text_icn_chain("th"), tuple(sorted((symbol_to_prime("t"), symbol_to_prime("h")))))

    def test_order_preserved_in_chain_not_icn(self) -> None:
        self.assertNotEqual(text_symbol_chain("tab"), text_symbol_chain("bat"))
        self.assertEqual(text_icn("tab"), text_icn("bat"))  # same letter multiset

    def test_anagram_same_icn_different_order_chain(self) -> None:
        self.assertEqual(text_icn("listen"), text_icn("silent"))
        self.assertNotEqual(text_symbol_chain("listen"), text_symbol_chain("silent"))


class TestImaginaryLine(unittest.TestCase):
    def test_solo_anchor_differs_from_layer0(self) -> None:
        psi_t = symbol_on_imaginary_line("t", n=7)
        psi0 = imaginary_start(7)
        self.assertNotEqual(psi_t.z, psi0.z)

    def test_subword_chain_places_on_plane(self) -> None:
        psi = subword_on_imaginary_line("th", n=7)
        self.assertIsNotNone(psi.z)
        self.assertIsNotNone(psi.zeta)


if __name__ == "__main__":
    unittest.main()
