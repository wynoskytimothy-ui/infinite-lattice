"""Morph synthesis — unique subwords, ed/es, shared correlation."""

from __future__ import annotations

import unittest

from aethos_symbol_morph import (
    build_morph_registry,
    explain_diminish_family,
    is_rare_word,
    meet_2way_on_line,
)


class TestMorphFamily(unittest.TestCase):
    def test_ed_es_different_primes(self) -> None:
        r = explain_diminish_family()
        self.assertTrue(r["different_suffix_prime"])

    def test_shared_root_correlation(self) -> None:
        r = explain_diminish_family()
        self.assertTrue(r["same_root_correlation"])

    def test_diminished_rare(self) -> None:
        vocab = {"diminished", "diminishes", "diminish", "red"}
        self.assertTrue(is_rare_word("diminished", vocab))
        self.assertFalse(is_rare_word("diminish", vocab))

    def test_diminished_meets_ed_not_es(self) -> None:
        reg = build_morph_registry({"diminished", "diminishes"})
        d = reg.composites["diminished"]
        s = reg.composites["diminishes"]
        self.assertEqual(d.suffix, "ed")
        self.assertEqual(s.suffix, "es")
        self.assertNotEqual(d.meeting_primes[1], s.meeting_primes[1])

    def test_three_five_eight(self) -> None:
        self.assertEqual(meet_2way_on_line(3, 5), 8)


if __name__ == "__main__":
    unittest.main()
