"""Imaginary line synthesis — sum of meeting primes."""

from __future__ import annotations

import unittest

from aethos_symbol_map import symbol_to_prime
from aethos_symbol_synthesis import (
    build_vocabulary,
    imaginary_sum,
    meet_2way_on_line,
    meet_3way_on_line,
    needed_meetings,
    synthesize_len2,
    SynthesisRegistry,
)


class TestImaginarySum(unittest.TestCase):
    def test_three_meets_five_is_eight(self) -> None:
        self.assertEqual(symbol_to_prime("a"), 3)
        self.assertEqual(symbol_to_prime("b"), 5)
        self.assertEqual(meet_2way_on_line(3, 5), 8)

    def test_three_way_sum(self) -> None:
        primes = tuple(symbol_to_prime(c) for c in "cat")
        self.assertEqual(meet_3way_on_line(*primes), imaginary_sum(*primes))


class TestLen2Synthesis(unittest.TestCase):
    def test_ab_promoted_at_eight(self) -> None:
        reg = SynthesisRegistry()
        comp = synthesize_len2("ab", reg)
        self.assertIsNotNone(comp)
        assert comp is not None
        self.assertEqual(comp.imaginary_position, 8)
        self.assertEqual(comp.meet_arity, 2)
        self.assertEqual(comp.meeting_primes, (3, 5))

    def test_ba_same_imag_different_word_collision(self) -> None:
        reg = SynthesisRegistry()
        synthesize_len2("ab", reg)
        comp_ba = synthesize_len2("ba", reg)
        # same multiset → same imag 8; second word blocked unless unique
        self.assertIsNone(comp_ba)


class TestVocabularyBuild(unittest.TestCase):
    def test_three_letter_three_way(self) -> None:
        reg = build_vocabulary({"cat", "the"})
        cat = needed_meetings("cat", reg)
        self.assertIsNotNone(cat)
        assert cat is not None
        self.assertEqual(cat["meet_arity"], 3)
        self.assertEqual(len(cat["meeting_primes"]), 3)

    def test_longer_word_two_part_split(self) -> None:
        reg = build_vocabulary({"at", "cat", "cats"})
        info = needed_meetings("cats", reg)
        self.assertIsNotNone(info)
        assert info is not None
        self.assertEqual(info["meet_arity"], 2)
        self.assertEqual(info["parts"], ("cat", "s"))


if __name__ == "__main__":
    unittest.main()
