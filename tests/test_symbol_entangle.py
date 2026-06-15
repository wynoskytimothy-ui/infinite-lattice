"""Composite entanglement and opposite polarity scoring."""

from __future__ import annotations

import unittest

from aethos_symbol_entangle import (
    Polarity,
    are_opposites,
    build_entanglement_index,
    polarity_of,
    score_context,
)
from aethos_symbol_morph import build_morph_registry


class TestPolarity(unittest.TestCase):
    def test_raise_positive_diminis_negative(self) -> None:
        self.assertEqual(polarity_of("raise"), Polarity.POSITIVE)
        self.assertEqual(polarity_of("diminis"), Polarity.NEGATIVE)
        self.assertTrue(are_opposites("diminis", "raise"))
        self.assertTrue(are_opposites("diminished", "raise"))

    def test_lower_negative(self) -> None:
        self.assertEqual(polarity_of("lower"), Polarity.NEGATIVE)


class TestEntanglement(unittest.TestCase):
    def test_diminis_raise_entangle_opposite(self) -> None:
        morph = build_morph_registry({"diminis", "diminished", "diminishes", "raise", "lower"})
        from aethos_symbol_entangle import EntanglementRegistry

        ent = EntanglementRegistry(morph=morph)
        sc = score_context("diminis and raise together", morph, ent)
        self.assertIn("diminis", sc.pieces_found)
        self.assertIn("raise", sc.pieces_found)
        self.assertEqual(len(sc.entangled), 1)
        self.assertTrue(sc.entangled[0].opposite)
        self.assertGreater(sc.entanglement_bonus, 0)

    def test_raise_improves_diminish_lowers(self) -> None:
        morph = build_morph_registry({"raise", "diminished", "lower"})
        from aethos_symbol_entangle import EntanglementRegistry

        ent = EntanglementRegistry(morph=morph)
        sc_raise = score_context("raise improves scores", morph, ent)
        sc_dim = score_context("diminished lower scores", morph, ent)
        self.assertGreater(sc_raise.base_score, 0)
        self.assertLess(sc_dim.base_score, 0)

    def test_corpus_builds_pairs(self) -> None:
        corpus = {
            "a": "diminis raise together",
            "b": "only raise here",
        }
        reg = build_entanglement_index(corpus)
        key = tuple(sorted(("diminis", "raise")))
        self.assertIn(key, reg.pairs)


if __name__ == "__main__":
    unittest.main()
