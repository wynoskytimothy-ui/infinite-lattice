"""Cellular membrane filler vs rare-rare signal correlations."""

from __future__ import annotations

import unittest

from aethos_symbol_cellular import (
    CellularContextScore,
    CellularRole,
    build_cellular_entanglement,
    build_cellular_profiles,
    score_context_cellular,
)
from aethos_symbol_entangle import EntanglementRegistry
from aethos_symbol_morph import build_morph_registry


class TestMembraneBlocksCorrelation(unittest.TestCase):
    def test_filler_does_not_entangle_with_rare(self) -> None:
        corpus = {
            "d1": "the diminished score",
            "d2": "the the the everywhere",
        }
        morph = build_morph_registry({"the", "diminished", "score", "lower"})
        cell = build_cellular_profiles(corpus, morph)
        self.assertEqual(cell.role_of("the"), CellularRole.MEMBRANE)

        ent = EntanglementRegistry(morph=morph)
        sc = score_context_cellular(
            "the diminished score", morph, ent, cell,
        )
        self.assertIsInstance(sc, CellularContextScore)
        assert isinstance(sc, CellularContextScore)
        # the is membrane — should not correlate with diminished
        rare_pairs = {(ep.left, ep.right) for ep in sc.rare_correlations}
        self.assertNotIn(("diminished", "the"), rare_pairs)
        self.assertNotIn(("the", "diminished"), rare_pairs)
        self.assertGreater(sc.blocked_correlations, 0)

    def test_rare_rare_entangles(self) -> None:
        corpus = {
            "d1": "diminis raise together",
            "d2": "hypothesis diminished rare",
        }
        ent, cell = build_cellular_entanglement(corpus)
        sc = ent.doc_scores["d1"]
        self.assertIsInstance(sc, CellularContextScore)
        assert isinstance(sc, CellularContextScore)
        self.assertGreaterEqual(len(sc.rare_correlations), 1)
        key = tuple(sorted(("diminis", "raise")))
        self.assertIn(key, cell.rare_pairs)


if __name__ == "__main__":
    unittest.main()
