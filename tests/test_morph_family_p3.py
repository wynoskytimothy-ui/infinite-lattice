"""Concrete Plane P3 — morph family links on SciFact brain."""

from __future__ import annotations

import unittest

from aethos_symbol_knowledge import SymbolKnowledgeIndex


class TestMorphFamilyP3(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.knowledge = SymbolKnowledgeIndex.load("scifact")

    def test_cell_family_has_morph_or_bridge(self) -> None:
        morph = self.knowledge.correlates("cell", "cellular")
        bridge = self.knowledge.correlates("cells", "cellular")
        self.assertTrue(
            morph is not None or bridge is not None,
            "cell family should be linked after P3",
        )

    def test_morph_links_refreshed_on_load(self) -> None:
        morph_n = sum(
            1 for lk in self.knowledge.cross_links.values() if lk.kind == "morph"
        )
        self.assertGreater(morph_n, 500, f"morph links={morph_n}")


if __name__ == "__main__":
    unittest.main()
