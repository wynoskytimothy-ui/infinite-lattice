"""Cascade retrieval — rare 2-way + 3-way bridge (zero-shot formula)."""

from __future__ import annotations

import unittest

from aethos_cascade_retrieval import search_docs_cascade, triple_meet_keys
from aethos_symbol_knowledge import SymbolKnowledgeIndex
from pipeline.bit_12_symbol_plane_index import build_symbol_plane_index


class TestCascadeRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = {
            "gold": (
                "zephyrox metaplasm fibrosis bleomycin lung zephyroxic pathway"
            ),
            "noise": (
                "protein kinase study cells treatment cancer patients clinical trial"
            ),
        }
        cls.knowledge = SymbolKnowledgeIndex.build_from_corpus(
            cls.corpus, dataset="cascade_toy",
        )
        cls.plane = build_symbol_plane_index(cls.knowledge)

    def test_triple_meet_keys_nonempty(self) -> None:
        keys = triple_meet_keys(
            self.knowledge, "zephyrox", "fibrosis", "bleomycin",
            quantize=self.plane.quantize,
        )
        self.assertGreater(len(keys), 0)

    def test_cascade_ranks_gold_first(self) -> None:
        route, ranked = search_docs_cascade(
            self.knowledge,
            self.plane,
            ["zephyrox", "fibrosis", "bleomycin"],
            max_candidates=10,
            limit=5,
        )
        self.assertEqual(route.tier, "cascade")
        self.assertGreater(len(ranked), 0)
        self.assertEqual(ranked[0][0], "gold")


if __name__ == "__main__":
    unittest.main()
