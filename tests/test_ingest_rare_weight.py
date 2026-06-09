"""Ingest-time rare link weighting — rare pairs dominate hub pairs."""

from __future__ import annotations

import unittest

from aethos_rare_rank import (
    INGEST_BOTH_RARE_FACTOR,
    is_rare_word,
)
from aethos_symbol_knowledge import SymbolKnowledgeIndex


class TestIngestRareWeight(unittest.TestCase):
    def _toy_corpus(self) -> dict[str, str]:
        # apple in 3 docs (hub via doc fraction); apple+phone co-occur once.
        # zephyrox+fibrosis co-occur once in the rare doc.
        return {
            "d_hub1": "apple phone chip software silicon network",
            "d_hub2": "apple pie fruit orchard cinnamon baking",
            "d_hub3": "apple computer laptop tablet device screen",
            "d_rare": "zephyrox metaplasm fibrosis pathway bleomycin repair",
        }

    def test_rare_pair_stronger_than_hub_pair(self) -> None:
        idx = SymbolKnowledgeIndex.build_from_corpus(
            self._toy_corpus(), dataset="ingest_rare_toy",
        )
        rare_link = idx.correlates("zephyrox", "fibrosis")
        hub_link = idx.correlates("apple", "phone")
        self.assertIsNotNone(rare_link)
        self.assertIsNotNone(hub_link)
        self.assertTrue(is_rare_word(idx, "zephyrox", ingest_safe=True))
        self.assertTrue(is_rare_word(idx, "fibrosis", ingest_safe=True))
        self.assertFalse(is_rare_word(idx, "apple", ingest_safe=True))
        # Equal raw co-occurrence (1); rare pair gets ingest boost only.
        self.assertEqual(hub_link.strength, 1.0)
        self.assertGreater(rare_link.strength, hub_link.strength)
        self.assertAlmostEqual(
            rare_link.strength,
            hub_link.strength * INGEST_BOTH_RARE_FACTOR,
        )

    def test_rare_weight_can_disable(self) -> None:
        corpus = self._toy_corpus()
        idx = SymbolKnowledgeIndex.build_from_corpus(
            corpus, dataset="ingest_rare_on",
        )
        boosted = idx.correlates("zephyrox", "fibrosis")
        self.assertIsNotNone(boosted)
        self.assertAlmostEqual(boosted.strength, INGEST_BOTH_RARE_FACTOR)

        idx.ingest_rare_weight = False
        idx.chamber_links.clear()
        from aethos_symbol_subjects import MASTER_CHAMBER
        idx._build_cross_links(
            MASTER_CHAMBER,
            cooccur=idx._chamber_cooccur[MASTER_CHAMBER],
            rare_boost=False,
        )
        plain = idx.correlates("zephyrox", "fibrosis")
        self.assertIsNotNone(plain)
        self.assertEqual(plain.strength, 1.0)


if __name__ == "__main__":
    unittest.main()
