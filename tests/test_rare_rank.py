"""Rare-weighted correlation ranking — toy corpus discrimination."""

from __future__ import annotations

import unittest

from aethos_rare_rank import (
    is_rare_word,
    rank_docs_rare_weighted,
    rare_neighbors,
    rare_query_triggers,
    score_doc_rare_correlations,
    search_docs_rare_correlations,
)
from aethos_symbol_knowledge import SymbolKnowledgeIndex
from eval_beir_symbol import query_words
from pipeline.bit_12_symbol_plane_index import (
    build_symbol_plane_index,
    route_symbol_plane_candidates,
)


class TestRareRank(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = {
            "gold": (
                "zephyrox metaplasm quanta fibrosis pathway "
                "bleomycin lung repair zephyroxic"
            ),
            "noise_a": (
                "protein kinase study cells treatment cancer patients "
                "clinical trial cohort analysis"
            ),
            "noise_b": (
                "genome mapping variants genetic sequence project "
                "population penetrance common rare"
            ),
        }
        self.knowledge = SymbolKnowledgeIndex.build_from_corpus(
            self.corpus, dataset="rare_rank_toy",
        )
        self.plane = build_symbol_plane_index(self.knowledge)
        self.query = "zephyrox fibrosis bleomycin"
        self.words = query_words(self.query)

    def test_rare_words_detected(self) -> None:
        self.assertTrue(is_rare_word(self.knowledge, "zephyrox"))
        self.assertTrue(is_rare_word(self.knowledge, "fibrosis"))
        self.assertFalse(is_rare_word(self.knowledge, "the"))

    def test_rare_neighbors_filtered(self) -> None:
        nbrs = rare_neighbors(self.knowledge, "zephyrox", limit=8)
        self.assertGreater(len(nbrs), 0)
        for word, strength in nbrs:
            self.assertTrue(is_rare_word(self.knowledge, word))
            self.assertGreater(strength, 0)

    def test_gold_scores_higher_rare_correlations(self) -> None:
        gold_score = score_doc_rare_correlations(
            self.knowledge, self.words, "gold", self.corpus["gold"],
        )
        noise_score = score_doc_rare_correlations(
            self.knowledge, self.words, "noise_a", self.corpus["noise_a"],
        )
        self.assertGreater(gold_score, noise_score)

    def test_rare_ranker_puts_gold_first(self) -> None:
        route = route_symbol_plane_candidates(
            self.knowledge, self.plane, self.words, max_candidates=50,
        )
        ranked = rank_docs_rare_weighted(
            self.knowledge,
            self.plane,
            self.words,
            route.doc_ids or list(self.corpus),
            self.corpus,
        )
        self.assertGreater(len(ranked), 0)
        self.assertEqual(ranked[0][0], "gold")

    def test_rare_query_triggers_rare_words(self) -> None:
        triggers = rare_query_triggers(self.knowledge, self.words)
        self.assertIn("zephyrox", triggers)
        self.assertIn("fibrosis", triggers)

    def test_search_docs_rare_correlations_primary_path(self) -> None:
        route, ranked = search_docs_rare_correlations(
            self.knowledge, self.plane, self.words, max_candidates=50, limit=10,
        )
        self.assertEqual(route.tier, "rare_correlation")
        self.assertGreater(len(ranked), 0)
        self.assertEqual(ranked[0][0], "gold")


if __name__ == "__main__":
    unittest.main()
