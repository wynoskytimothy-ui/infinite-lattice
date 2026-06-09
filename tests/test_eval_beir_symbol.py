"""Smoke test for eval_beir_symbol helpers."""

from __future__ import annotations

import unittest

from aethos_symbol_knowledge import SymbolKnowledgeIndex
from eval_beir_symbol import evaluate_symbol_beir, mrr_at_k, query_words
from pipeline.bit_12_symbol_plane_index import build_symbol_plane_index


class TestEvalBeirSymbol(unittest.TestCase):
    def test_metrics_on_toy_corpus(self) -> None:
        corpus = {
            "gold": "cancer cell breast tumor treatment patients clinical",
            "d2": "protein gene expression rna virus",
            "d3": "sports football team game season",
        }
        knowledge = SymbolKnowledgeIndex.build_from_corpus(corpus, dataset="eval_toy")
        plane = build_symbol_plane_index(knowledge)
        queries = {"q1": "cancer breast treatment"}
        qrels = {"q1": {"gold": 1}}

        result = evaluate_symbol_beir(knowledge, plane, queries, qrels)
        self.assertEqual(result.n_queries, 1)
        self.assertGreaterEqual(result.ndcg_at_10, 0.0)
        self.assertGreater(result.route_recall, 0.0)

    def test_mrr(self) -> None:
        self.assertEqual(mrr_at_k(["a", "b", "gold"], {"gold": 1}, 10), 1 / 3)

    def test_query_words_filters_stopwords(self) -> None:
        words = query_words("What is the cancer cell treatment?")
        self.assertIn("cancer", words)
        self.assertNotIn("the", words)


if __name__ == "__main__":
    unittest.main()
