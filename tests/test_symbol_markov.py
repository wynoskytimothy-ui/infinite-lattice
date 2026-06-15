"""Markov correlation cascade — predict + strengthen on miss."""

from __future__ import annotations

import unittest

from aethos_symbol_knowledge import PRETRAIN_QUANTUM_GOLD, SymbolKnowledgeIndex
from aethos_symbol_markov import MarkovCorrelationBrain, build_markov_brain


class TestSymbolMarkov(unittest.TestCase):
    def test_predict_after_quantum(self) -> None:
        knowledge = SymbolKnowledgeIndex.build_from_corpus(
            PRETRAIN_QUANTUM_GOLD, dataset="markov_test",
        )
        brain = MarkovCorrelationBrain(knowledge=knowledge)
        brain.ingest_corpus()
        preds = brain.predict_next("quantum", top_k=5)
        words = {p.word for p in preds}
        self.assertTrue(words & {"zero", "dimension", "inductive", "biometrics"})

    def test_walk_strengthens_on_miss(self) -> None:
        knowledge = SymbolKnowledgeIndex.build_from_corpus(
            {"d1": "alpha beta gamma"},
            dataset="walk",
        )
        brain = MarkovCorrelationBrain(knowledge=knowledge)
        brain.ingest_corpus()
        before = len(knowledge.cross_links)
        brain.observe_step("alpha", "unexpected", strengthen_on_miss=True)
        self.assertGreater(brain.mismatch_strengthen, 0)
        self.assertTrue(knowledge.remembers("alpha", "unexpected"))

    def test_hit_on_known_bigram(self) -> None:
        corpus = {"d1": "cancer cell cancer cell treatment"}
        knowledge = SymbolKnowledgeIndex.build_from_corpus(corpus, dataset="bigram")
        brain = build_markov_brain(knowledge, attach_plane_index=False)
        r = brain.observe_step("cancer", "cell", top_k=3)
        self.assertTrue(r.hit)

    def test_accuracy_tracks_hits(self) -> None:
        knowledge = SymbolKnowledgeIndex.build_from_corpus(
            PRETRAIN_QUANTUM_GOLD, dataset="acc",
        )
        brain = MarkovCorrelationBrain(knowledge=knowledge)
        brain.ingest_corpus()
        brain.walk_text(PRETRAIN_QUANTUM_GOLD["gold_quantum_biometrics"])
        acc = brain.accuracy()
        self.assertGreater(acc["steps"], 0)
        self.assertGreater(acc["top5"], 0.0)


if __name__ == "__main__":
    unittest.main()
