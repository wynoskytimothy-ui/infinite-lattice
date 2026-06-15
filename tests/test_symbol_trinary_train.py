"""Query → gold rare 3-way correlation training."""

from __future__ import annotations

import unittest

from aethos_symbol_knowledge import PRETRAIN_QUANTUM_GOLD, SymbolKnowledgeIndex
from aethos_symbol_trinary_train import TrinaryTrainer


class TestTrinaryTrain(unittest.TestCase):
    def test_promote_rare_triple_from_gold(self) -> None:
        knowledge = SymbolKnowledgeIndex.build_from_corpus(
            PRETRAIN_QUANTUM_GOLD, dataset="tri",
        )
        trainer = TrinaryTrainer(knowledge=knowledge)
        report = trainer.train_query(
            "q1",
            "quantum zero dimension",
            ["gold_quantum_biometrics"],
        )
        self.assertGreater(report.triples_promoted, 0)
        a, b, c = report.promoted[0].words
        self.assertTrue(knowledge.remembers(a, b))
        self.assertTrue(knowledge.remembers(a, c))
        self.assertTrue(knowledge.remembers(b, c))

    def test_multiple_golds_picks_rarest(self) -> None:
        corpus = {
            "gold_rare": "quantum zero dimension exotic pathway",
            "gold_common": "protein cell cancer treatment patients study",
        }
        knowledge = SymbolKnowledgeIndex.build_from_corpus(corpus, dataset="multi")
        trainer = TrinaryTrainer(knowledge=knowledge)
        rarest, triples = trainer.find_rare_triples(
            "quantum dimension",
            ["gold_rare", "gold_common"],
        )
        self.assertEqual(rarest, "gold_rare")
        self.assertGreater(len(triples), 0)

    def test_predict_triple_completion(self) -> None:
        knowledge = SymbolKnowledgeIndex.build_from_corpus(
            PRETRAIN_QUANTUM_GOLD, dataset="pred",
        )
        trainer = TrinaryTrainer(knowledge=knowledge)
        trainer.train_query(
            "q1", "quantum zero dimension", ["gold_quantum_biometrics"],
        )
        pred = trainer.predict_triple_completion(
            "quantum zero dimension", ["quantum", "zero"],
        )
        self.assertGreater(len(pred), 0)


class TestAppleDisambiguation(unittest.TestCase):
    """apple+phone (technical) vs apple+pie (fruit) — 3-ways stay separate."""

    def test_stacked_corpus_disambiguates(self) -> None:
        tech = {"gold_tech": "apple phone chip software silicon processor technical"}
        fruit = {"gold_fruit": "apple pie fruit orchard harvest baking cinnamon"}
        knowledge = SymbolKnowledgeIndex.build_from_corpus(tech, dataset="apple")
        knowledge.stack_corpus(fruit, name="fruit")

        trainer = TrinaryTrainer(knowledge=knowledge)
        trainer.train_query("q_tech", "apple phone chip", ["gold_tech"])
        trainer.train_query("q_fruit", "apple pie fruit", ["gold_fruit"])

        tech_pred = trainer.predict_triple_completion(
            "apple phone chip", ["apple", "phone"],
        )
        fruit_pred = trainer.predict_triple_completion(
            "apple pie fruit", ["apple", "pie"],
        )
        tech_words = {w for w, _ in tech_pred}
        fruit_words = {w for w, _ in fruit_pred}

        self.assertTrue(tech_words & {"chip", "software", "silicon", "processor"})
        self.assertTrue(fruit_words & {"fruit", "orchard", "cinnamon", "harvest"})
        self.assertFalse(tech_words & {"pie", "cinnamon"})
        self.assertFalse(fruit_words & {"chip", "silicon"})

        # stacking preserved both corpora
        self.assertIn("gold_tech", knowledge.corpus)
        self.assertIn("gold_fruit", knowledge.corpus)
        self.assertTrue(knowledge.remembers("apple", "phone"))
        self.assertTrue(knowledge.remembers("apple", "fruit"))


if __name__ == "__main__":
    unittest.main()
