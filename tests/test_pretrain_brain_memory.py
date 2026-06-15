"""Pretrain gold doc + compound learn memory."""

from __future__ import annotations

import unittest

from aethos_symbol_knowledge import PRETRAIN_QUANTUM_GOLD, SymbolKnowledgeIndex


class TestPretrainBrainMemory(unittest.TestCase):
    def test_gold_doc_teaches_quantum_dimension(self) -> None:
        sparse = {"d1": "protein gene expression study"}
        brain = SymbolKnowledgeIndex.build_from_corpus(sparse, dataset="sparse")
        self.assertFalse(brain.remembers("quantum", "dimension"))

        report = brain.compound_learn(PRETRAIN_QUANTUM_GOLD)
        self.assertGreater(report["links_added"], 0)
        self.assertTrue(brain.remembers("quantum", "dimension"))
        self.assertTrue(brain.remembers("zero", "dimension"))
        self.assertTrue(brain.remembers("inductive", "biometrics"))

    def test_query_gold_links(self) -> None:
        brain = SymbolKnowledgeIndex.build_from_corpus(
            PRETRAIN_QUANTUM_GOLD, dataset="gold_only",
        )
        chk = brain.query_gold_links(
            ["quantum", "zero", "dimension"],
            "gold_quantum_biometrics",
        )
        self.assertTrue(chk["all_pairs_linked"])
        self.assertTrue(chk["term_in_gold"]["quantum"])


if __name__ == "__main__":
    unittest.main()
