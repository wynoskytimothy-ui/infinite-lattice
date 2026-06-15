"""Fast ingest — same correlations, less work on ingest."""

from __future__ import annotations

import unittest

from aethos_symbol_knowledge import SymbolKnowledgeIndex
from aethos_symbol_subjects import MASTER_CHAMBER


class TestIngestFast(unittest.TestCase):
    def test_lazy_builds_on_read(self) -> None:
        idx = SymbolKnowledgeIndex.build_from_corpus(
            {
                "d1": "apple phone chip software silicon",
                "d2": "apple pie fruit orchard cinnamon",
            },
            subjects=frozenset(),
            doc_subjects={"d1": {4}, "d2": {11}},
        )
        self.assertIn(4, idx._chamber_dirty)
        self.assertIsNotNone(idx.correlates("apple", "phone", chamber=4))
        self.assertIsNotNone(idx.correlates("apple", "pie", chamber=11))
        self.assertIsNone(idx.correlates("apple", "phone", chamber=11))
        self.assertNotIn(4, idx._chamber_dirty)

    def test_incremental_compound_learn(self) -> None:
        brain = SymbolKnowledgeIndex.build_from_corpus(
            {"d0": "quantum field"},
            subjects={1},
        )
        report = brain.compound_learn(
            {"d1": "quantum zero dimension Hilbert"},
            subjects={1},
        )
        self.assertGreater(report["links_after"], report["links_before"])
        self.assertIn("ingest_ms", report)
        self.assertIsNotNone(brain.correlates("quantum", "zero"))

    def test_doc_evidence_stored_once(self) -> None:
        idx = SymbolKnowledgeIndex.build_from_corpus(
            {"d": "apple phone chip"},
            subjects={4, 11, 18},
        )
        self.assertEqual(len(idx._doc_evidence), 1)
        ev = idx._doc_evidence["d"]
        self.assertIn(4, ev.chambers)


if __name__ == "__main__":
    unittest.main()
