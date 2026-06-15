"""Symbol knowledge index — cross-correlations, gaps, persistence."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aethos_symbol_knowledge import SymbolKnowledgeIndex


class TestSymbolKnowledge(unittest.TestCase):
    def test_root_morph_link_to_inflected(self) -> None:
        corpus = {
            "d1": "diminish the score was lower",
            "d2": "diminished scores over time",
        }
        idx = SymbolKnowledgeIndex.build_from_corpus(corpus, dataset="p3_root")
        link = idx.correlates("diminish", "diminished")
        self.assertIsNotNone(link, "root should morph-link to inflected form")
        assert link is not None
        self.assertEqual(link.kind, "morph")

    def test_direct_and_bridge(self) -> None:
        corpus = {
            "d1": "diminished score was lower",
            "d2": "diminishes over time in study",
        }
        idx = SymbolKnowledgeIndex.build_from_corpus(corpus, dataset="test")
        # direct: diminished + lower in d1
        self.assertIsNotNone(idx.correlates("diminished", "lower"))
        # morph: diminished + diminishes share root
        morph = idx.correlates("diminished", "diminishes")
        self.assertIsNotNone(morph)
        assert morph is not None
        self.assertEqual(morph.kind, "morph")
        # bridge: diminishes never touched lower but shares root with diminished
        bridge = idx.correlates("diminishes", "lower")
        self.assertIsNotNone(bridge)
        assert bridge is not None
        self.assertEqual(bridge.kind, "bridge")

    def test_gap_and_merge(self) -> None:
        sparse = {"d1": "quantum field theory"}
        rich = {"d2": "quantum zero dimension Hilbert space analysis"}
        idx = SymbolKnowledgeIndex.build_from_corpus(sparse, dataset="gaps")
        merged = idx.merge_corpus(rich)
        self.assertIsNotNone(merged.correlates("quantum", "zero"))
        self.assertIsNotNone(merged.correlates("zero", "dimension"))

    def test_save_load(self) -> None:
        corpus = {"d1": "hypothesis diminished rare pathway"}
        idx = SymbolKnowledgeIndex.build_from_corpus(corpus, dataset="persist")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "k.pkl"
            idx.save(path)
            loaded = SymbolKnowledgeIndex.load("persist", path=path)
            self.assertEqual(loaded.summary()["n_docs"], 1)
            self.assertGreater(loaded.summary()["total_cross_links"], 0)


if __name__ == "__main__":
    unittest.main()
