"""Geometry-native fast ingest + train."""

from __future__ import annotations

import unittest

from aethos_geometry_ingest import (
    FastTrinaryTrainer,
    ingest_doc_fast,
    patch_plane_for_words,
    train_query_fast,
)
from aethos_symbol_knowledge import PRETRAIN_QUANTUM_GOLD, SymbolKnowledgeIndex
from eval_beir_symbol import query_words
from pipeline.bit_12_symbol_plane_index import build_symbol_plane_index


class TestGeometryIngest(unittest.TestCase):
    def test_ingest_doc_fast_adds_kappa_keys(self) -> None:
        knowledge = SymbolKnowledgeIndex.build_from_corpus(
            {"d0": "quantum field"},
            dataset="geo_ingest",
        )
        plane = build_symbol_plane_index(knowledge)
        before = len(plane.doc_keys.get("d0", set()))

        report = ingest_doc_fast(
            plane, knowledge, "d1", "apple phone chip software",
            update_knowledge=True,
        )
        self.assertGreater(report["keys_added"], 0)
        self.assertIn("d1", plane.doc_keys)
        self.assertGreater(len(plane.doc_keys["d1"]), before)

    def test_train_query_fast_patches_plane(self) -> None:
        knowledge = SymbolKnowledgeIndex.build_from_corpus(
            PRETRAIN_QUANTUM_GOLD, dataset="geo_train",
        )
        plane = build_symbol_plane_index(knowledge)
        pairs_before = len(plane.pair_keys)

        out = train_query_fast(
            knowledge,
            plane,
            "q1",
            "quantum zero dimension",
            ["gold_quantum_biometrics"],
        )
        self.assertGreater(out["triples_promoted"], 0)
        self.assertGreaterEqual(out["train_ms"], 0.0)
        touch = set(query_words("quantum zero dimension"))
        patch = patch_plane_for_words(knowledge, plane, touch)
        self.assertIn("pair_meets_added", patch)
        self.assertGreaterEqual(len(plane.pair_keys), pairs_before)

    def test_fast_trainer_cached_doc_freq(self) -> None:
        knowledge = SymbolKnowledgeIndex.build_from_corpus(
            {"d": "apple phone chip"}, dataset="geo_freq",
        )
        trainer = FastTrinaryTrainer(knowledge)
        trainer.warm_doc_freq("apple phone")
        self.assertGreater(trainer._word_doc_freq("apple"), 0)
        self.assertEqual(trainer._word_doc_freq("missing"), 0)


if __name__ == "__main__":
    unittest.main()
