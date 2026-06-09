"""31 subject chambers + master subconscious — ingest, audit, query routing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aethos_symbol_knowledge import SymbolKnowledgeIndex
from aethos_symbol_subjects import (
    MASTER_CHAMBER,
    chamber_to_branch_wing,
    master_audit,
    subjects_for_dataset,
    vote_query_chambers,
    write_master_audit,
)


class TestSubjectChambers(unittest.TestCase):
    def test_dataset_tags(self) -> None:
        self.assertIn(10, subjects_for_dataset("scifact"))
        self.assertIn(18, subjects_for_dataset("fiqa"))

    def test_multi_subject_ingest(self) -> None:
        tech = {"d_tech": "apple phone chip software silicon processor"}
        food = {"d_food": "apple pie fruit orchard cinnamon harvest"}
        idx = SymbolKnowledgeIndex.build_from_corpus(
            tech,
            dataset="tech",
            subjects={4},
        )
        idx.ingest_corpus(food, {11})

        self.assertIsNotNone(idx.correlates("apple", "phone", chamber=4))
        self.assertIsNotNone(idx.correlates("apple", "pie", chamber=11))
        self.assertIsNotNone(idx.correlates("apple", "phone", chamber=MASTER_CHAMBER))
        self.assertIsNotNone(idx.correlates("apple", "pie", chamber=MASTER_CHAMBER))
        self.assertIsNone(idx.correlates("apple", "phone", chamber=11))
        self.assertIsNone(idx.correlates("apple", "pie", chamber=4))

    def test_query_votes_tech_chamber(self) -> None:
        voted = vote_query_chambers(["apple", "phone", "chip"])
        self.assertIn(4, voted)
        idx = SymbolKnowledgeIndex.build_from_corpus(
            {"d": "apple phone chip"},
            subjects={4},
        )
        active = idx.active_chambers_for_query(["apple", "phone"])
        self.assertIn(4, active)
        self.assertIn(MASTER_CHAMBER, active)

    def test_master_audit_disambiguates_apple(self) -> None:
        idx = SymbolKnowledgeIndex.build_from_corpus(
            {"d1": "apple phone chip silicon"},
            subjects={4},
        )
        idx.ingest_corpus({"d2": "apple pie fruit orchard"}, {11})
        report = master_audit(idx, min_direct_strength=1.0)
        self.assertGreater(report["n_disambiguated"], 0)
        self.assertEqual(report["n_smear_conflicts"], 0)

    def test_chamber_branch_wing_map(self) -> None:
        branch, wing = chamber_to_branch_wing(9)
        self.assertEqual(int(branch), 2)
        self.assertEqual(wing, 1)

    def test_save_load_chambers(self) -> None:
        idx = SymbolKnowledgeIndex.build_from_corpus(
            {"d": "quantum cell biology"},
            subjects={1, 9},
        )
        idx.correlates("quantum", "cell", chamber=1)
        idx.correlates("quantum", "cell", chamber=9)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ch.pkl"
            idx.save(path)
            loaded = SymbolKnowledgeIndex.load("ch", path=path)
            self.assertGreater(loaded.summary()["n_chambers"], 1)
            self.assertIn("0", loaded.summary()["chamber_link_counts"])

    def test_write_audit_log(self) -> None:
        idx = SymbolKnowledgeIndex.build_from_corpus(
            {"a": "apple phone chip", "b": "apple pie fruit"},
            subjects={4},
        )
        idx.ingest_corpus({"b": "apple pie fruit orchard"}, {11})
        with tempfile.TemporaryDirectory() as tmp:
            out = write_master_audit(idx, path=Path(tmp) / "audit.json")
            self.assertTrue(out.is_file())


if __name__ == "__main__":
    unittest.main()
