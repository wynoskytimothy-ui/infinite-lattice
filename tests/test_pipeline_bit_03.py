"""BIT 3 gate — doc attractor sets K(doc) from hub signatures."""
import unittest

from aethos_hub_signature import build_all_hub_signatures
from aethos_token_processor import TokenProcessor
from aethos_tokenize import tokenize_words
from diagnose_corpus import SMALL_CORPUS
from pipeline.bit_03_doc_attractor_set import (
    build_attractor_index_from_hub_signatures,
    doc_attractor_set_from_signature,
    top_hub_key_for_doc,
    verify_bit03_gate,
)


class TestBit03DocAttractorSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipe = TokenProcessor()
        cls.doc_tokens = {
            f"d{i}": frozenset(tokenize_words(text))
            for i, text in enumerate(SMALL_CORPUS)
        }
        cls.doc_ids = list(cls.doc_tokens.keys())
        cls.pipe.ingest(*SMALL_CORPUS)
        cls.registry = cls.pipe.registry
        cls.sigs = build_all_hub_signatures(
            cls.doc_ids,
            cls.doc_tokens,
            cls.registry,
            top_k=12,
        )
        cls.index = build_attractor_index_from_hub_signatures(
            cls.registry,
            cls.sigs,
        )

    def test_every_doc_has_keys(self):
        for doc_id in self.doc_ids:
            keys = self.index.doc_keys.get(doc_id, set())
            self.assertGreater(len(keys), 0, doc_id)

    def test_duplicate_kappa_collapses_to_one_witness(self):
        for doc_id, sig in self.sigs.items():
            doc_set = doc_attractor_set_from_signature(self.registry, sig)
            self.assertEqual(len(doc_set.keys), len(doc_set.witnesses))

    def test_top_hub_retrieves_doc(self):
        for doc_id, sig in self.sigs.items():
            top = top_hub_key_for_doc(self.registry, sig)
            self.assertIsNotNone(top)
            key, _word = top
            hits = self.index.query_by_key(key, radius=0)
            self.assertIn(doc_id, hits)

    def test_gate_passes_on_corpus(self):
        passed, total, failures = verify_bit03_gate(
            self.registry,
            self.sigs,
            sample_size=50,
        )
        self.assertEqual(failures, [], failures)
        self.assertEqual(passed, total)

    def test_triple_bucket_isolated(self):
        """BIT 2+3: κ=(12,5,15) at r=0 only hits docs we explicitly add."""
        from aethos_physics import SpacetimeCell
        from pipeline.bit_02_attractor_key import kappa_from_cell

        cell = SpacetimeCell.promote_witness((3, 5, 7), (3, 5))
        key = kappa_from_cell(cell)
        idx = build_attractor_index_from_hub_signatures(self.registry, {})
        idx.add("only_triple", key, "witness")
        idx.add("other", (99, 99, 99), "noise")
        hits = idx.query_by_key(key, radius=0)
        self.assertEqual(hits, ["only_triple"])


if __name__ == "__main__":
    unittest.main()
