"""BIT 7 gate — meet witness index and routing."""
import unittest

from aethos_hub_signature import build_all_hub_signatures
from aethos_intersection_nodes import MeetKind
from aethos_token_processor import TokenProcessor
from aethos_tokenize import tokenize_words
from diagnose_corpus import SMALL_CORPUS
from pipeline.bit_07_meet_witness import (
    TRIPLE_PROMOTION_KEY,
    build_meet_witness_index,
    candidates_from_meet_witness,
    probe_solo_swap_witness,
    triple_promotion_key,
    triple_promotion_witness,
    verify_bit07_gate,
    verify_bit07_routing_gate,
)


class TestBit07MeetWitness(unittest.TestCase):
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
        cls.index = build_meet_witness_index(cls.sigs, cls.registry)

    def test_solo_swap_activates(self):
        w = probe_solo_swap_witness(3, 5)
        self.assertIsNotNone(w)
        self.assertEqual(w.kind, MeetKind.SOLO_SWAP)

    def test_triple_promotion_kappa(self):
        self.assertEqual(triple_promotion_key(3, 5, 7), TRIPLE_PROMOTION_KEY)
        w = triple_promotion_witness(3, 5, 7)
        cell = w.spacetime_cell(chain=(3, 5, 7))
        self.assertEqual(cell.z, complex(12, 5))

    def test_geometry_gate(self):
        ok, failures = verify_bit07_gate(self.registry, self.sigs)
        self.assertTrue(ok, failures)

    def test_legacy_dict_compatible(self):
        legacy = self.index.legacy_dict()
        self.assertIsInstance(legacy, dict)
        for p, docs in legacy.items():
            self.assertIsInstance(docs, set)
            self.assertGreaterEqual(p, 107)

    def test_meet_routing_finds_phone_docs(self):
        cands = candidates_from_meet_witness(
            ["phone"],
            self.registry,
            self.index,
            min_factor_hits=1,
        )
        phone_docs = {did for did, toks in self.doc_tokens.items() if "phone" in toks}
        self.assertTrue(phone_docs & set(cands))

    def test_min_two_hits_reduces_noise(self):
        loose = candidates_from_meet_witness(
            ["phone"],
            self.registry,
            self.index,
            min_factor_hits=1,
        )
        strict = candidates_from_meet_witness(
            ["phone"],
            self.registry,
            self.index,
            min_factor_hits=2,
        )
        self.assertLessEqual(len(strict), len(loose))

    def test_routing_gate(self):
        ok, avg, failures = verify_bit07_routing_gate(
            self.registry,
            self.sigs,
            self.doc_tokens,
            min_factor_hits=1,
        )
        self.assertTrue(ok or avg > 0, failures)


if __name__ == "__main__":
    unittest.main()
