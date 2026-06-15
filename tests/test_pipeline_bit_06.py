"""BIT 6 gate — notch fingerprint bound to κ(cell(w*))."""
import unittest

from aethos_hub_signature import build_all_hub_signatures
from aethos_token_processor import TokenProcessor
from aethos_tokenize import tokenize_words
from diagnose_corpus import SMALL_CORPUS
from pipeline.bit_01_word_cell import word_to_spacetime_cell
from pipeline.bit_02_attractor_key import kappa_from_cell
from pipeline.bit_06_notch_bind import (
    bind_notch_from_hub_signature,
    build_all_notch_fingerprints,
    encode_notches_for_word,
    score_bound_notch_pair,
    verify_bit06_gate,
)
from aethos_notch_encoder import NOTCH_BYTES, notch_similarity


class TestBit06NotchBind(unittest.TestCase):
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
        cls.fps = build_all_notch_fingerprints(cls.sigs, cls.registry)

    def test_gate_passes(self):
        ok, failures = verify_bit06_gate(self.registry, self.sigs)
        self.assertTrue(ok, failures)

    def test_rebuild_identical_bytes(self):
        a = build_all_notch_fingerprints(self.sigs, self.registry)
        b = build_all_notch_fingerprints(self.sigs, self.registry)
        for doc_id in a:
            self.assertEqual(a[doc_id].notches, b[doc_id].notches)

    def test_kappa_matches_top_hub_cell(self):
        for doc_id, fp in self.fps.items():
            cell = word_to_spacetime_cell(self.registry, fp.top_hub)
            self.assertEqual(fp.attractor_key, kappa_from_cell(cell))

    def test_payload_budget(self):
        for fp in self.fps.values():
            self.assertLessEqual(fp.payload_bytes, 100)
            self.assertEqual(len(fp.notches) % NOTCH_BYTES, 0)

    def test_same_word_same_notches(self):
        a = encode_notches_for_word(self.registry, "phone")
        b = encode_notches_for_word(self.registry, "phone")
        self.assertEqual(a, b)

    def test_self_similarity_one(self):
        fp = self.fps["d0"]
        sim = notch_similarity(fp.decoded_notches(), fp.decoded_notches())
        self.assertAlmostEqual(sim, 1.0)

    def test_score_same_hub_word(self):
        fp = next(iter(self.fps.values()))
        score = score_bound_notch_pair([fp.top_hub], fp, self.registry)
        self.assertGreater(score, 0.0)

    def test_bind_from_signature(self):
        sig = self.sigs["d2"]
        fp = bind_notch_from_hub_signature(sig, self.registry)
        self.assertIsNotNone(fp)
        self.assertIn(fp.top_hub, sig.hubs)


if __name__ == "__main__":
    unittest.main()
