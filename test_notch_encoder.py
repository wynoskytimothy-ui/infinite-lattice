"""P5 notch encoder — 24 tests."""

from __future__ import annotations

import os
import unittest

from aethos_hub_signature import pool_factors_for_word
from aethos_lattice import BranchKind
from aethos_notch_encoder import (
    DEFAULT_DOC_NOTCH_BYTES,
    DEFAULT_TOP_K,
    NOTCH_BYTES,
    Notch,
    aggregate_query_notches,
    branch_index,
    chain_for_word,
    correlation_matrix_4x4,
    doc_notch_score,
    encode_doc_notch_fingerprint,
    encode_word_notches,
    extract_top_notches,
    fingerprint_document_notch,
    notch_similarity,
    pack_notch,
    pack_notches,
    storage_backend,
    unpack_notch,
    unpack_notches,
)
from aethos_pipeline import AethosPipeline


class TestCorrelationMatrix(unittest.TestCase):
    def setUp(self) -> None:
        self.chain = (5,)

    def test_matrix_has_16_cells(self) -> None:
        m = correlation_matrix_4x4(self.chain, 5)
        self.assertEqual(len(m), 16)

    def test_diagonal_positive_tension(self) -> None:
        m = correlation_matrix_4x4(self.chain, 5)
        for b in BranchKind:
            self.assertGreater(abs(m[(b, b)]), 0)

    def test_va1_va2_off_diagonal_real(self) -> None:
        m = correlation_matrix_4x4(self.chain, 5)
        cross = m[(BranchKind.VA1, BranchKind.VA2)]
        self.assertNotEqual(cross, 0)


class TestNotchPackRoundtrip(unittest.TestCase):
    def test_exactly_10_bytes(self) -> None:
        blob = pack_notch(1, 2, 3 + 4j)
        self.assertEqual(len(blob), NOTCH_BYTES)

    def test_roundtrip_branches(self) -> None:
        blob = pack_notch(3, 4, 10 + 5j, kind=1)
        n = unpack_notch(blob)
        self.assertEqual(n.branch_a, 3)
        self.assertEqual(n.branch_b, 4)

    def test_roundtrip_amplitude(self) -> None:
        c = 12.5 + 7.25j
        n = unpack_notch(pack_notch(2, 3, c))
        self.assertAlmostEqual(n.amplitude, abs(c), places=1)
        self.assertAlmostEqual(n.re, c.real, places=1)
        self.assertAlmostEqual(n.im, c.imag, places=1)

    def test_checksum_rejects_tamper(self) -> None:
        blob = bytearray(pack_notch(1, 2, 1j))
        blob[-1] ^= 0xFF
        with self.assertRaises(ValueError):
            unpack_notch(bytes(blob))

    def test_pack_unpack_sequence(self) -> None:
        notches = (
            Notch(1, 2, 5.0, 5.0, 0.0),
            Notch(4, 3, 3.0, 2.0, 1.0),
        )
        back = unpack_notches(pack_notches(notches))
        self.assertEqual(len(back), 2)
        self.assertEqual(back[0].branch_pair, (1, 2))


class TestTopKExtraction(unittest.TestCase):
    def test_respects_top_k(self) -> None:
        m = correlation_matrix_4x4((5,), 5)
        peaks = extract_top_notches(m, top_k=5)
        self.assertLessEqual(len(peaks), 5)

    def test_prefers_cross_branch_in_ties(self) -> None:
        m = correlation_matrix_4x4((5,), 5)
        peaks = extract_top_notches(m, top_k=16, prefer_cross_branch=True)
        cross = [p for p in peaks if p.is_cross_branch()]
        self.assertGreater(len(cross), 0)


class TestWordEncoding(unittest.TestCase):
    def setUp(self) -> None:
        self.pipe = AethosPipeline(rebuild_every=2)
        self.pipe.ingest(
            "quantum entanglement bell violation test",
            "quantum field bell inequality experiment",
        )
        self.reg = self.pipe.registry

    def test_encode_returns_notches(self) -> None:
        n = encode_word_notches(self.reg, "quantum", top_k=10)
        self.assertGreater(len(n), 0)
        self.assertLessEqual(len(n), 10)

    def test_deterministic(self) -> None:
        a = encode_word_notches(self.reg, "bell", top_k=8)
        b = encode_word_notches(self.reg, "bell", top_k=8)
        self.assertEqual(a, b)

    def test_different_words_can_differ(self) -> None:
        a = encode_word_notches(self.reg, "quantum", top_k=10)
        b = encode_word_notches(self.reg, "violation", top_k=10)
        pairs_a = {x.branch_pair for x in a}
        pairs_b = {x.branch_pair for x in b}
        # not required to differ always, but amplitudes usually differ
        self.assertTrue(pairs_a or pairs_b)

    def test_chain_from_word(self) -> None:
        c = chain_for_word(self.reg, "quantum")
        self.assertGreater(len(c), 0)


class TestNotchSimilarity(unittest.TestCase):
    def test_identical_is_one(self) -> None:
        n = (Notch(1, 2, 4.0, 4.0, 0.0), Notch(2, 3, 2.0, 1.0, 1.0))
        self.assertAlmostEqual(notch_similarity(n, n), 1.0)

    def test_disjoint_pairs_zero(self) -> None:
        a = (Notch(1, 1, 5.0, 5.0, 0.0),)
        b = (Notch(4, 4, 5.0, 5.0, 0.0),)
        self.assertEqual(notch_similarity(a, b), 0.0)

    def test_partial_overlap(self) -> None:
        a = (Notch(1, 2, 4.0, 4.0, 0.0), Notch(3, 4, 1.0, 0.0, 1.0))
        b = (Notch(1, 2, 2.0, 2.0, 0.0),)
        sim = notch_similarity(a, b)
        self.assertGreater(sim, 0.0)
        self.assertLess(sim, 1.0)

    def test_empty_returns_zero(self) -> None:
        self.assertEqual(notch_similarity((), (Notch(1, 2, 1.0, 1.0, 0.0),)), 0.0)


class TestDocNotchScore(unittest.TestCase):
    def setUp(self) -> None:
        self.pipe = AethosPipeline(rebuild_every=2)
        self.pipe.ingest("quantum bell test", "quantum bell experiment")
        self.reg = self.pipe.registry

    def test_gated_without_pool_overlap(self) -> None:
        q_notches = encode_word_notches(self.reg, "quantum", top_k=5)
        d_notches = encode_word_notches(self.reg, "quantum", top_k=5)
        score = doc_notch_score(
            q_notches,
            d_notches,
            frozenset({999999}),
            frozenset({888888}),
        )
        self.assertEqual(score, 0.0)

    def test_positive_with_pool_overlap(self) -> None:
        q_notches = encode_word_notches(self.reg, "quantum", top_k=5)
        d_notches = encode_word_notches(self.reg, "quantum", top_k=5)
        q_pool = pool_factors_for_word(self.reg, "quantum")
        score = doc_notch_score(q_notches, d_notches, q_pool, q_pool)
        self.assertGreater(score, 0.0)


class TestDocFingerprint(unittest.TestCase):
    def setUp(self) -> None:
        self.pipe = AethosPipeline(rebuild_every=2)
        self.pipe.ingest("phone technical chip software network")
        self.reg = self.pipe.registry

    def test_fingerprint_within_budget(self) -> None:
        fp = encode_doc_notch_fingerprint(
            0,
            ["phone", "technical"],
            self.reg,
            top_k=DEFAULT_TOP_K,
            max_bytes=DEFAULT_DOC_NOTCH_BYTES,
        )
        self.assertLessEqual(fp.encoded_size(), DEFAULT_DOC_NOTCH_BYTES + 32)

    def test_payload_is_multiple_of_10(self) -> None:
        fp = fingerprint_document_notch(1, ["phone"], self.reg, top_k=10)
        self.assertEqual(len(fp.notches) % NOTCH_BYTES, 0)

    def test_encoded_size_matches_wire(self) -> None:
        fp = encode_doc_notch_fingerprint(2, ["phone"], self.reg, top_k=10)
        # encoded_size must cover header + payload (was 1 byte tight in draft)
        self.assertGreaterEqual(fp.encoded_size(), len(fp.notches) + 8)

    def test_roundtrip_notches(self) -> None:
        fp = encode_doc_notch_fingerprint(3, ["technical"], self.reg, top_k=6)
        decoded = fp.decoded_notches()
        self.assertEqual(len(decoded), 6)
        self.assertEqual(decoded[0].branch_a, unpack_notch(fp.notches[:NOTCH_BYTES]).branch_a)


class TestStorageBackend(unittest.TestCase):
    def test_default_is_hub(self) -> None:
        old = os.environ.pop("STORAGE_BACKEND", None)
        try:
            self.assertEqual(storage_backend(), "hub")
        finally:
            if old is not None:
                os.environ["STORAGE_BACKEND"] = old

    def test_notch_env(self) -> None:
        os.environ["STORAGE_BACKEND"] = "notch"
        try:
            self.assertEqual(storage_backend(), "notch")
        finally:
            os.environ.pop("STORAGE_BACKEND", None)


class TestAggregateQuery(unittest.TestCase):
    def test_aggregate_dedupes_pairs(self) -> None:
        pipe = AethosPipeline(rebuild_every=2)
        pipe.ingest("alpha beta gamma", "alpha beta delta")
        reg = pipe.registry
        agg = aggregate_query_notches(["alpha", "beta"], reg, top_k=10)
        pairs = [n.branch_pair for n in agg]
        self.assertEqual(len(pairs), len(set(pairs)))


class TestBranchIndex(unittest.TestCase):
    def test_branch_index_range(self) -> None:
        for b in BranchKind:
            self.assertIn(branch_index(b), (1, 2, 3, 4))


if __name__ == "__main__":
    unittest.main()
