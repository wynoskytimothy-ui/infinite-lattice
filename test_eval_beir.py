"""Tests for BEIR eval harness — hub signature scoring path."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aethos_hub_signature import (
    LatticeHubSignature,
    QueryProfile,
    build_hub_signature,
    prime_factor_meet_score,
    rank_with_hub_signatures,
    score_hub_signature,
    signature_report,
)
from core.phi_lattice import prime_factor_similarity
from aethos_pipeline import AethosPipeline
from aethos_tokenize import tokenize_words
from eval_beir import (
    build_meet_index,
    candidate_generation_tier,
    candidate_ids,
    doc_text,
    load_corpus,
    merge_qrels,
    ndcg_at_k,
    recall_at_k,
)
from aethos_hub_signature import build_all_hub_signatures


class TestBeirMetrics(unittest.TestCase):
    def test_ndcg_perfect(self) -> None:
        rel = {"a": 1, "b": 1}
        ranked = ["a", "b", "c"]
        self.assertAlmostEqual(ndcg_at_k(ranked, rel, 2), 1.0)

    def test_recall(self) -> None:
        rel = {"a": 1, "b": 1, "c": 1}
        ranked = ["x", "a", "b"]
        self.assertAlmostEqual(recall_at_k(ranked, rel, 2), 1 / 3)

    def test_ndcg_miss(self) -> None:
        rel = {"z": 1}
        ranked = ["a", "b", "c"]
        self.assertEqual(ndcg_at_k(ranked, rel, 10), 0.0)


class TestHubSignature(unittest.TestCase):
    def setUp(self) -> None:
        self.pipe = AethosPipeline(rebuild_every=2)
        self.docs = (
            "phone technical chip software network",
            "apple fruit pie orchard dessert",
            "zebra animal wildlife savanna plain",
        )
        self.pipe.ingest(*self.docs)

    def test_build_hub_signature_basic(self) -> None:
        sig = build_hub_signature("d0", self.docs[0], self.pipe.registry, top_k=4)
        self.assertEqual(sig.doc_id, "d0")
        self.assertGreater(len(sig.hubs), 0)
        self.assertLessEqual(len(sig.hubs), 4)

    def test_hub_words_are_subset_of_doc(self) -> None:
        doc = self.docs[0]
        sig = build_hub_signature("d0", doc, self.pipe.registry, top_k=8)
        doc_words = set(tokenize_words(doc))
        for w in sig.hub_words():
            self.assertIn(w, doc_words)

    def test_hub_coords_consistent(self) -> None:
        sig = build_hub_signature(
            "d0", self.docs[0], self.pipe.registry, top_k=8, use_pin_wire=False
        )
        for coord, word in sig.hub_coords.items():
            self.assertIn(word, sig.hubs)
            self.assertEqual(sig.hubs[word].coord, coord)

    def test_pin_wire_uses_pin_maps_not_float_coords(self) -> None:
        sig = build_hub_signature(
            "d0", self.docs[0], self.pipe.registry, top_k=8, use_pin_wire=True
        )
        self.assertEqual(len(sig.hub_coords), 0)
        self.assertGreater(len(sig.hub_pins), 0)
        self.assertGreater(len(sig.hub_leg_sum), 0)
        for entry in sig.hubs.values():
            self.assertIsNone(entry.coord)
            self.assertIsNotNone(entry.pin)

    def test_hub_pool_factors_exclude_letters_only(self) -> None:
        sig = build_hub_signature("d0", self.docs[0], self.pipe.registry, top_k=8)
        for entry in sig.hubs.values():
            for p in entry.pool_factors:
                self.assertGreaterEqual(p, 107)

    def test_encoded_size_positive(self) -> None:
        sig = build_hub_signature("d0", self.docs[0], self.pipe.registry, top_k=4)
        self.assertGreater(sig.encoded_size(), 0)

    def test_signature_report_runs(self) -> None:
        sig = build_hub_signature("d0", self.docs[0], self.pipe.registry, top_k=4)
        report = signature_report(sig)
        self.assertIn("d0", report)


class TestHubScoring(unittest.TestCase):
    def setUp(self) -> None:
        self.pipe = AethosPipeline(rebuild_every=2)
        self.docs = (
            "phone technical chip software network",
            "apple fruit pie orchard dessert",
        )
        self.pipe.ingest(*self.docs)
        self.sigs = {
            str(i): build_hub_signature(str(i), d, self.pipe.registry, top_k=8)
            for i, d in enumerate(self.docs)
        }

    def _profile(self, query: str) -> QueryProfile:
        from aethos_hub_signature import build_query_profile
        from eval_beir import build_neighbor_weights, build_corpus_index
        from aethos_tokenize import tokenize_words
        nm = build_neighbor_weights(self.pipe.registry)
        from collections import Counter
        doc_tokens = {}; doc_tf = {}; doc_len = {}
        for i, d in enumerate(self.docs):
            words = tokenize_words(d)
            tf = Counter(words)
            doc_tokens[str(i)] = frozenset(tf.keys())
            doc_tf[str(i)] = dict(tf)
            doc_len[str(i)] = len(words)
        cidx = build_corpus_index(list(doc_tokens.keys()), doc_tokens, doc_tf, doc_len)
        return build_query_profile(
            query,
            self.pipe.registry,
            neighbor_map=nm,
            doc_freq=cidx.doc_freq,
            n_docs=len(self.docs),
        )

    def test_phone_query_prefers_tech_doc(self) -> None:
        profile = self._profile("phone chip technical")
        s0 = score_hub_signature(profile, self.sigs["0"])
        s1 = score_hub_signature(profile, self.sigs["1"])
        self.assertGreater(s0, s1)

    def test_fruit_query_prefers_food_doc(self) -> None:
        profile = self._profile("apple fruit pie")
        s0 = score_hub_signature(profile, self.sigs["0"])
        s1 = score_hub_signature(profile, self.sigs["1"])
        self.assertGreater(s1, s0)

    def test_rank_with_hub_signatures_order(self) -> None:
        profile = self._profile("phone technical")
        ranked = rank_with_hub_signatures(
            profile, list(self.sigs.keys()), self.sigs, list(self.sigs.keys()), top_k=2
        )
        self.assertEqual(ranked[0], "0")

    def test_prime_factor_meet_nonzero(self) -> None:
        profile = self._profile("phone chip")
        self.assertGreater(len(profile.word_pool_factors), 0)
        s = prime_factor_meet_score(profile, self.sigs["0"])
        self.assertGreaterEqual(s, 0.0)

    def test_letter_only_word_skipped_in_signal5b(self) -> None:
        from aethos_hub_signature import _pool_factors, pool_factor_jaccard

        ent = _pool_factors(2, ())  # letter prime only — no pool factors
        self.assertEqual(len(ent), 0)
        self.assertEqual(pool_factor_jaccard(ent, frozenset({107})), 0.0)

    def test_prime_factor_similarity_apple_fixture(self) -> None:
        tech = 101 * 103 * 107
        food = 101 * 109 * 113
        q_phone = 101 * 103
        self.assertGreater(prime_factor_similarity(q_phone, tech), 0.6)
        self.assertLess(prime_factor_similarity(q_phone, food), 0.35)


class TestP4CandidateTiers(unittest.TestCase):
    def setUp(self) -> None:
        self.pipe = AethosPipeline(rebuild_every=2)
        self.docs = {
            "d0": "zebra wildlife savanna migration ecology",
            "d1": "phone technical chip software network",
        }
        self.pipe.ingest(self.docs["d0"], self.docs["d1"])
        self.tokens = {
            "d0": frozenset(w for w in self.docs["d0"].split()),
            "d1": frozenset(w for w in self.docs["d1"].split()),
        }
        self.inv: dict[str, set[str]] = {}
        for did, toks in self.tokens.items():
            for w in toks:
                self.inv.setdefault(w, set()).add(did)
        self.all_ids = ["d0", "d1"]
        self.hub_sigs = build_all_hub_signatures(
            self.all_ids,
            self.tokens,
            self.pipe.registry,
            top_k=8,
        )
        self.meet_index = build_meet_index(self.hub_sigs, self.pipe.registry)

    def test_tier1_lexical_when_overlap(self) -> None:
        tier = candidate_generation_tier(
            ["phone", "technical"],
            self.inv,
            {},
            self.all_ids,
            meet_index=self.meet_index,
            registry=self.pipe.registry,
        )
        self.assertEqual(tier, "tier1_lexical")
        cands = candidate_ids(
            ["phone", "technical"],
            self.inv,
            {},
            self.all_ids,
            meet_index=self.meet_index,
            registry=self.pipe.registry,
        )
        self.assertEqual(set(cands), {"d1"})

    def test_tier3_not_full_corpus_on_oov_query(self) -> None:
        """OOV query must not fall back to all_ids when meet index exists."""
        tier = candidate_generation_tier(
            ["quantum", "entanglement"],
            self.inv,
            {},
            self.all_ids,
            meet_index=self.meet_index,
            registry=self.pipe.registry,
        )
        self.assertIn(
            tier,
            ("tier2_meet_exact", "tier3_meet_pool", "tier3_meet_fuzzy", "tier4_full_corpus"),
        )
        cands = candidate_ids(
            ["quantum", "entanglement"],
            self.inv,
            {},
            self.all_ids,
            meet_index=self.meet_index,
            registry=self.pipe.registry,
        )
        self.assertLessEqual(len(cands), len(self.all_ids))


class TestBeirLoaders(unittest.TestCase):
    def test_load_mini_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "corpus.jsonl"
            p.write_text(
                '{"_id":"1","title":"A","text":"hello"}\n'
                '{"_id":"2","title":"B","text":"world"}\n',
                encoding="utf-8",
            )
            docs = load_corpus(p, max_docs=1)
            self.assertEqual(len(docs), 1)
            self.assertEqual(doc_text(docs["1"]), "A hello")


if __name__ == "__main__":
    unittest.main()
