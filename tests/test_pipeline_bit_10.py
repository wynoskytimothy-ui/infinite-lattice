"""BIT 10 gate — signal 8a κ Jaccard and optional candidate cap."""
import unittest

from aethos_hub_signature import (
    build_query_profile,
    rank_with_hub_signatures,
)
from aethos_token_processor import TokenProcessor
from aethos_tokenize import tokenize_words
from diagnose_corpus import SMALL_CORPUS
from eval_beir import build_neighbor_weights
from pipeline.bit_03_doc_attractor_set import build_attractor_index_from_hub_signatures
from pipeline.bit_09_query_cell_profile import build_query_cell_profile
from pipeline.bit_10_score_fusion import (
    GATE_LAMBDA_KAPPA,
    cap_candidates_by_kappa_overlap,
    signal_8a_kappa_jaccard,
    verify_bit10_gate,
)
from aethos_hub_signature import build_all_hub_signatures


class TestBit10ScoreFusion(unittest.TestCase):
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
        cls.index = build_attractor_index_from_hub_signatures(cls.registry, cls.sigs)
        cls.doc_freq: dict[str, int] = {}
        for toks in cls.doc_tokens.values():
            for w in toks:
                cls.doc_freq[w] = cls.doc_freq.get(w, 0) + 1
        cls.n_docs = len(cls.doc_ids)
        cls.neighbor_map = build_neighbor_weights(cls.registry)
        cls.query = "phone technical software"
        cls.profile = build_query_profile(
            cls.query,
            cls.registry,
            neighbor_map=cls.neighbor_map,
            doc_freq=cls.doc_freq,
            n_docs=cls.n_docs,
        )
        cls.cell_profile = build_query_cell_profile(
            cls.registry,
            cls.query,
            neighbor_map=cls.neighbor_map,
            doc_freq=cls.doc_freq,
            n_docs=cls.n_docs,
        )
        cls.kappa_keys = cls.cell_profile.kappa_neighbor_q

    def test_signal_8a_fires_on_overlap(self):
        s_d0 = signal_8a_kappa_jaccard(
            self.profile, "d0", self.index, self.kappa_keys,
            lambda_kappa=GATE_LAMBDA_KAPPA,
        )
        s_d3 = signal_8a_kappa_jaccard(
            self.profile, "d3", self.index, self.kappa_keys,
            lambda_kappa=GATE_LAMBDA_KAPPA,
        )
        self.assertGreater(s_d0, 0.0)
        self.assertGreater(s_d0, s_d3)
        expected = (
            max(self.profile.idf.values())
            * self.index.score_doc_overlap(self.kappa_keys, "d0")
            * GATE_LAMBDA_KAPPA
        )
        self.assertAlmostEqual(s_d0, expected, places=9)

    def test_cap_reduces_candidate_pool(self):
        pool = list(self.doc_ids) * 50
        capped = cap_candidates_by_kappa_overlap(
            pool,
            self.index,
            self.kappa_keys,
            cap=200,
        )
        self.assertLessEqual(len(capped), 200)
        self.assertGreater(len(capped), 0)
        overlap_first = self.index.score_doc_overlap(self.kappa_keys, capped[0])
        self.assertGreater(overlap_first, 0.0)

    def test_zero_bm25_query_still_gets_8a(self):
        empty_tokens = {did: frozenset() for did in self.doc_ids}
        ranked = rank_with_hub_signatures(
            self.profile,
            ["d0", "d3"],
            self.sigs,
            self.doc_ids,
            doc_tokens=empty_tokens,
            attractor_index=self.index,
            query_kappa_keys=self.kappa_keys,
            top_k=2,
        )
        self.assertEqual(ranked[0], "d0")

    def test_gated_doc_gets_no_8a(self):
        ranked_base = rank_with_hub_signatures(
            self.profile,
            ["d0", "d3"],
            self.sigs,
            self.doc_ids,
            doc_tokens=self.doc_tokens,
            top_k=2,
        )
        ranked_8a = rank_with_hub_signatures(
            self.profile,
            ["d0", "d3"],
            self.sigs,
            self.doc_ids,
            doc_tokens=self.doc_tokens,
            attractor_index=self.index,
            query_kappa_keys=self.kappa_keys,
            top_k=2,
        )
        self.assertEqual(ranked_base[0], "d0")
        self.assertEqual(ranked_8a[0], "d0")
        s8_d3 = signal_8a_kappa_jaccard(
            self.profile, "d3", self.index, self.kappa_keys,
        )
        if s8_d3 > 0.0:
            self.assertNotIn("d3", ranked_8a[:1])

    def test_gate_helper_passes(self):
        ok, failures = verify_bit10_gate(
            self.registry,
            self.profile,
            self.cell_profile,
            self.index,
            self.sigs,
            self.doc_ids,
            self.doc_ids,
            doc_tokens=self.doc_tokens,
        )
        self.assertTrue(ok, failures)

    def test_cap_zero_is_noop(self):
        original = list(self.doc_ids)
        capped = cap_candidates_by_kappa_overlap(
            original,
            self.index,
            self.kappa_keys,
            cap=0,
        )
        self.assertEqual(capped, original)


if __name__ == "__main__":
    unittest.main()
