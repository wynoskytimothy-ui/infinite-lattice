"""BIT 9 gate — unified QueryCellProfile (cells + κ + z_obs)."""
import math
import unittest

from aethos_hub_signature import build_all_hub_signatures, build_query_profile
from aethos_token_processor import TokenProcessor
from aethos_tokenize import tokenize_words
from diagnose_corpus import SMALL_CORPUS
from eval_beir import build_neighbor_weights
from pipeline.bit_03_doc_attractor_set import build_attractor_index_from_hub_signatures
from pipeline.bit_04_candidate_router import query_attractor_keys
from pipeline.bit_09_query_cell_profile import (
    QueryCellProfile,
    bm25_idf,
    build_query_cell_profile,
    cells_match_hub_gate,
    verify_bit09_gate,
)


class TestBit09QueryCellProfile(unittest.TestCase):
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

    def _profile(self, query: str) -> QueryCellProfile:
        return build_query_cell_profile(
            self.registry,
            query,
            neighbor_map=self.neighbor_map,
            doc_freq=self.doc_freq,
            n_docs=self.n_docs,
        )

    def test_gate_passes_on_small_corpus(self):
        ok, failures = verify_bit09_gate(
            self.registry,
            neighbor_map=self.neighbor_map,
            doc_freq=self.doc_freq,
            n_docs=self.n_docs,
            index=self.index,
        )
        self.assertTrue(ok, failures)

    def test_idf_matches_build_query_profile(self):
        query = "phone technical software"
        profile = self._profile(query)
        hub = build_query_profile(
            query,
            self.registry,
            neighbor_map=self.neighbor_map,
            doc_freq=self.doc_freq,
            n_docs=self.n_docs,
        )
        for w, score in hub.idf.items():
            self.assertAlmostEqual(profile.idf.get(w, -1.0), score, places=9)

    def test_kappa_neighbor_superset(self):
        profile = self._profile("phone technical")
        self.assertTrue(profile.kappa_q.issubset(profile.kappa_neighbor_q))
        plain = query_attractor_keys(
            self.registry,
            profile.words,
            neighbor_map=self.neighbor_map,
            expand_neighbors=False,
        )
        expanded = query_attractor_keys(
            self.registry,
            profile.words,
            neighbor_map=self.neighbor_map,
            expand_neighbors=True,
        )
        self.assertEqual(profile.kappa_q, plain)
        self.assertEqual(profile.kappa_neighbor_q, expanded)

    def test_cells_match_bit01_hub(self):
        profile = self._profile("phone apple technical")
        failures = cells_match_hub_gate(self.registry, profile)
        self.assertEqual(failures, [])

    def test_z_obs_finite_and_positive(self):
        profile = self._profile("phone technical")
        self.assertTrue(math.isfinite(profile.z_obs_q))
        self.assertGreater(profile.z_obs_q, 0.0)
        for w in profile.routed_words:
            if w in profile.cells:
                self.assertIn(profile.band_ids[w], range(4))

    def test_kappa_overlap_related_beats_unrelated(self):
        profile = self._profile("phone technical")
        related = self.index.score_doc_overlap(profile.kappa_q, "d0")
        unrelated = self.index.score_doc_overlap(profile.kappa_q, "d3")
        self.assertGreater(related, unrelated)

    def test_routed_words_exclude_stopwords(self):
        profile = self._profile("the phone has technical software")
        self.assertIn("phone", profile.cells)
        self.assertIn("technical", profile.cells)
        self.assertNotIn("the", profile.cells)
        self.assertNotIn("has", profile.cells)


if __name__ == "__main__":
    unittest.main()
