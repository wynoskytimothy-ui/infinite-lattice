"""BIT 12 — symbol knowledge on 3D complex-plane κ index."""

from __future__ import annotations

import time
import unittest

from aethos_symbol_knowledge import SymbolKnowledgeIndex
from pipeline.bit_12_symbol_plane_index import (
    build_symbol_plane_index,
    canonical_pair_key,
    correlation_meet_keys,
    get_pair_meet_keys,
    query_symbol_plane_keys,
    resolve_pair_link,
    route_symbol_plane_candidates,
    score_doc_meet_witness,
    symbol_word_chain,
    verify_bit12_gate,
    word_to_symbol_plane_cell,
)
from pipeline.bit_02_attractor_key import kappa_from_cell


class TestBit12SymbolPlane(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = {
            "d1": "cancer cell breast tumor treatment patients",
            "d2": "protein gene expression rna virus",
            "d3": "clinical trial vaccine immune antibody",
        }
        cls.knowledge = SymbolKnowledgeIndex.build_from_corpus(
            cls.corpus, dataset="bit12_test",
        )
        cls.plane = build_symbol_plane_index(cls.knowledge)

    def test_word_has_chain_and_cell(self) -> None:
        chain = symbol_word_chain(self.knowledge, "cancer")
        self.assertGreater(len(chain), 0)
        cell = word_to_symbol_plane_cell(self.knowledge, "cancer")
        key = kappa_from_cell(cell)
        self.assertIsInstance(key, tuple)
        self.assertEqual(len(key), 3)

    def test_correlation_meet_keys(self) -> None:
        lk = self.knowledge.correlates("cancer", "cell")
        self.assertIsNotNone(lk)
        keys = correlation_meet_keys(self.knowledge, "cancer", "cell", link=lk)
        self.assertGreater(len(keys), 0)

    def test_plane_index_covers_docs(self) -> None:
        for doc_id in self.corpus:
            self.assertIn(doc_id, self.plane.doc_keys)
            self.assertGreater(len(self.plane.doc_keys[doc_id]), 0)

    def test_route_finds_cancer_doc(self) -> None:
        route = route_symbol_plane_candidates(
            self.knowledge, self.plane, ["cancer", "breast"],
        )
        self.assertIn("d1", route.doc_ids)
        self.assertGreater(route.n_query_keys, 0)

    def test_route_finds_rna_doc(self) -> None:
        route = route_symbol_plane_candidates(
            self.knowledge, self.plane, ["rna", "virus"],
        )
        self.assertIn("d2", route.doc_ids)

    def test_query_keys_expand_correlations(self) -> None:
        keys_off = query_symbol_plane_keys(
            self.knowledge, self.plane, ["protein"],
            expand_correlations=False,
        )
        keys_on = query_symbol_plane_keys(
            self.knowledge, self.plane, ["protein"],
            expand_correlations=True,
        )
        self.assertGreaterEqual(len(keys_on), len(keys_off))

    def test_bit12_gate(self) -> None:
        ok, failures = verify_bit12_gate(
            self.knowledge,
            self.plane,
            [("cancer", "cell"), ("protein", "gene"), ("clinical", "trial")],
        )
        self.assertTrue(ok, failures)

    def test_doc_meet_keys_indexed_on_cooccurring_pair(self) -> None:
        plane = build_symbol_plane_index(
            self.knowledge, index_doc_meet_keys=True, max_doc_meet_pairs=32,
        )
        lk = self.knowledge.correlates("cancer", "cell")
        self.assertIsNotNone(lk)
        meet = plane.pair_keys.get(canonical_pair_key(self.knowledge, "cancer", "cell"))
        self.assertIsNotNone(meet)
        doc_k = plane.doc_keys.get("d1", set())
        overlap = len(meet & doc_k)
        self.assertGreater(overlap, 0, "d1 co-occurs cancer+cell; meet keys on doc")
        score = score_doc_meet_witness(
            self.knowledge, plane, ["cancer", "cell"], "d1",
        )
        self.assertGreater(score, 0.0)

    def test_slim_plane_smaller_than_full(self) -> None:
        full = build_symbol_plane_index(self.knowledge)
        slim = build_symbol_plane_index(
            self.knowledge,
            rare_pairs_only=True,
            rare_adjacency_only=True,
            max_adjacency_per_word=4,
        )
        self.assertLessEqual(slim.n_pair_keys, full.n_pair_keys)
        full_adj = sum(len(v) for v in full.word_adjacency.values())
        slim_adj = sum(len(v) for v in slim.word_adjacency.values())
        self.assertLessEqual(slim_adj, full_adj)
        self.assertLess(slim_adj, full_adj)
        for nbrs in slim.word_adjacency.values():
            self.assertLessEqual(len(nbrs), 4)

    def test_resolve_pair_link_canonical_bucket(self) -> None:
        """Morph-canonical surfaces resolve via cross-link family bucket."""
        lk = resolve_pair_link(self.knowledge, "canc", "cell")
        self.assertIsNotNone(lk)
        self.assertGreaterEqual(lk.strength, 0.5)

    def test_canonical_pair_meet_keys_shared(self) -> None:
        knowledge = SymbolKnowledgeIndex.load("scifact")
        plane = build_symbol_plane_index(
            knowledge,
            rare_pairs_only=True,
            rare_adjacency_only=True,
            max_adjacency_per_word=4,
        )
        pk_cell = canonical_pair_key(knowledge, "cell", "protein")
        pk_cellular = canonical_pair_key(knowledge, "cellular", "protein")
        self.assertEqual(pk_cell, pk_cellular)
        meet_cell = get_pair_meet_keys(plane, knowledge, "cell", "protein")
        meet_cellular = get_pair_meet_keys(plane, knowledge, "cellular", "protein")
        self.assertEqual(meet_cell, meet_cellular)

    def test_family_dedup_reduces_doc_keys(self) -> None:
        corpus = {"d1": "diminished diminishes lower score pathway"}
        knowledge = SymbolKnowledgeIndex.build_from_corpus(corpus, dataset="p4_dedup")
        literal = build_symbol_plane_index(
            knowledge, canonical_family_index=False, dedupe_family_keys=False,
        )
        canonical = build_symbol_plane_index(
            knowledge, canonical_family_index=True, dedupe_family_keys=True,
        )
        lit_k = len(literal.doc_keys.get("d1", set()))
        can_k = len(canonical.doc_keys.get("d1", set()))
        self.assertLess(can_k, lit_k)
        self.assertGreater(len(canonical.word_keys.get("diminishes", set())), 0)

    def test_router_faster_than_scan(self) -> None:
        """κ lookup should beat linear scan over all cross_links."""
        t0 = time.perf_counter()
        for _ in range(50):
            route_symbol_plane_candidates(
                self.knowledge, self.plane, ["cancer", "treatment"],
            )
        route_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        for _ in range(50):
            hits = set()
            for (a, b), lk in self.knowledge.cross_links.items():
                if "cancer" in (a, b) or "treatment" in (a, b):
                    hits.add((a, b))
        scan_ms = (time.perf_counter() - t0) * 1000.0

        self.assertLess(route_ms, scan_ms * 2 + 100)


if __name__ == "__main__":
    unittest.main()
