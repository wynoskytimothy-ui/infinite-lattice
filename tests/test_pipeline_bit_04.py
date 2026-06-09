"""BIT 4 gate — attractor candidate router C(q)."""
import unittest

from aethos_hub_signature import build_all_hub_signatures
from aethos_token_processor import TokenProcessor
from aethos_tokenize import tokenize_words
from diagnose_corpus import SMALL_CORPUS
from eval_beir import build_meet_index, build_neighbor_weights, candidate_ids
from pipeline.bit_03_doc_attractor_set import build_attractor_index_from_hub_signatures
from pipeline.bit_04_candidate_router import (
    DEFAULT_RADIUS,
    candidates_from_attractors,
    gold_recall_in_candidates,
    query_attractor_keys,
    route_query_candidates,
    verify_bit04_gate,
    verify_bit04_gate_legacy_tuple,
)
from pipeline.bit_07_meet_witness import build_meet_witness_index


class TestBit04CandidateRouter(unittest.TestCase):
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
        cls.inv: dict[str, set[str]] = {}
        for did, toks in cls.doc_tokens.items():
            for w in toks:
                cls.inv.setdefault(w, set()).add(did)
        cls.neighbor_map = build_neighbor_weights(cls.registry)
        cls.meet_index = build_meet_witness_index(cls.sigs, cls.registry)

    def test_query_keys_nonempty_for_content_word(self):
        keys = query_attractor_keys(self.registry, ["phone"])
        self.assertGreater(len(keys), 0)

    def test_attractor_finds_phone_doc(self):
        cands, keys = candidates_from_attractors(
            ["phone"],
            self.registry,
            self.index,
            radius=DEFAULT_RADIUS,
        )
        self.assertGreater(len(keys), 0)
        phone_docs = self.inv.get("phone", set())
        self.assertTrue(phone_docs & set(cands))

    def test_router_uses_bit4_tier_when_enough_hits(self):
        route = route_query_candidates(
            ["phone", "technical"],
            self.registry,
            self.index,
            self.inv,
            self.neighbor_map,
            self.doc_ids,
            min_candidates=1,
            meet_index=self.meet_index,
        )
        self.assertIn(route.tier, ("bit4_attractor", "bit4_attractor_lexical", "bit4_meet_supplement"))
        self.assertGreater(route.n_merged, 0)

    def test_eval_candidate_ids_with_attractor_index(self):
        cands = candidate_ids(
            ["phone", "technical"],
            self.inv,
            self.neighbor_map,
            self.doc_ids,
            meet_index=self.meet_index,
            registry=self.registry,
            attractor_index=self.index,
            min_attractor_candidates=1,
        )
        self.assertIn("d1", cands)

    def test_neighbor_kappa_expands_keys(self):
        keys_plain = query_attractor_keys(self.registry, ["phone"])
        keys_nb = query_attractor_keys(
            self.registry,
            ["phone"],
            neighbor_map=self.neighbor_map,
            expand_neighbors=True,
        )
        self.assertGreaterEqual(len(keys_nb), len(keys_plain))

    def test_default_route_smaller_than_lexical_union(self):
        route_default = route_query_candidates(
            ["phone", "technical"],
            self.registry,
            self.index,
            self.inv,
            self.neighbor_map,
            self.doc_ids,
            min_candidates=1,
            meet_index=self.meet_index,
        )
        route_lex = route_query_candidates(
            ["phone", "technical"],
            self.registry,
            self.index,
            self.inv,
            self.neighbor_map,
            self.doc_ids,
            min_candidates=1,
            meet_index=self.meet_index,
            union_lexical=True,
        )
        self.assertLessEqual(route_default.n_merged, route_lex.n_merged)

    def test_gate_report_splits_recall(self):
        queries = {"q1": "phone technical software"}
        qrels = {"q1": {"d0": 1, "d2": 1}}
        report = verify_bit04_gate(
            self.registry,
            queries,
            qrels,
            self.doc_ids,
            self.doc_tokens,
            self.sigs,
            self.inv,
            self.neighbor_map,
            index=self.index,
            meet_index=self.meet_index,
            min_candidates=1,
            target=0.5,
        )
        self.assertEqual(report.n_gold_pairs_in_corpus, 2)
        self.assertGreaterEqual(report.recall_merged, 0.0)

    def test_gold_recall_manual_queries(self):
        """Phone query should capture phone docs in routed set."""
        queries = {
            "q1": "phone technical software",
            "q2": "apple fruit pie",
        }
        qrels = {
            "q1": {"d0": 1, "d2": 1},
            "q2": {"d3": 1, "d4": 1},
        }
        ok, avg, failures = verify_bit04_gate_legacy_tuple(
            self.registry,
            queries,
            qrels,
            self.doc_ids,
            self.doc_tokens,
            self.sigs,
            self.inv,
            self.neighbor_map,
            index=self.index,
            meet_index=self.meet_index,
            min_candidates=1,
            target=0.5,
        )
        self.assertGreater(avg, 0.0)
        self.assertTrue(ok, failures)

    def test_fallback_when_attractor_empty(self):
        route = route_query_candidates(
            ["zzzznonexistent"],
            self.registry,
            self.index,
            self.inv,
            self.neighbor_map,
            self.doc_ids,
            min_candidates=100,
            meet_index=self.meet_index,
        )
        self.assertEqual(route.tier, "bit4_fallback")

    def test_triple_kappa_isolated_routing(self):
        from aethos_physics import SpacetimeCell
        from pipeline.bit_02_attractor_key import kappa_from_cell

        cell = SpacetimeCell.promote_witness((3, 5, 7), (3, 5))
        key = kappa_from_cell(cell)
        idx = build_attractor_index_from_hub_signatures(self.registry, {})
        idx.add("triple_doc", key, "witness")
        cands, _ = candidates_from_attractors(
            ["nonword"],
            self.registry,
            idx,
            radius=0,
        )
        self.assertEqual(cands, [])


if __name__ == "__main__":
    unittest.main()
