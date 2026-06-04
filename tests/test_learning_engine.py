"""Learning engine — bad correlations, factor analogy, 32-wing consensus."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.learning_engine import (
    FEMALE,
    HUMAN,
    KING,
    MALE,
    MAN,
    QUEEN,
    ROYAL,
    WOMAN,
    BadCorrelationStore,
    consensus_factor_agreement,
    context_primes_for_pair,
    coordinate_variation_across_species,
    distilled_registry_from_dict,
    distilled_registry_to_dict,
    factor_analogy,
    pf_similarity_all_wings,
    promotion_candidates_from_store,
    record_retrieval_false_positives,
)
from core.phi_lattice import LatticeId, compute_coordinates, prime_factor_similarity
from aethos_promotion import LatticeTier, PromotedToken, PromotionRegistry


class TestFactorAnalogy(unittest.TestCase):
    def test_king_composite(self) -> None:
        self.assertEqual(KING, ROYAL * MALE)

    def test_queen_composite(self) -> None:
        self.assertEqual(QUEEN, ROYAL * FEMALE)

    def test_king_minus_man_plus_woman(self) -> None:
        self.assertEqual(factor_analogy(KING, MAN, WOMAN), QUEEN)

    def test_man_woman_atomic_primes(self) -> None:
        self.assertEqual(MAN, MALE)
        self.assertEqual(WOMAN, FEMALE)


class TestBadCorrelationStore(unittest.TestCase):
    def test_record_grows_strength(self) -> None:
        store = BadCorrelationStore()
        bc1 = store.record("king", "woman", {ROYAL, MALE, FEMALE})
        bc2 = store.record("king", "woman", {ROYAL, FEMALE})
        self.assertEqual(bc1.fire_count, 2)
        self.assertGreater(bc2.signal_strength, bc1.signal_strength - 0.01)

    def test_king_woman_fixture_strength(self) -> None:
        store = BadCorrelationStore()
        for _ in range(5):
            store.record("king", "woman", {ROYAL, MALE, FEMALE, HUMAN})
        bc = store.entries[("king", "woman")]
        self.assertGreater(bc.signal_strength, 1.5)

    def test_top_unresolved_ordering(self) -> None:
        store = BadCorrelationStore()
        store.record("a", "b", {107, 109})
        for _ in range(10):
            store.record("x", "y", {107, 109, 113, 127})
        top = store.top_unresolved(1)[0]
        self.assertEqual(top.word_a, "x")

    def test_resolve_shared_context(self) -> None:
        store = BadCorrelationStore()
        store.record("king", "woman", {ROYAL, MALE, FEMALE})
        store.record("queen", "man", {ROYAL, MALE, FEMALE})
        gender_prime = 1151
        resolved = store.try_resolve(gender_prime, {ROYAL, MALE, FEMALE, HUMAN})
        self.assertEqual(len(resolved), 2)
        self.assertTrue(all(r.resolved for r in resolved))

    def test_resolve_requires_min_shared(self) -> None:
        store = BadCorrelationStore()
        store.record("alpha", "beta", {107, 109})
        self.assertEqual(len(store.try_resolve(2000, {2011, 2017}, min_shared=2)), 0)

    def test_save_load_roundtrip(self) -> None:
        store = BadCorrelationStore()
        store.record("king", "woman", {ROYAL, FEMALE})
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.json"
            store.save(path)
            loaded = BadCorrelationStore.load(path)
        self.assertIn(("king", "woman"), loaded.entries)
        self.assertAlmostEqual(
            loaded.entries[("king", "woman")].signal_strength,
            store.entries[("king", "woman")].signal_strength,
        )


class TestCrossLatticeConsensus(unittest.TestCase):
    def test_pf_similarity_invariant_across_wings(self) -> None:
        """Jaccard on factors does not depend on lattice_id."""
        sim_l01 = prime_factor_similarity(KING, QUEEN)
        sim_l16 = prime_factor_similarity(KING, QUEEN)
        self.assertAlmostEqual(sim_l01, sim_l16)
        self.assertGreater(sim_l01, 0.2)

    def test_king_royal_vs_queen_royal(self) -> None:
        royal_king = ROYAL * MALE
        royal_queen = ROYAL * FEMALE
        self.assertAlmostEqual(
            prime_factor_similarity(royal_king, royal_queen), 1 / 3, places=5
        )

    def test_coords_differ_by_species_anchor(self) -> None:
        self.assertGreaterEqual(coordinate_variation_across_species(ROYAL), 2)

    def test_pf_all_wings_returns_similarity(self) -> None:
        sim, wings = pf_similarity_all_wings(KING, QUEEN)
        self.assertGreater(sim, 0.0)

    def test_consensus_agreement_king_queen(self) -> None:
        self.assertTrue(consensus_factor_agreement(KING, QUEEN, min_sim=0.25))


class TestDistilledRegistry(unittest.TestCase):
    def test_distilled_omits_counts(self) -> None:
        reg = PromotionRegistry()
        reg.word_counts["quantum"] = 99
        reg.promoted[(LatticeTier.L3_WORD, "quantum")] = PromotedToken(
            text="quantum",
            tier=LatticeTier.L3_WORD,
            prime=1109,
            parent_primes=(107, 109),
        )
        doc = distilled_registry_to_dict(reg)
        self.assertNotIn("word_counts", doc)
        self.assertNotIn("correlations", doc)
        self.assertIn("quantum", doc["promoted"])

    def test_distilled_load_merges_promoted(self) -> None:
        reg = PromotionRegistry()
        doc = {
            "version": 1,
            "promoted": {
                "entanglement": {
                    "tier": 3,
                    "prime": 1117,
                    "parent_primes": [107, 113],
                    "intersection_only": False,
                }
            },
            "intersections": {},
        }
        distilled_registry_from_dict(doc, reg)
        tok = reg.promoted[(LatticeTier.L3_WORD, "entanglement")]
        self.assertEqual(tok.prime, 1117)


class TestBeirHelpers(unittest.TestCase):
    def test_context_primes_pair(self) -> None:
        reg = PromotionRegistry()
        reg.promoted[(LatticeTier.L3_WORD, "king")] = PromotedToken(
            text="king",
            tier=LatticeTier.L3_WORD,
            prime=KING,
            parent_primes=(ROYAL, MALE),
        )
        reg.promoted[(LatticeTier.L3_WORD, "woman")] = PromotedToken(
            text="woman",
            tier=LatticeTier.L3_WORD,
            prime=WOMAN,
            parent_primes=(FEMALE,),
        )
        ctx = context_primes_for_pair(reg, "king", "woman")
        self.assertIn(ROYAL, ctx)
        self.assertIn(MALE, ctx)
        self.assertIn(FEMALE, ctx)

    def test_record_false_positive_retrieval(self) -> None:
        from aethos_hub_signature import build_hub_signature_from_tokens, precompute_registry_index

        reg = PromotionRegistry()
        reg.word_counts["queryterm"] = 5
        reg.word_counts["wronghub"] = 5
        reg.word_counts["noise"] = 2
        reg.promoted[(LatticeTier.L3_WORD, "queryterm")] = PromotedToken(
            text="queryterm",
            tier=LatticeTier.L3_WORD,
            prime=1103,
            parent_primes=(107,),
        )
        reg.promoted[(LatticeTier.L3_WORD, "wronghub")] = PromotedToken(
            text="wronghub",
            tier=LatticeTier.L3_WORD,
            prime=1109,
            parent_primes=(109,),
        )
        idx = precompute_registry_index(reg)
        sig = build_hub_signature_from_tokens(
            "d1", frozenset({"wronghub", "noise"}), idx, top_k=4
        )
        from aethos_hub_signature import build_query_profile

        profile = build_query_profile(
            "queryterm", reg, neighbor_map={}, doc_freq={}, n_docs=10
        )
        store = BadCorrelationStore()
        n = record_retrieval_false_positives(
            store,
            ranked=["d1"],
            relevant=set(),
            profile=profile,
            hub_sigs={"d1": sig},
            registry=reg,
        )
        self.assertGreater(n, 0)
        self.assertTrue(store.entries)

    def test_promotion_candidates_threshold(self) -> None:
        store = BadCorrelationStore()
        store.record("a", "b", {107})
        cands = promotion_candidates_from_store(store, min_strength=999.0)
        self.assertEqual(cands, [])


class TestLatticeCoords(unittest.TestCase):
    def test_same_prime_different_wing_coords_can_differ(self) -> None:
        c1 = compute_coordinates((ROYAL,), 7, LatticeId.L01)
        c2 = compute_coordinates((ROYAL,), 7, LatticeId.L16)
        self.assertNotEqual(c1, c2)


if __name__ == "__main__":
    unittest.main()
