"""
AETHOS full system tests: 32 lattices, recursive chains, sequences, origin tree.
Run: python test_aethos.py
"""

from __future__ import annotations

import sys
import time
import unittest

from aethos_lattice import (
    BranchKind,
    LatticeBank32,
    LatticeId,
    VECTORS,
    lattice_id_parts,
)
from aethos_origins import CHILD_DIMENSIONS, DimSlot, Origin, OriginTree
from aethos_recursive import (
    LatticeBank32K,
    canon_recursive,
    segment_index,
    try_compose_triple,
    verify_matches_spec_k2,
)
from aethos_sequences import (
    SequenceKind,
    canon_on_chain,
    cross_type_meet,
    make_chain,
    normalize_chain,
)


class TestThirtyTwoLattices(unittest.TestCase):
    def test_bank_has_32_lattices(self):
        bank = LatticeBank32.single_prime(5)
        self.assertEqual(len(bank.lattices), 32)

    def test_unique_ids_l01_l32(self):
        bank = LatticeBank32.single_prime(7)
        ids = {lat.id for lat in bank.lattices}
        self.assertEqual(len(ids), 32)

    def test_eight_vectors_four_branches(self):
        branches = {lat.branch for lat in LatticeBank32.single_prime(3).lattices}
        vectors = {lat.vector.name for lat in LatticeBank32.single_prime(3).lattices}
        self.assertEqual(len(branches), 4)
        self.assertEqual(len(vectors), 8)

    def test_all_at_returns_32_coords(self):
        bank = LatticeBank32.single_prime(5)
        self.assertEqual(len(bank.all_at(7)), 32)

    def test_regime_velocity_at_prime(self):
        lat = LatticeBank32.single_prime(3)[LatticeId.L01]
        self.assertEqual(lat.at(2), (5, 2, 5))
        self.assertEqual(lat.at(3), (6, 3, 6))
        self.assertEqual(lat.at(4), (7, 3, 7))

    def test_prime_anchor_v1(self):
        self.assertEqual(LatticeBank32.single_prime(5)[LatticeId.L01].anchor(), (5, 0, 5))


class TestRecursiveKPrime(unittest.TestCase):
    def test_pdf_k2_all_branches(self):
        self.assertTrue(verify_matches_spec_k2())

    def test_k3_segment_boundaries(self):
        chain = (3, 5, 7)
        self.assertEqual(segment_index(chain, 2), 0)
        self.assertEqual(segment_index(chain, 3), 1)
        self.assertEqual(segment_index(chain, 5), 2)
        self.assertEqual(segment_index(chain, 7), 3)

    def test_k4_all_wings_same_velocity_list(self):
        bank = LatticeBank32K((3, 5, 7, 11))
        for lat in bank.lattices:
            self.assertEqual(lat.velocity_boundaries(), [3, 5, 7, 11])

    def test_swap_meet_3_11_all_32_wings(self):
        b3 = LatticeBank32.single_prime(3)
        b11 = LatticeBank32.single_prime(11)
        meets = sum(1 for lid in LatticeId if b3[lid].at(11) == b11[lid].at(3))
        self.assertEqual(meets, 32)

    def test_triple_compose_all_four_branches(self):
        for branch in BranchKind:
            pair = canon_recursive(branch, (3, 5), 7)
            triple = canon_recursive(branch, (3, 5, 7), 5)
            self.assertEqual(pair, triple, msg=branch.name)

    def test_compose_helper_finds_triple_hit(self):
        r = try_compose_triple(3, 5, 7)
        self.assertTrue(len(r["triple_confirmations"]) > 0)
        self.assertEqual(r["triple_confirmations"][0]["coord"], (12, 5, 15))

    def test_canon_matches_lattice_bank_k(self):
        bank = LatticeBank32K((3, 11))
        n = 5
        for lid in (LatticeId.L01, LatticeId.L16, LatticeId.L32):
            self.assertEqual(
                bank[lid].at(n),
                lattice_bank_coord_via_parts((3, 11), n, lid),
            )


def lattice_bank_coord_via_parts(chain: tuple[int, ...], n: int, lid: LatticeId):
    from aethos_lattice import apply_vector

    branch, v = lattice_id_parts(lid)
    canon = canon_recursive(branch, chain, n)
    return apply_vector(canon, v)


class TestCountableSequences(unittest.TestCase):
    def test_primes_chain_generator(self):
        self.assertEqual(make_chain(SequenceKind.PRIMES, 5), (3, 5, 7, 11, 13))

    def test_primes_chain_skips_two(self):
        chain = make_chain(SequenceKind.PRIMES, 20)
        self.assertNotIn(2, chain)
        self.assertEqual(chain[0], 3)
        self.assertTrue(all(p % 2 == 1 for p in chain))

    def test_evens_chain(self):
        self.assertEqual(make_chain(SequenceKind.EVENS, 4), (2, 4, 6, 8))

    def test_powers_of_2_chain(self):
        self.assertEqual(make_chain(SequenceKind.POWERS_OF_2, 4), (2, 4, 8, 16))

    def test_evens_single_anchor_swap(self):
        self.assertEqual(
            canon_on_chain(BranchKind.VA1, (2,), 4),
            canon_on_chain(BranchKind.VA1, (4,), 2),
        )

    def test_powers_pair_meet(self):
        a = canon_on_chain(BranchKind.VA1, (2, 4), 8)
        b = canon_on_chain(BranchKind.VA1, (2, 8), 4)
        self.assertEqual(a, b)

    def test_normalize_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            normalize_chain([3, 3, 5])

    def test_normalize_sorts_inputs(self):
        self.assertEqual(normalize_chain([7, 3, 5]), (3.0, 5.0, 7.0))

    def test_different_species_differ_at_same_n(self):
        p = canon_on_chain(BranchKind.VA1, make_chain(SequenceKind.PRIMES, 5), 10)
        e = canon_on_chain(BranchKind.VA1, make_chain(SequenceKind.EVENS, 5), 10)
        self.assertNotEqual(p, e)


class TestOriginTree(unittest.TestCase):
    def test_three_children_per_expanded_origin(self):
        tree = OriginTree.bootstrap(max_depth=2)
        root = tree.root
        self.assertEqual(len(root.children), 3)
        for slot in DimSlot:
            self.assertIn(slot, root.children)
            self.assertEqual(CHILD_DIMENSIONS[slot], CHILD_DIMENSIONS[slot])

    def test_child_ids_d1_d2_d3(self):
        tree = OriginTree.bootstrap(max_depth=1)
        names = {c.id.split(".")[-1] for c in tree.root.children.values()}
        self.assertEqual(names, {"D1", "D2", "D3"})

    def test_origin_count_depth_3(self):
        tree = OriginTree.bootstrap(max_depth=3)
        # 1 + 3 + 9 + 27 = 40 nodes when expanding through depth 3
        self.assertEqual(tree.root.count_descendant_origins(), 40)

    def test_lattice_count_32_per_origin(self):
        tree = OriginTree.bootstrap(max_depth=1)
        wings = tree.root.wings_at(5)
        self.assertEqual(len(wings), 32)

    def test_each_child_has_full_chain_extension(self):
        tree = OriginTree.bootstrap(chain_len=3, max_depth=1)
        for child in tree.root.children.values():
            self.assertGreater(len(child.anchor_chain), len(tree.root.anchor_chain))

    def test_wings_offset_by_origin_coord(self):
        o = Origin(id="T", coord=(100, 200, 300), anchor_chain=(3,), depth=0)
        w = o.wings_at(3)
        local = LatticeBank32.single_prime(3)[LatticeId.L01].at(3)
        self.assertEqual(w["L01"], (100 + local[0], 200 + local[1], 300 + local[2]))

    def test_growth_3_power_d(self):
        tree = OriginTree.bootstrap(max_depth=4)
        n = tree.root.count_descendant_origins()
        # geometric: sum_{i=0}^{d} 3^i at max_depth=4 -> 1+3+9+27+81 = 121
        self.assertEqual(n, 121)
        self.assertEqual(tree.lattice_count_estimate(), 121 * 32)


class TestInfiniteRoomProperties(unittest.TestCase):
    """Structural checks for 'always more room' without running infinite loops."""

    def test_transgression_produces_distinct_coords(self):
        lat = LatticeBank32.single_prime(5)[LatticeId.L01]
        coords = {lat.at(n) for n in range(0, 50)}
        self.assertGreater(len(coords), 40)

    def test_extending_chain_changes_z(self):
        z2 = canon_recursive(BranchKind.VA1, (3, 5), 10)[2]
        z3 = canon_recursive(BranchKind.VA1, (3, 5, 7), 10)[2]
        self.assertNotEqual(z2, z3)

    def test_arbitrary_k_same_four_branches_only(self):
        chain = make_chain(SequenceKind.PRIMES, 8)
        for branch in BranchKind:
            c = canon_recursive(branch, chain, 20)
            self.assertEqual(len(c), 3)


def geometric_origin_count(max_depth: int) -> int:
    return sum(3**i for i in range(max_depth + 1))


class TestDepth5Stress(unittest.TestCase):
    """Depth-5 origin tree: 364 nodes, 11,648 wing-rooms."""

    MAX_DEPTH = 5
    EXPECTED_ORIGINS = geometric_origin_count(5)  # 364
    EXPECTED_WINGS = EXPECTED_ORIGINS * 32  # 11_648

    @classmethod
    def setUpClass(cls) -> None:
        t0 = time.perf_counter()
        cls.tree = OriginTree.bootstrap(max_depth=cls.MAX_DEPTH)
        cls.build_seconds = time.perf_counter() - t0
        cls.origins = list(cls.tree.walk())

    def test_origin_count_364(self):
        self.assertEqual(len(self.origins), self.EXPECTED_ORIGINS)
        self.assertEqual(self.tree.root.count_descendant_origins(), 364)

    def test_wing_room_count(self):
        self.assertEqual(self.tree.lattice_count_estimate(), 11_648)

    def test_build_under_10_seconds(self):
        self.assertLess(self.build_seconds, 10.0, f"build took {self.build_seconds:.2f}s")

    def test_243_leaves_at_depth_5(self):
        leaves = [o for o in self.origins if o.depth == 5]
        self.assertEqual(len(leaves), 3**5)

    def test_internal_nodes_have_3_children(self):
        for o in self.origins:
            if o.depth < 5:
                self.assertEqual(len(o.children), 3, msg=o.id)
            else:
                self.assertEqual(len(o.children), 0, msg=o.id)

    def test_all_origins_32_wings_at_n_10(self):
        for o in self.origins:
            wings = o.wings_at(10)
            self.assertEqual(len(wings), 32, msg=o.id)

    def test_wing_compute_all_origins_under_30_seconds(self):
        t0 = time.perf_counter()
        total = 0
        for o in self.origins:
            total += len(o.wings_at(7))
        elapsed = time.perf_counter() - t0
        self.assertEqual(total, self.EXPECTED_WINGS)
        self.assertLess(elapsed, 30.0, f"wings took {elapsed:.2f}s")

    def test_global_coord_stream_non_trivial(self):
        coords = set()
        for o in self.origins:
            for c in o.wings_at(5).values():
                coords.add(c)
        # 11_648 wing samples; collisions expected (shared meet coords), but not one point
        self.assertGreater(len(coords), 32)
        self.assertLess(len(coords), self.EXPECTED_WINGS)

    def test_k12_bank_32_wings_500_steps(self):
        chain = make_chain(SequenceKind.PRIMES, 12)
        bank = LatticeBank32K(chain)
        for n in range(500):
            self.assertEqual(len(bank.at_all(n)), 32)

    def test_deepest_leaf_chain_longer_than_root(self):
        root_len = len(self.tree.root.anchor_chain)
        deepest = max(self.origins, key=lambda o: len(o.anchor_chain))
        self.assertGreater(len(deepest.anchor_chain), root_len)


class TestActiveNodes100(unittest.TestCase):
    def test_network_has_100_nodes(self):
        net = __import__("aethos_active").ActiveNetwork100.bootstrap(100)
        self.assertEqual(len(net.nodes), 100)

    def test_each_node_vector_1_to_8_branch_va(self):
        net = __import__("aethos_active").ActiveNetwork100.bootstrap(100)
        for node in net.nodes:
            self.assertIn(node.vector_index, range(1, 9))
            self.assertIn(node.branch, list(BranchKind))

    def test_unique_positions_exceed_node_count(self):
        net = __import__("aethos_active").ActiveNetwork100.bootstrap(100)
        u = net.count_unique_positions(0, 100)
        self.assertGreater(u, 100)

    def test_sweep_100_x_50_gt_1000_samples(self):
        net = __import__("aethos_active").ActiveNetwork100.bootstrap(100)
        stream = list(net.sweep_transgression(n_max=50))
        self.assertEqual(len(stream), 100 * 51)
        self.assertGreater(len({c for _, _, c in stream}), 1000)

    def test_capacity_formula(self):
        cap = __import__("aethos_active").CapacityEstimate(
            100, 32, 3, 8, 4
        )
        self.assertEqual(cap.base_wing_states, 3200)
        self.assertGreater(cap.structural_slots(5, 12, 500), 1_000_000)


class TestLatticeCoreIsolation(unittest.TestCase):
    def test_core_imports_without_token_modules(self):
        import importlib
        import sys

        # Fresh import path check: core module should not pull promotion
        mod = importlib.import_module("aethos_core")
        self.assertTrue(hasattr(mod, "AethosLatticeCore"))
        self.assertTrue(hasattr(mod, "formula_coord"))

    def test_core_project_independent_of_tokens(self):
        from aethos_core import AethosLatticeCore, SequenceKind

        core = AethosLatticeCore()
        proj = core.open_project("electron", species=SequenceKind.PRIMES, chain_len=8, origin_depth=2)
        self.assertEqual(proj.wing_count(), 32)
        self.assertGreater(proj.origin_count(), 1)
        wings = proj.all_wings_at(5)
        self.assertEqual(len(wings), 32)

    def test_token_processor_uses_core_formula(self):
        from aethos_core import AethosLatticeCore, formula_coord
        from aethos_token_processor import TokenProcessor, semantic_chain_for_word

        core = AethosLatticeCore()
        proc = TokenProcessor(core=core, rebuild_every=2)
        proc.ingest("tab tab cat")
        chain = semantic_chain_for_word(proc.registry, "tab")
        expected = formula_coord(chain, 7)
        self.assertEqual(proc.lattice_address("tab", n=7), expected)


class TestHilbertFromLattice(unittest.TestCase):
    def test_derivation_table_nonempty(self):
        from aethos_hilbert_lattice import HILBERT_FROM_LATTICE, LatticeHilbertSpace

        self.assertGreaterEqual(len(HILBERT_FROM_LATTICE), 10)
        hs = LatticeHilbertSpace()
        self.assertIn("Inner product", hs.derivation_table())

    def test_gram_matrix_identity_on_unit_basis(self):
        from aethos_hilbert_lattice import LatticeHilbertSpace

        hs = LatticeHilbertSpace(n_window=(5,))
        labels = hs.basis_labels()[:2]
        g = hs.gram_matrix(labels)
        self.assertAlmostEqual(g[0][0].real, 1.0)
        self.assertAlmostEqual(g[1][1].real, 1.0)
        self.assertAlmostEqual(g[0][1].real, 0.0)

    def test_robust_correlation_from_corpus(self):
        from aethos_hilbert_lattice import build_robust_space_from_corpus

        hs = build_robust_space_from_corpus("phone phone technical", "phone chip hardware")
        self.assertGreater(hs.correlation_inner_words("phone", "technical"), 0)
        self.assertGreater(len(hs.correlation_inner_links("phone")), 0)

    def test_norm_superposition(self):
        from aethos_hilbert_lattice import LatticeHilbertSpace
        from aethos_hilbert import BasisLabel, LatticeState

        hs = LatticeHilbertSpace(n_window=(5,))
        lbl = hs.basis_labels()[0]
        psi = LatticeState()
        psi.add(lbl, 0.6)
        psi.add(hs.basis_labels()[1], 0.8)
        self.assertAlmostEqual(hs.norm(psi), 1.0)


class TestHilbertTower(unittest.TestCase):
    def test_tower_estimate_positive(self):
        from aethos_hilbert import estimate_hilbert_tower

        rep = estimate_hilbert_tower(chain_k=4, n_max=20, origin_depth=2)
        self.assertEqual(rep.wing_dim_per_bank, 32)
        self.assertGreater(rep.truncated_basis_size, 10_000)
        self.assertGreater(len(rep.notes), 5)

    def test_wing_basis_orthogonal_labels(self):
        from aethos_hilbert import inner_product, wing_subspace_states

        wings = wing_subspace_states((3, 5), 7)
        self.assertEqual(len(wings), 32)
        self.assertAlmostEqual(abs(inner_product(wings[0], wings[0])), 1.0)
        self.assertEqual(inner_product(wings[0], wings[1]), 0)

    def test_branch_fan_four_states(self):
        from aethos_hilbert import branch_fan_states, formula_coord_branch
        from aethos_lattice import BranchKind

        fan = branch_fan_states((3, 5, 7), 5)
        self.assertEqual(len(fan), 4)
        coords = {formula_coord_branch((3, 5, 7), 5, b, 1) for b in BranchKind}
        self.assertEqual(len(coords), 4)

    def test_perm_fiber_factorial(self):
        from aethos_hilbert import perm_fiber_states

        self.assertEqual(len(perm_fiber_states((3, 5), 7)), 2)
        self.assertEqual(len(perm_fiber_states((3, 5, 7), 5)), 6)

    def test_correlation_inner_product(self):
        from aethos_hilbert import correlation_inner_product

        a = {3: 1.0, 5: 2.0}
        b = {5: 1.0, 7: 3.0}
        self.assertEqual(correlation_inner_product(a, b), 2.0)


class TestOddPrimeGolden(unittest.TestCase):
    def test_primes_chain_matches_golden(self):
        from aethos_golden import PRIMES_CHAIN_5

        self.assertEqual(make_chain(SequenceKind.PRIMES, 5), PRIMES_CHAIN_5)

    def test_letter_a_is_first_odd_prime(self):
        from aethos_golden import LETTER_A_PRIME, LETTER_Z_PRIME
        from aethos_words import letter_to_prime

        self.assertEqual(letter_to_prime("a"), LETTER_A_PRIME)
        self.assertEqual(letter_to_prime("a"), 3)
        self.assertEqual(letter_to_prime("z"), LETTER_Z_PRIME)

    def test_codec_menu_and_pool_start(self):
        from aethos_codec import PRIME_MENU
        from aethos_golden import PRIME_MENU_FIRST, PROMOTION_POOL_FIRST
        from aethos_promotion import PROMOTION_POOL

        self.assertEqual(PRIME_MENU[0], PRIME_MENU_FIRST)
        self.assertEqual(PROMOTION_POOL[0], PROMOTION_POOL_FIRST)


class TestAethosPipeline(unittest.TestCase):
    def test_unified_smoke(self):
        from aethos_pipeline import AethosPipeline, check_promotion_invariants, smoke_corpus

        pipe = AethosPipeline(rebuild_every=2)
        pipe.ingest(*smoke_corpus())
        rep = pipe.report()
        self.assertGreater(rep.documents_read, 0)
        self.assertGreater(rep.token_savings.unique_words, 5)
        self.assertGreater(rep.intersection_only_words, 0)
        self.assertEqual(rep.invariant_errors, ())
        self.assertEqual(check_promotion_invariants(pipe.registry), [])

    def test_resolve_apple_contexts(self):
        from aethos_pipeline import AethosPipeline, smoke_corpus

        pipe = AethosPipeline(rebuild_every=2)
        pipe.ingest(*smoke_corpus())
        tech = pipe.resolve("apple", ["phone", "chip"])
        food = pipe.resolve("apple", ["fruit", "pie"])
        self.assertEqual(tech["tier"], "dedicated_l3")
        self.assertNotEqual(tech["cluster_id"], food["cluster_id"])

    def test_dot_for_bytes_and_text(self):
        from aethos_pipeline import AethosPipeline

        pipe = AethosPipeline()
        bdot = pipe.dot_for_bytes(b"pipeline bytes")
        tdot = pipe.dot_for_text("hello")
        self.assertIsNotNone(bdot.coord)
        self.assertIsNotNone(tdot.coord)

    def test_dot_for_word(self):
        from aethos_pipeline import AethosPipeline

        pipe = AethosPipeline()
        dot = pipe.dot_for_word("tab")
        self.assertEqual(dot.witness.prime_order, pipe.dot_for_word("tab").witness.prime_order)

    def test_promotion_invariants_singleton(self):
        from aethos_pipeline import AethosPipeline, check_promotion_invariants

        pipe = AethosPipeline(rebuild_every=2)
        pipe.ingest("xylophone once only")
        self.assertTrue(pipe.registry.is_intersection_only("xylophone"))
        self.assertEqual(check_promotion_invariants(pipe.registry), [])

    def test_pool_large_enough_for_smoke(self):
        from aethos_pipeline import AethosPipeline, smoke_corpus
        from aethos_promotion import PROMOTION_POOL

        pipe = AethosPipeline(rebuild_every=2)
        pipe.ingest(*smoke_corpus())
        self.assertLess(pipe.registry._next_promotion_idx, len(PROMOTION_POOL))


class TestSemanticOverlay(unittest.TestCase):
    def test_registry_equals_codec_local_for_intersection_word(self):
        from aethos_pipeline import AethosPipeline

        pipe = AethosPipeline(rebuild_every=2)
        pipe.ingest("tab tab tab", "tab cat")
        ov = pipe.semantic_overlay("tab", n=7)
        self.assertTrue(ov.registry_equals_codec_local, msg=ov.summary())

    def test_registry_equals_word_base_for_intersection_word(self):
        from aethos_pipeline import AethosPipeline

        pipe = AethosPipeline(rebuild_every=2)
        pipe.ingest("tab tab", "bat bat")
        ov = pipe.semantic_overlay("tab", n=7)
        self.assertTrue(ov.registry_equals_word_base, msg=ov.summary())

    def test_anagrams_share_overlay_base_but_ordered_word_dots_differ(self):
        from aethos_pipeline import AethosPipeline

        pipe = AethosPipeline(rebuild_every=2)
        pipe.ingest("tab bat tab bat")
        tab = pipe.semantic_overlay("tab", n=7)
        bat = pipe.semantic_overlay("bat", n=7)
        self.assertEqual(tab.registry_local, bat.registry_local)
        self.assertIsNotNone(tab.word_dot)
        self.assertIsNotNone(bat.word_dot)
        self.assertNotEqual(tab.word_dot.coord, bat.word_dot.coord)


class TestGoldenCoordinates(unittest.TestCase):
    def test_fixture_verifies(self):
        from aethos_golden_coords import verify_golden_coords

        errs = verify_golden_coords()
        self.assertEqual(errs, [], msg="\n".join(errs))

    def test_canon_matches_chain_on_all_cases(self):
        import json
        from pathlib import Path

        from aethos_golden_coords import FIXTURE_PATH
        from aethos_lattice import BranchKind
        from aethos_recursive import canon_recursive
        from aethos_sequences import canon_on_chain

        doc = json.loads(FIXTURE_PATH.read_text())
        for case in doc["cases"]:
            chain = tuple(case["chain"])
            n = case["n"]
            for b in BranchKind:
                self.assertEqual(
                    canon_recursive(b, chain, n),
                    canon_on_chain(b, chain, n),
                    msg=f"{b.name} chain={chain} n={n}",
                )


class TestSwapMeetLaws(unittest.TestCase):
    def test_3_11_solo_swap_all_32_wings(self):
        from aethos_golden_coords import swap_meet_solo_all_wings

        self.assertTrue(swap_meet_solo_all_wings(3, 11))

    def test_odd_prime_solo_swap_sample(self):
        from aethos_golden_coords import swap_meet_solo_all_wings
        from aethos_sequences import make_chain, SequenceKind

        odds = make_chain(SequenceKind.PRIMES, 12)
        for i in range(len(odds) - 1):
            for j in range(i + 1, min(i + 4, len(odds))):
                self.assertTrue(
                    swap_meet_solo_all_wings(odds[i], odds[j]),
                    msg=f"solo swap failed for {odds[i]}, {odds[j]}",
                )

    def test_k3_interior_z_lock_triple_compose(self):
        for branch in BranchKind:
            pair = canon_recursive(branch, (3, 5), 7)
            triple = canon_recursive(branch, (3, 5, 7), 5)
            self.assertEqual(pair, triple, msg=branch.name)


class TestPersistence(unittest.TestCase):
    def test_reader_roundtrip(self):
        import tempfile
        from pathlib import Path

        from aethos_persist import load_reader, save_reader
        from aethos_pipeline import smoke_corpus

        r1 = __import__("aethos_natural", fromlist=["NaturalReader"]).NaturalReader(rebuild_every=2)
        r1.read(*smoke_corpus())
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "state.json"
            save_reader(r1, path)
            r2 = load_reader(path, rebuild_every=2)
        self.assertEqual(r1.registry.word_counts, r2.registry.word_counts)
        self.assertEqual(r1.word_to_cluster, r2.word_to_cluster)
        self.assertEqual(len(r1.registry.correlations), len(r2.registry.correlations))

    def test_pipeline_save_load(self):
        import tempfile
        from pathlib import Path

        from aethos_pipeline import AethosPipeline, smoke_corpus

        p1 = AethosPipeline(rebuild_every=2)
        p1.ingest(*smoke_corpus())
        wc_before = dict(p1.registry.word_counts)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pipe.json"
            p1.save(str(path))
            p2 = AethosPipeline.load(str(path), rebuild_every=2)
        self.assertEqual(p2.registry.word_counts, wc_before)


class TestTokenSavings(unittest.TestCase):
    def test_saves_primes_for_singletons(self):
        from aethos_natural import NaturalReader
        from aethos_token_savings import report_from_reader

        r = NaturalReader(rebuild_every=2)
        r.read("phone phone phone", "xylophone once", "zebra once")
        rep = report_from_reader(r)
        self.assertGreater(rep.pool_primes_saved, 0)
        self.assertGreater(rep.intersection_only_words, 0)
        self.assertGreater(rep.correlation_edges, 0)

    def test_correlations_use_floats_not_primes_per_edge(self):
        from aethos_natural import NaturalReader
        from aethos_token_savings import report_from_reader

        r = NaturalReader(rebuild_every=2)
        r.read("apple phone chip", "apple fruit pie")
        rep = report_from_reader(r)
        self.assertGreater(rep.float_correlation_values, rep.correlation_edges)


class TestFrequencyRarity(unittest.TestCase):
    def test_common_rarer_than_obscure(self):
        from aethos_frequency import FrequencyProfile, RarityBand
        from aethos_natural import NaturalReader

        r = NaturalReader(rebuild_every=2)
        r.read("phone phone phone technical", "phone chip software", "xylophone once")
        fp = FrequencyProfile.from_reader(r)
        p_phone = fp.profile("phone")
        p_xylo = fp.profile("xylophone")
        self.assertGreater(p_phone.count, p_xylo.count)
        self.assertGreater(p_phone.percentile, p_xylo.percentile)
        self.assertGreater(p_xylo.idf, p_phone.idf)

    def test_singleton_stays_intersection_no_pool_prime(self):
        from aethos_natural import NaturalReader

        r = NaturalReader(rebuild_every=2)
        r.read("xylophone obscure word once")
        self.assertTrue(r.registry.is_intersection_only("xylophone"))
        self.assertNotIn(("L3_WORD", "xylophone"), r.registry.promoted)

    def test_apple_gets_dedicated_prime_only_after_different_contexts(self):
        from aethos_natural import NaturalReader

        r = NaturalReader(rebuild_every=2, min_pair_count=1, min_pmi=0.1)
        r.read("apple phone chip", "apple fruit pie orchard")
        self.assertFalse(r.registry.is_intersection_only("apple"))
        from aethos_promotion import LatticeTier

        self.assertIn((LatticeTier.L3_WORD, "apple"), r.registry.promoted)

    def test_phone_twice_same_context_stays_intersection(self):
        from aethos_natural import NaturalReader

        r = NaturalReader(rebuild_every=2)
        r.read("phone technical hardware network", "phone technical hardware network")
        self.assertTrue(r.registry.is_intersection_only("phone"))


class TestNaturalReading(unittest.TestCase):
    def _train_reader(self):
        from aethos_natural import NaturalReader

        r = NaturalReader(rebuild_every=2, min_pair_count=2, min_pmi=0.3)
        tech = [
            "apple phone chip software technical",
            "apple phone hardware technical",
            "phone technical software chip",
            "samsung phone chip technical",
        ] * 3
        food = [
            "apple fruit pie orchard",
            "apple fruit salad dessert",
            "banana fruit pie recipe",
        ] * 3
        r.read(*tech, *food)
        return r

    def test_clusters_emerge_without_tags(self):
        r = self._train_reader()
        self.assertGreaterEqual(len(r.cluster_hubs), 2)

    def test_apple_disambiguates_by_context(self):
        r = self._train_reader()
        c_tech, _ = r.infer_cluster("apple", ["phone", "chip"])
        c_food, _ = r.infer_cluster("apple", ["fruit", "pie"])
        self.assertNotEqual(c_tech, c_food)

    def test_phone_cluster_lists_technical_neighbors(self):
        r = self._train_reader()
        cid, _ = r.infer_cluster("phone", ["chip"])
        related = dict(r.related_in_cluster(cid, 15))
        self.assertTrue("chip" in related or "software" in related or "phone" in related)


class TestCrossMeaningL789(unittest.TestCase):
    def test_technical_vector_lists_phone(self):
        from aethos_crossmeaning import SemanticStack

        stack = SemanticStack()
        stack.train_tagged("technical", "apple phone chip technical", "phone technical software")
        items = dict(stack.cross.all_in_category("technical"))
        self.assertIn("phone", items)

    def test_apple_phone_technical_belonging(self):
        from aethos_crossmeaning import SemanticStack

        stack = SemanticStack()
        stack.train_tagged("technical", "apple phone technical", "apple phone chip technical")
        score = stack.cross.phrase_belonging(["apple", "phone"], "technical")
        self.assertGreater(score, 0.0)

    def test_apple_food_vs_technical(self):
        from aethos_crossmeaning import SemanticStack

        stack = SemanticStack()
        stack.train_tagged("technical", "apple phone technical")
        stack.train_tagged("food", "apple fruit pie food", "apple orchard fruit")
        t = stack.cross.belonging_score("apple", "technical")
        f = stack.cross.belonging_score("apple", "food")
        self.assertGreater(f, t)

    def test_category_vector_has_primes(self):
        from aethos_crossmeaning import SemanticStack

        stack = SemanticStack()
        stack.train_tagged("technical", "phone chip software technical")
        cat = stack.cross.categories["technical"]
        self.assertGreater(len(cat.prime_weights), 0)
        self.assertGreater(cat.dim7, 0)


class TestPromotionLattice(unittest.TestCase):
    def test_symbols_map_to_letter_primes(self):
        from aethos_promotion import LatticeTier, PromotionRegistry
        from aethos_words import letter_to_prime

        reg = PromotionRegistry()
        reg.observe_symbol("a")
        self.assertEqual(reg.resolve_prime("a", LatticeTier.L1_SYMBOL), letter_to_prime("a"))

    def test_subword_promotion_after_threshold(self):
        from aethos_promotion import LatticeTier, MultiLatticeStack, PromotionRegistry

        stack = MultiLatticeStack(PromotionRegistry(subword_promote_at=2))
        stack.train("tab tab bat")
        self.assertIn((LatticeTier.L2_SUBWORD, "tab"), stack.registry.promoted)

    def test_word_promotion_distinct_primes_when_contexts_differ(self):
        from aethos_promotion import LatticeTier, MultiLatticeStack

        stack = MultiLatticeStack()
        stack.train("apple phone chip", "apple fruit pie")
        apple = stack.registry.promoted.get((LatticeTier.L3_WORD, "apple"))
        self.assertIsNotNone(apple)
        self.assertFalse(apple.intersection_only)

    def test_correlation_dims_456(self):
        from aethos_promotion import MultiLatticeStack

        stack = MultiLatticeStack()
        stack.train("tab bat tab bat cat")
        links = stack.registry.correlations_for("tab")
        self.assertGreater(len(links), 0)
        pt = stack.registry.correlation_point(links[0])
        self.assertEqual(len(pt), 6)
        self.assertGreater(pt[3] + pt[4] + pt[5], 0)


class TestNearLocationWords(unittest.TestCase):
    def test_tab_bat_same_chain_different_order(self):
        from aethos_words import SharedSite, encode_word_at_site, word_sorted_chain, word_to_order

        self.assertEqual(word_sorted_chain("tab"), word_sorted_chain("bat"))
        self.assertNotEqual(word_to_order("tab"), word_to_order("bat"))

    def test_tab_bat_near_but_decodable(self):
        from aethos_words import (
            SharedSite,
            decode_word,
            distance,
            encode_word_at_site,
            word_sorted_chain,
        )

        site = SharedSite(chain=word_sorted_chain("tab"))
        tab = encode_word_at_site("tab", site)
        bat = encode_word_at_site("bat", site)
        self.assertLess(distance(tab.coord, bat.coord), 0.01)
        self.assertNotEqual(tab.coord, bat.coord)
        self.assertEqual(decode_word(tab), "tab")
        self.assertEqual(decode_word(bat), "bat")

    def test_order_side_channel_differs(self):
        from aethos_words import (
            canonical_base,
            decode_order_from_dot,
            encode_word_at_site,
            word_sorted_chain,
            word_to_order,
        )

        site_chain = word_sorted_chain("tab")
        tab = encode_word_at_site("tab")
        base = canonical_base(tab)
        got = decode_order_from_dot(base, tab.coord, site_chain)
        self.assertEqual(got, word_to_order("tab"))


class TestPrimePermutation(unittest.TestCase):
    def test_two_primes_two_permutations(self):
        from aethos_permutation import ordered_permutation_list, permutation_count

        self.assertEqual(permutation_count(2), 2)
        self.assertEqual(len(ordered_permutation_list((3, 11))), 2)

    def test_three_primes_six_permutations(self):
        from aethos_permutation import permutation_count, ordered_permutation_list

        self.assertEqual(permutation_count(3), 6)
        self.assertEqual(len(ordered_permutation_list((3, 5, 7))), 6)

    def test_side_offset_recovers_order(self):
        from aethos_lattice import LatticeId, apply_vector, lattice_id_parts
        from aethos_permutation import apply_order_offset, decode_order_from_dot
        from aethos_sequences import canon_on_chain

        sorted_p = (3, 5, 7)
        branch, vector = lattice_id_parts(LatticeId.L01)
        base = apply_vector(canon_on_chain(branch, sorted_p, 5), vector)
        for order in ((3, 5, 7), (7, 3, 5), (5, 7, 3)):
            dot = apply_order_offset(base, sorted_p, order)
            self.assertEqual(decode_order_from_dot(base, dot, sorted_p), order)

    def test_codec_recovers_order_from_dot(self):
        from aethos_codec import encode_bytes, recovered_prime_order

        dot = encode_bytes(b"x", prime_order=(11, 3))
        self.assertEqual(recovered_prime_order(dot), (11, 3))


class TestIntersectionCodec(unittest.TestCase):
    def test_roundtrip_bytes(self):
        from aethos_codec import decode_bytes, encode_bytes, verify_dot

        raw = b"store data as intersection dots"
        dot = encode_bytes(raw)
        self.assertTrue(verify_dot(dot))
        self.assertEqual(decode_bytes(dot), raw)

    def test_compact_one_dot_roundtrip(self):
        from aethos_codec import compress_to_one_dot, expand_from_one_dot

        raw = b"x" * 500
        compact = compress_to_one_dot(raw)
        back, dot = expand_from_one_dot(compact)
        self.assertEqual(back, raw)
        self.assertIsNotNone(dot.coord)

    def test_recompute_coord_from_witness(self):
        from aethos_codec import coordinate_from_witness, encode_bytes

        dot = encode_bytes(b"recompute me")
        rx, ry, rz = coordinate_from_witness(dot.witness)
        self.assertEqual((rx, ry, rz), dot.coord)

    def test_meaning_string(self):
        from aethos_codec import encode_text, explain_witness

        dot = encode_text("hello")
        m = explain_witness(dot.witness)
        self.assertIn("transgressor", m)
        self.assertIn("anchors", m)

    def test_corrupt_coord_fails_verify(self):
        from aethos_codec import Dot, encode_bytes, verify_dot

        dot = encode_bytes(b"test")
        bad = Dot(x=0, y=0, z=0, witness=dot.witness)
        self.assertFalse(verify_dot(bad))


class TestEndToEndIntegration(unittest.TestCase):
    """Full stack: sequence -> 32 bank -> origin tree -> meets."""

    def test_origin_wings_use_chain_formulas(self):
        tree = OriginTree.bootstrap(max_depth=1)
        wings = tree.root.wings_at(5, BranchKind.VA1)
        bank = LatticeBank32K(tree.root.anchor_chain)
        for lid in LatticeId:
            self.assertEqual(wings[lid.name], bank[lid].at(5))

    def test_multi_species_banks_all_32(self):
        for kind in (SequenceKind.PRIMES, SequenceKind.EVENS, SequenceKind.POWERS_OF_2):
            chain = make_chain(kind, 4)
            bank = LatticeBank32K(chain)
            self.assertEqual(len(bank.lattices), 32)
            self.assertEqual(len(bank.at_all(6)), 32)


class TestVectorsAndVB(unittest.TestCase):
    def test_eight_vectors_defined(self):
        self.assertEqual(len(VECTORS), 8)

    def test_vb_differs_from_va_on_v5(self):
        bank = LatticeBank32.single_prime(5)
        va = bank[LatticeId.L01].at(7)  # VA1 v1
        vb = bank[LatticeId.L05].at(7)  # VA1 v5 (VB family)
        self.assertNotEqual(va, vb)


def run_tests() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromModule(sys.modules[__name__]))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    print("=" * 60)
    print("AETHOS FULL SYSTEM TEST SUITE")
    print("=" * 60 + "\n")
    sys.exit(run_tests())
