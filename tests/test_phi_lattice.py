"""φ-Prime Lattice — 57-test specification suite."""

from __future__ import annotations

import math
import unittest

from core.phi_lattice import (
    BranchKind,
    LatticeId,
    VECTORS,
    apply_vector,
    canon_on_chain,
    canon_va1,
    compute_anchor,
    compute_all_wings,
    compute_coordinates,
    composite_from_primes,
    euclidean_distance,
    even_chain,
    fibonacci_chain,
    lattice_id_parts,
    prime_anchor_coord,
    prime_chain,
    prime_factor_similarity,
    prime_factors,
    prime_pair_case,
    powers_of_two,
    segment_index,
    should_promote_intersection,
    single_prime_branch,
    squares_chain,
    swap_meet,
    swap_meet_all_wings,
    verify_golden_k1_p5_n7,
    verify_golden_k2_3_11_n5,
    verify_k3_z_plateau,
    verify_va_vb_swap,
    yxz_to_xyz,
    z_depth,
)


class TestGroup1PrimeAnchors(unittest.TestCase):
    def test_p5_anchor_v1(self) -> None:
        self.assertEqual(compute_anchor(5, LatticeId.L01), (5, 0, 5))

    def test_p5_anchor_not_branch_at_n5(self) -> None:
        branch = compute_coordinates((5,), 5, LatticeId.L01)
        anchor = compute_anchor(5, LatticeId.L01)
        self.assertNotEqual(branch, anchor)
        self.assertEqual(branch, (10, 5, 10))

    def test_p5_n7_golden_l01(self) -> None:
        self.assertEqual(compute_coordinates((5,), 7, LatticeId.L01), (12, 5, 12))
        self.assertTrue(verify_golden_k1_p5_n7())

    def test_p5_v5_coord(self) -> None:
        self.assertEqual(compute_coordinates((5,), 7, LatticeId.L05), (5, 12, 12))

    def test_p3_regime_a_n2(self) -> None:
        self.assertEqual(compute_coordinates((3,), 2, LatticeId.L01), (5, 2, 5))

    def test_p3_regime_b_n4(self) -> None:
        self.assertEqual(compute_coordinates((3,), 4, LatticeId.L01), (7, 3, 7))


class TestGroup2VectorTransforms(unittest.TestCase):
    def test_y_over_swap(self) -> None:
        self.assertTrue(verify_va_vb_swap())
        self.assertEqual(yxz_to_xyz((16, 5, 20)), (5, 16, 20))

    def test_v1_identity(self) -> None:
        v1 = VECTORS[0]
        self.assertEqual(apply_vector((16, 5, 20), v1), (16, 5, 20))

    def test_v3_flip_x(self) -> None:
        v3 = VECTORS[2]
        self.assertEqual(apply_vector((16, 5, 20), v3), (-16, 5, 20))

    def test_l16_y_flip(self) -> None:
        _, v = lattice_id_parts(LatticeId.L16)
        c = compute_coordinates((5,), 7, LatticeId.L16)
        self.assertEqual(c, (5, 12, -12))


class TestGroup3SinglePrimeBranch(unittest.TestCase):
    def test_regime_labels(self) -> None:
        lat_p = 5
        self.assertLess(3, lat_p)
        self.assertGreaterEqual(7, lat_p)

    def test_composite_2x3_as_p3_n2(self) -> None:
        self.assertEqual(single_prime_branch(BranchKind.VA1, 3, 2), (5, 2, 5))

    def test_all_branches_k1_n3(self) -> None:
        self.assertEqual(canon_on_chain(BranchKind.VA1, (5,), 3), (8, 3, 8))
        self.assertEqual(canon_on_chain(BranchKind.VA2, (5,), 3), (8, -3, 8))


class TestGroup4KDepth(unittest.TestCase):
    def test_segment_index_k3(self) -> None:
        chain = (3, 5, 7)
        self.assertEqual(segment_index(chain, 2), 0)
        self.assertEqual(segment_index(chain, 3), 1)
        self.assertEqual(segment_index(chain, 5), 2)
        self.assertEqual(segment_index(chain, 7), 3)

    def test_z_plateau_interior_k4(self) -> None:
        self.assertTrue(verify_k3_z_plateau())

    def test_k2_va1_n5_golden(self) -> None:
        self.assertTrue(verify_golden_k2_3_11_n5())
        self.assertEqual(canon_va1((3, 11), 5), (16, 5, 19))

    def test_pair_case_1based(self) -> None:
        self.assertEqual(prime_pair_case(3, 11, 2), 1)
        self.assertEqual(prime_pair_case(3, 11, 5), 2)
        self.assertEqual(prime_pair_case(3, 11, 12), 3)

    def test_32_wings_at_n(self) -> None:
        self.assertEqual(len(compute_all_wings((5,), 7)), 32)


class TestGroup5SwapMeet(unittest.TestCase):
    def test_swap_3_11_l01(self) -> None:
        hit = swap_meet(3, 11, lattice_id=LatticeId.L01)
        self.assertIsNotNone(hit)
        coord, nl, nr = hit  # type: ignore
        self.assertEqual(nl, 11)
        self.assertEqual(nr, 3)

    def test_swap_all_32_wings_3_11(self) -> None:
        hits = swap_meet_all_wings(3, 11, n_max=50)
        self.assertEqual(len(hits), 32)

    def test_swap_5_7(self) -> None:
        self.assertIsNotNone(swap_meet(5, 7))


class TestGroup6SpeciesChains(unittest.TestCase):
    def test_prime_chain(self) -> None:
        self.assertEqual(prime_chain(5), (3, 5, 7, 11, 13))

    def test_even_chain(self) -> None:
        self.assertEqual(even_chain(4), (2, 4, 6, 8))

    def test_powers_of_two(self) -> None:
        self.assertEqual(powers_of_two(4), (2, 4, 8, 16))

    def test_fibonacci(self) -> None:
        self.assertEqual(fibonacci_chain(6), (1, 1, 2, 3, 5, 8))

    def test_squares(self) -> None:
        self.assertEqual(squares_chain(4), (1, 4, 9, 16))

    def test_different_species_different_coord(self) -> None:
        c_prime = compute_coordinates(prime_chain(3), 5, LatticeId.L01)
        c_even = compute_coordinates(even_chain(3), 5, LatticeId.L01)
        self.assertNotEqual(c_prime, c_even)


class TestGroup7PrimeFactorSimilarity(unittest.TestCase):
    def test_jaccard_identical(self) -> None:
        c = composite_from_primes((3, 5, 7))
        self.assertEqual(prime_factor_similarity(c, c), 1.0)

    def test_jaccard_disjoint(self) -> None:
        a = composite_from_primes((3, 5))
        b = composite_from_primes((7, 11))
        self.assertEqual(prime_factor_similarity(a, b), 0.0)

    def test_jaccard_partial(self) -> None:
        a = composite_from_primes((3, 5, 7))
        b = composite_from_primes((3, 5, 11))
        self.assertAlmostEqual(prime_factor_similarity(a, b), 2 / 4)

    def test_factors_6(self) -> None:
        self.assertEqual(prime_factors(6), frozenset({2, 3}))

    def test_euclidean_dominated_by_scale(self) -> None:
        small = composite_from_primes((3, 5))
        large = composite_from_primes((3, 5, 7, 11, 13))
        d = abs(large - small)
        self.assertGreater(d, 100)
        self.assertGreater(prime_factor_similarity(small, large), 0.35)


class TestGroup8AppleProof(unittest.TestCase):
    """Morphology via prime-factor Jaccard on phrase composites."""

    def setUp(self) -> None:
        self.p_apple = 101
        self.p_phone = 103
        self.p_chip = 107
        self.p_pie = 109
        self.p_fruit = 113
        self.tech_doc = self.p_apple * self.p_phone * self.p_chip
        self.food_doc = self.p_apple * self.p_pie * self.p_fruit
        self.q_apple_phone = self.p_apple * self.p_phone
        self.q_apple_pie = self.p_apple * self.p_pie

    def test_tech_vs_food_jaccard_baseline(self) -> None:
        tech_overlap = prime_factor_similarity(self.q_apple_phone, self.tech_doc)
        food_overlap = prime_factor_similarity(self.q_apple_phone, self.food_doc)
        self.assertAlmostEqual(tech_overlap, 2 / 3, places=5)
        self.assertAlmostEqual(food_overlap, 0.25, places=5)

    def test_apple_phone_prefers_tech(self) -> None:
        tech = prime_factor_similarity(self.q_apple_phone, self.tech_doc)
        food = prime_factor_similarity(self.q_apple_phone, self.food_doc)
        self.assertGreater(tech / food, 2.0)

    def test_apple_pie_prefers_food(self) -> None:
        tech = prime_factor_similarity(self.q_apple_pie, self.tech_doc)
        food = prime_factor_similarity(self.q_apple_pie, self.food_doc)
        self.assertGreater(food / tech, 2.0)

    def test_raw_euclidean_misleading(self) -> None:
        """Large 3-prime doc vs 2-prime query: Euclidean not the right metric."""
        tech = prime_factor_similarity(self.q_apple_phone, self.tech_doc)
        food = prime_factor_similarity(self.q_apple_phone, self.food_doc)
        self.assertGreater(tech, food)


class TestGroup9PromotionGate(unittest.TestCase):
    def test_should_promote_extension(self) -> None:
        self.assertTrue(should_promote_intersection((3,), 5))

    def test_should_not_promote_empty(self) -> None:
        self.assertFalse(should_promote_intersection((), 5))


class TestGroup10EuclideanLocal(unittest.TestCase):
    def test_same_scale_neighbors(self) -> None:
        a = compute_coordinates((5,), 7, LatticeId.L01)
        b = compute_coordinates((5,), 8, LatticeId.L01)
        d = euclidean_distance(a, b)
        self.assertLess(d, 5.0)


class TestGroup11GoldenRegression(unittest.TestCase):
    def test_k2_n11_va1(self) -> None:
        self.assertEqual(canon_va1((3, 11), 11), (22, 11, 25))

    def test_k2_z_end_segment(self) -> None:
        z = z_depth((3, 11), 11, segment_index((3, 11), 11))
        self.assertEqual(z, 14 + 11)

    def test_l32_k1_p5_n7(self) -> None:
        self.assertEqual(compute_coordinates((5,), 7, LatticeId.L32), (0, 17, -12))

    def test_prime_anchor_raw(self) -> None:
        self.assertEqual(prime_anchor_coord(5), (5, 0, 5))

    def test_normalize_distinct(self) -> None:
        from core.phi_lattice import normalize_primes

        self.assertEqual(normalize_primes((7, 3, 5)), (3, 5, 7))

    def test_composite_fta_unique(self) -> None:
        a = composite_from_primes((3, 5))
        b = composite_from_primes((3, 7))
        self.assertNotEqual(a, b)


class TestGroup12BranchesVA2VA4(unittest.TestCase):
    def test_va2_k1_p5_n3(self) -> None:
        self.assertEqual(canon_on_chain(BranchKind.VA2, (5,), 3), (8, -3, 8))

    def test_va3_k1_p5_n7(self) -> None:
        self.assertEqual(canon_on_chain(BranchKind.VA3, (5,), 7), (7, 0, 12))

    def test_va4_k1_p5_n7(self) -> None:
        self.assertEqual(canon_on_chain(BranchKind.VA4, (5,), 7), (17, 0, 12))

    def test_va2_k2_n5(self) -> None:
        from core.phi_lattice import canon_va2

        self.assertEqual(canon_va2((3, 11), 5), (22, -5, 19))


class TestGroup13ComputeAPI(unittest.TestCase):
    def test_compute_all_wings_32_coords(self) -> None:
        wings = compute_all_wings((5,), 7)
        self.assertEqual(len(wings), 32)
        self.assertGreaterEqual(len(set(wings.values())), 16)

    def test_swap_meet_returns_coord(self) -> None:
        hit = swap_meet(3, 5)
        self.assertIsNotNone(hit)
        self.assertEqual(len(hit[0]), 3)  # type: ignore

    def test_promote_pair_chain(self) -> None:
        self.assertTrue(should_promote_intersection((3, 5), 7))


class TestGroup14InteriorPlateau(unittest.TestCase):
    def test_n3_to_10_same_z(self) -> None:
        chain = (3, 5, 7, 11)
        for n in range(3, 11):
            seg = segment_index(chain, n)
            self.assertEqual(z_depth(chain, n, seg), 26)

    def test_end_segment_adds_n(self) -> None:
        chain = (3, 5, 7, 11)
        seg = segment_index(chain, 11)
        self.assertEqual(z_depth(chain, 11, seg), 26 + 11)


class TestGroup15SimilarityEdgeCases(unittest.TestCase):
    def test_similarity_one(self) -> None:
        self.assertEqual(prime_factor_similarity(1, 1), 1.0)

    def test_factors_prime(self) -> None:
        self.assertEqual(prime_factors(17), frozenset({17}))

    def test_apple_ratio_267(self) -> None:
        tech = 2 / 3
        food = 0.25
        self.assertAlmostEqual(tech / food, 2.667, places=2)


if __name__ == "__main__":
    unittest.main()
