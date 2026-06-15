"""Tests for 32 sub-quadrant rotation rules."""
import unittest

from aethos_complex_rotation import (
    all_sub_quadrants,
    branch_fan_z,
    index_to_branch_wing,
    klein_four_on_z,
    match_lattice_apply,
    rotate_cycle,
    rotate_step,
    sub_quadrant_index,
    verify_klein_identity,
    wing_mask_from_index,
    wing_orbit_z,
)
from aethos_lattice import BranchKind


class TestKleinFour(unittest.TestCase):
    def test_identities(self):
        self.assertTrue(verify_klein_identity(12 + 5j))

    def test_four_distinct_spring_z(self):
        z0 = 12 + 5j
        orbit = klein_four_on_z(z0)
        self.assertEqual(len(set(orbit)), 4)


class TestWingMask(unittest.TestCase):
    def test_eight_wings_match_masks(self):
        for w in range(1, 9):
            m = wing_mask_from_index(w)
            self.assertGreaterEqual(m, 0)
            self.assertLess(m, 8)

    def test_mask_matches_transform(self):
        chain = (3, 5, 7)
        for b in BranchKind:
            for m in range(8):
                self.assertTrue(match_lattice_apply(b, chain, 5, m))


class Test32SubQuadrants(unittest.TestCase):
    def test_all_32_distinct_psi(self):
        sqs = all_sub_quadrants((3, 5, 7), 5)
        self.assertEqual(len(sqs), 32)
        keys = {(sq.psi.z, sq.psi.zeta) for sq in sqs}
        self.assertEqual(len(keys), 32)

    def test_rotate_cycle_visits_32(self):
        indices = [k for k, _ in rotate_cycle((3, 5, 7), 5)]
        self.assertEqual(len(indices), 32)
        self.assertEqual(len(set(indices)), 32)

    def test_index_roundtrip(self):
        for k in range(32):
            b, w = index_to_branch_wing(k)
            self.assertEqual(sub_quadrant_index(b, w), k)

    def test_rotate_step_mod32(self):
        self.assertEqual(rotate_step(31, 1), 0)
        self.assertEqual(rotate_step(0, -1), 31)


class TestBranchFan(unittest.TestCase):
    def test_four_branches_at_v1(self):
        fan = branch_fan_z((3, 5, 7), 5, 1)
        self.assertEqual(len(fan), 4)
        self.assertEqual(len(set(fan.values())), 4)

    def test_va1_wing_orbit_four_z(self):
        orbit = wing_orbit_z((3, 5, 7), 5, BranchKind.VA1)
        self.assertEqual(len(set(orbit.values())), 4)


if __name__ == "__main__":
    unittest.main()
