"""Tests for SpacetimeCell — 3D complex plane physics bridge."""
import math
import unittest

from aethos_lattice import BranchKind
from aethos_physics import (
    SpacetimeCell,
    SpacetimeIntervalKind,
    anchor_crossing_modulus_squared,
    layer0_lightlike_c,
    triple_meet_cells,
)


class TestLayer0(unittest.TestCase):
    def test_modulus_squared_is_2n2(self):
        for n in (1, 3, 7, 100):
            cell = SpacetimeCell.layer0(n)
            self.assertAlmostEqual(cell.modulus_squared, 2 * n * n)

    def test_lightlike_step_from_origin(self):
        c = layer0_lightlike_c()
        o = SpacetimeCell.layer0(0)
        one = SpacetimeCell.layer0(1)
        self.assertEqual(
            o.interval_kind_to(one, c=c),
            SpacetimeIntervalKind.LIGHTLIKE,
        )


class TestAnchorCrossing(unittest.TestCase):
    def test_displacement_2p2(self):
        self.assertEqual(anchor_crossing_modulus_squared(5), 50)
        self.assertEqual(anchor_crossing_modulus_squared(7), 98)


class TestPlateau(unittest.TestCase):
    def test_interior_lock(self):
        cell = SpacetimeCell.at((3, 5, 7, 11), 5)
        self.assertTrue(cell.is_interior_plateau())
        self.assertEqual(cell.expected_plateau_zeta(), 26.0)
        self.assertEqual(cell.zeta, 26.0)

    def test_boundary_not_plateau(self):
        cell = SpacetimeCell.at((3, 5, 7, 11), 11)
        self.assertFalse(cell.is_interior_plateau())


class TestTripleMeet(unittest.TestCase):
    def test_all_witnesses_same_spring(self):
        cells = triple_meet_cells(3, 5, 7)
        coords = {(c.z, c.zeta) for c in cells.values()}
        self.assertEqual(len(coords), 1)
        self.assertEqual(cells["ap"].n, 7.0)
        self.assertEqual(cells["aq"].n, 5.0)
        self.assertEqual(cells["pq"].n, 3.0)

    def test_promote_witness_matches(self):
        promoted = SpacetimeCell.promote_witness((3, 5, 7), (3, 5))
        self.assertEqual(promoted.n, 7.0)
        self.assertEqual(promoted.z, complex(12, 5))


class TestBranchPair(unittest.TestCase):
    def test_va1_va2_real_sum(self):
        va1 = SpacetimeCell.at((3, 5, 7), 5, BranchKind.VA1)
        va2 = SpacetimeCell.at((3, 5, 7), 5, BranchKind.VA2)
        s = va1.branch_pair_sum(va2)
        self.assertAlmostEqual(s.imag, 0.0)
        self.assertAlmostEqual(s.real, 30.0)


class TestIntervalKind(unittest.TestCase):
    def test_spacelike_with_c1_on_layer0(self):
        o = SpacetimeCell.layer0(0)
        one = SpacetimeCell.layer0(1)
        self.assertEqual(
            o.interval_kind_to(one, c=1.0),
            SpacetimeIntervalKind.SPACELIKE,
        )

    def test_timelike_when_c_exceeds_sqrt3(self):
        o = SpacetimeCell.layer0(0)
        one = SpacetimeCell.layer0(1)
        self.assertEqual(
            o.interval_kind_to(one, c=2.0),
            SpacetimeIntervalKind.TIMELIKE,
        )

    def test_interval_squared_formula(self):
        a = SpacetimeCell.layer0(3)
        b = SpacetimeCell.layer0(5)
        dn = 2.0
        dz = b.z - a.z
        expected = dn**2 - abs(dz) ** 2 - (b.zeta - a.zeta) ** 2
        self.assertAlmostEqual(a.interval_squared_to(b, c=1.0), expected)


if __name__ == "__main__":
    unittest.main()
