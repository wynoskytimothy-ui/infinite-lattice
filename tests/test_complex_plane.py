"""Tests for aethos_complex_plane — 3D complex plane formalism."""
import unittest

from aethos_complex_plane import (
    ComplexPlane3D,
    all_branch_phases,
    canon_complex,
    derive_va1_closed_form,
    equalize_witness,
    imaginary_start,
    missing_member,
    swap_meet,
    triple_equalization,
    wing_transform,
)
from aethos_lattice import BranchKind
from aethos_sequences import SequenceKind, canon_on_chain, make_chain


class TestImaginaryStart(unittest.TestCase):
    def test_n_plus_ni(self):
        for n in (0, 1, 3, 7):
            psi = imaginary_start(n)
            self.assertEqual(psi.z, complex(n, n))
            self.assertEqual(psi.zeta, float(n))


class TestCountableChain(unittest.TestCase):
    def test_evens_matches_canon_on_chain(self):
        chain = make_chain(SequenceKind.EVENS, 4)
        n = 10.0
        psi = canon_complex(BranchKind.VA1, chain, n)
        exp = canon_on_chain(BranchKind.VA1, chain, n)
        self.assertEqual(psi.coord, exp)

    def test_va1_closed_form(self):
        chain = (3, 5, 7)
        for n in (3, 5, 7, 10):
            got = derive_va1_closed_form(chain, n)
            exp = canon_on_chain(BranchKind.VA1, chain, n)
            self.assertEqual(got, exp)


class TestTripleEqualization(unittest.TestCase):
    def test_all_pairs_same_node(self):
        eq = triple_equalization(3, 5, 7)
        coords = {psi.coord for _, psi in eq.values()}
        self.assertEqual(len(coords), 1)
        self.assertEqual(eq["ap"][0], 7.0)
        self.assertEqual(eq["aq"][0], 5.0)
        self.assertEqual(eq["pq"][0], 3.0)

    def test_missing_member(self):
        self.assertEqual(missing_member((3, 5, 7), (3, 5)), 7.0)
        self.assertEqual(missing_member((3, 5, 7), (3, 7)), 5.0)
        self.assertEqual(missing_member((3, 5, 7), (5, 7)), 3.0)

    def test_equalize_witness(self):
        n, psi = equalize_witness((3, 5, 7), (3, 5))
        self.assertEqual(n, 7.0)
        self.assertEqual(psi.coord, (12, 5, 15))


class TestSameNodeDifferentComplex(unittest.TestCase):
    def test_four_branches_differ_at_n5(self):
        phases = all_branch_phases((3, 5, 7), 5)
        zs = {psi.z for psi in phases.values()}
        self.assertEqual(len(zs), 4)
        zetas = {psi.zeta for psi in phases.values()}
        self.assertEqual(len(zetas), 1)


class TestSwapMeet(unittest.TestCase):
    def test_3_5_swap(self):
        left, right = swap_meet(3, 5)
        self.assertEqual(left.coord, right.coord)
        self.assertEqual(left.z, complex(8, 3))


class TestWingTransform(unittest.TestCase):
    def test_v1_v5_differ(self):
        chain = (3, 5, 7)
        v1 = wing_transform(BranchKind.VA1, chain, 5, 1)
        v5 = wing_transform(BranchKind.VA1, chain, 5, 5)
        self.assertNotEqual(v1.z, v5.z)
        self.assertEqual(v1.zeta, v5.zeta)


if __name__ == "__main__":
    unittest.main()
