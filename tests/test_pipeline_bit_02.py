"""BIT 2 gate — quantized attractor keys κ and neighborhoods."""
import unittest

from aethos_lattice import BranchKind
from aethos_physics import SpacetimeCell
from pipeline.bit_02_attractor_key import (
    DEFAULT_QUANTIZE,
    attractor_neighbors,
    kappa,
    kappa_at_branch,
    kappa_branch_fan,
    kappa_from_cell,
    verify_bit02_gate,
    verify_bit02_rotation_gate,
)


class TestBit02AttractorKey(unittest.TestCase):
    def test_triple_witness_kappa(self):
        cell = SpacetimeCell.promote_witness((3, 5, 7), (3, 5))
        self.assertEqual(kappa_from_cell(cell), (12, 5, 15))
        self.assertEqual(
            kappa(complex(12, 5), 15.0),
            (12, 5, 15),
        )

    def test_neighbors_radius_zero(self):
        key = (12, 5, 15)
        self.assertEqual(attractor_neighbors(key, radius=0), {key})

    def test_neighbors_radius_one_count(self):
        key = (12, 5, 15)
        self.assertEqual(len(attractor_neighbors(key, radius=1)), 27)

    def test_gate_passes(self):
        ok, failures = verify_bit02_gate(quantize=DEFAULT_QUANTIZE)
        self.assertTrue(ok, failures)
        self.assertEqual(failures, [])

    def test_quantize_coarsens(self):
        cell = SpacetimeCell.promote_witness((3, 5, 7), (3, 5))
        fine = kappa_from_cell(cell, quantize=1.0)
        coarse = kappa_from_cell(cell, quantize=2.0)
        self.assertEqual(fine, (12, 5, 15))
        self.assertEqual(coarse, (6, 2, 8))

    def test_rotation_gate_passes(self):
        ok, failures = verify_bit02_rotation_gate()
        self.assertTrue(ok, failures)

    def test_branch_fan_four_distinct_kappas(self):
        fan = kappa_branch_fan((3, 5, 7), 5.0)
        self.assertEqual(len(fan), 4)
        self.assertEqual(len(set(fan)), 4)
        self.assertEqual(
            kappa_at_branch((3, 5, 7), 5.0, BranchKind.VA1),
            (12, 5, 15),
        )


if __name__ == "__main__":
    unittest.main()
