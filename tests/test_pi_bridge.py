"""PARTIAL functor: pi lattice <-> 3D complex plane bridge."""
import unittest

from aethos_pi_bridge import (
    embed_pi_vertex_layer0,
    pi_branch_bits_to_wing_mask,
    pi_depth_to_transgressor,
    pi_dyadic_point,
    pi_k0_is_spring_i,
    pi_layer0_direction_matches,
    pi_unit_circle_address,
)
from aethos_complex_rotation import wing_index_from_mask


class TestPiBridge(unittest.TestCase):
    def test_layer0_pi_fourth_direction(self):
        self.assertTrue(pi_layer0_direction_matches())

    def test_pi_k0_is_spring_i(self):
        self.assertTrue(pi_k0_is_spring_i())

    def test_dyadic_i_is_unit_im(self):
        re, im = pi_dyadic_point(0, 1)
        self.assertAlmostEqual(re, 0.0, places=6)
        self.assertAlmostEqual(im, 1.0, places=6)

    def test_dyadic_pi_fourth(self):
        re, im = pi_dyadic_point(1, 1)
        self.assertAlmostEqual(re, im, places=6)  # 45 deg

    def test_embed_layer0_positive_n(self):
        psi = embed_pi_vertex_layer0(1, 1)
        self.assertAlmostEqual(psi.z.real, psi.z.imag)

    def test_branch_bits_to_wing(self):
        mask = pi_branch_bits_to_wing_mask((0, 1, 0))
        wing = wing_index_from_mask(mask)
        self.assertGreaterEqual(wing, 1)
        self.assertLessEqual(wing, 8)

    def test_depth_to_n(self):
        self.assertEqual(pi_depth_to_transgressor(3), 8.0)

    def test_pi_address_legs_bounded(self):
        x, y = pi_unit_circle_address(2, 0)
        self.assertLessEqual(x * x + y * y, 2.0)


if __name__ == "__main__":
    unittest.main()
